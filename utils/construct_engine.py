import time
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .engine import Engine
from .evaluation import test
from .center import calculate_id_features, update_id_features


def construct_engine(engine_args, log_freq, log_dir, checkpoint_dir, checkpoint_freq, lr_scheduler, lr_scheduler_iter, metric_dict, query_iterator, gallary_iterator, id_feature_params, id_iterator, test_params):
    if lr_scheduler is not None and lr_scheduler_iter is not None:
        raise RuntimeError(
            'Either lr_scheduler or lr_scheduler_iter can be used')
    engine = Engine(**engine_args)

    # tensorboard log setting
    writer = SummaryWriter(log_dir)

    # id features
    global id_features
    id_features = None

    ########## helper functions ##########
    # TODO: serialization of id feature training
    def save_model(state, filename):
        torch.save({
            'state_dict': state['network'].state_dict(),
            'optimizer': state['optimizer'].state_dict(),
            'metrics': {key: metric_dict[key].value() for key in metric_dict},
            'id_features': id_features
        }, filename)

    def reset_metrics():
        for key in metric_dict:
            metric_dict[key].reset()

    def update_metrics(state):
        paras = dict(logits=state['output'],
                     target=state['sample'][1], loss=state['loss'])
        with torch.no_grad():
            for key in metric_dict:
                metric_dict[key](**paras)

    def wrap_data(state):
        if state['gpu_ids'] is not None:
            if len(state['gpu_ids']) == 1:
                state['sample'][0] = state['sample'][0].cuda(
                state['gpu_ids'][0], non_blocking=True)
            for key in state['sample'][1]:
                if isinstance(state['sample'][1][key], list):
                    continue
                state['sample'][1][key] = state['sample'][1][key].cuda(
                    state['gpu_ids'][0], non_blocking=True)

    ########## callback functions ##########
    def on_start(state):
        reset_metrics()
        if state['train']:
            if state['gpu_ids'] is None:
                print('Training/Validating without gpus ...')
            else:
                if not torch.cuda.is_available():
                    raise RuntimeError('Cuda is not available')
                print(
                    'Training/Validating on gpu: {}'.format(state['gpu_ids']))

            if state['iteration'] == 0:
                filename = os.path.join(checkpoint_dir, 'init_model.pth.tar')
                save_model(state, filename)
        else:
            print('-------------Start Validation at {} For Epoch {}-------------'.format(
                time.strftime('%c'), state['epoch']))

    def on_start_epoch(state):
        print('-------------Start Training at {} For Epoch {}------------'.format(
            time.strftime('%c'), state['epoch']))
        if lr_scheduler is not None:
            scheduler = lr_scheduler
        else:
            scheduler = lr_scheduler_iter
            lr_scheduler_iter.step(state['iteration'])

        for i, lr in enumerate(scheduler.get_lr()):
            writer.add_scalar(
                'global/learning_rate_{}'.format(i), lr, state['epoch'])
        reset_metrics()
        if id_feature_params['warm_up_epochs'] is not None and state['epoch'] > id_feature_params['warm_up_epochs']:
            global id_features
            if id_feature_params['update_freq'] == 'epoch' or id_features is None:
                id_features = calculate_id_features(
                    state['network'], id_iterator, state['gpu_ids'], method=id_feature_params['method'])

    def on_end_sample(state):
        wrap_data(state)
        global id_features
        if state['train'] and id_features is not None:  # add id feature as label
            state['sample'][1]['id_features'] = id_features[[
                state['sample'][1]['pid']]]
            state['sample'][1]['id_feature_dict'] = id_features

        if lr_scheduler_iter is not None:
            lr_scheduler_iter.step(state['iteration'])
            for i, lr in enumerate(lr_scheduler_iter.get_lr()):
                writer.add_scalar(
                    'training_iter/learning_rate_{}'.format(i), lr, state['iteration'])

    def on_end_forward(state):
        update_metrics(state)
        global id_features
        if state['train'] and id_features is not None and id_feature_params['update_freq'] == 'iteration':
            id_features = update_id_features(state['output'], state['sample'][1])

    def on_end_update(state):
        if state['iteration'] % log_freq == 0:
            for key in metric_dict:
                writer.add_scalar('training_iter/{}'.format(key),
                                  metric_dict[key].value(), state['iteration'])

        if lr_scheduler_iter is not None:
            lr_scheduler_iter.step(state['iteration']+1)


    def on_end_epoch(state):
        for key in metric_dict:
            writer.add_scalar('training/{}'.format(key),
                              metric_dict[key].value(), state['epoch'])

        if (state['epoch'] + 1) % checkpoint_freq == 0:
            file_name = 'e{}t{}.pth.tar'.format(
                state['epoch'], state['iteration'])
            file_name = os.path.join(checkpoint_dir, file_name)
            save_model(state, file_name)
        # start testing
        t = time.strftime('%c')
        print(
            '*************************Start testing at {}**********************'.format(t))
        result = test(state['network'], query_iterator,
                      gallary_iterator, state['gpu_ids'],**test_params)

        for key in result:
            writer.add_scalar('test/{}'.format(key),
                              result[key], state['epoch'])

        # Note: adjust learning after calling optimizer.step() as required by update after pytorch 1.1.0
        if lr_scheduler is not None:
            lr_scheduler.step(state['epoch']+1)

    def on_end(state):
        t = time.strftime('%c')
        if state['train']:
            save_model(state, os.path.join(
                checkpoint_dir, 'final_model.pth.tar'))
            print(
                '*********************Training done at {}***********************'.format(t))
            writer.close()
        else:
            for key in metric_dict:
                writer.add_scalar('validation/{}'.format(key),
                                  metric_dict[key].value(), state['epoch'])

    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_sample'] = on_end_sample
    engine.hooks['on_end_forward'] = on_end_forward
    engine.hooks['on_end_update'] = on_end_update
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_end'] = on_end

    return engine
