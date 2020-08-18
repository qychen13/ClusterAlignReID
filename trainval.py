import os
import torch

from arguments import ArgumentsTrainVal
from config import get_config
from datasets import construct_dataset
from models import construct_model
from utils.construct_engine import construct_engine


def main():
    ############ arguments setup #############
    args = ArgumentsTrainVal().parse_args()
    print('***********************Arguments***********************')
    print(args)

    ############ get configuration info #############
    config = get_config(args)
    print('***********************Configurations***********************')
    print(config)

    ############ checkpoints and logs directory setup##############
    checkpoint_dir = os.path.join('checkpoints', args.check_log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir

    log_dir = os.path.join('logs', args.check_log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    ########### get model setup ############
    model = construct_model(args, config)
    print('***********************Model************************')
    print(model)
    # prepare the model before restoring and calling optimzier constrcutor
    if args.gpu_ids is not None:
        model.cuda(args.gpu_ids[0])

    ############ optimizer setup ###########
    optimizer = config.optimizer_func(model)

    ########### serialization of the running ###########
    if args.restore_file is None:
        # move initialization into the model constuctor __init__
        # config.initialization_func(model)
        pass
    else:
        if args.gpu_ids is None:
            checkpoint = torch.load(args.restore_file)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu_ids[0])
            checkpoint = torch.load(args.restore_file, map_location=loc)

        if 'new-optim' in args.version:
            print('==> Reload weights from {}'.format(args.restore_file))
            ckpt = checkpoint
            if 'state_dict' in checkpoint:
                ckpt = checkpoint['state_dict']
            model.load_state_dict(ckpt)
        else:
            if args.resume_iteration == 0:
                print('==> Transfer model weights from {}'.format(args.restore_file))
                if 'external-bnneck' in args.model_name:
                    feature_extractor = model.base
                else:
                    feature_extractor = model.feature_extractor
                msg = feature_extractor.load_state_dict(
                    checkpoint, strict=False)
                print(msg)
            else:
                print('==> Resume checkpoint {}'.format(args.restore_file))
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                for group in optimizer.param_groups:
                    group['initial_lr'] = args.learning_rate
                    group['lr'] = args.learning_rate

    ############ dataset setup #############
    if 'id' in args.version:
        train_iterator, val_iterator, query_iterator, gallary_iterator, id_iterator = construct_dataset(
        args, config)
    else:
        train_iterator, val_iterator, query_iterator, gallary_iterator = construct_dataset(
            args, config)
        id_iterator = None

    ############ learning rate scheduler setup ############
    # TODO: add lr_scheduler state_dict
    if config.lr_scheduler_func:
        lr_scheduler = config.lr_scheduler_func(
            optimizer, **config.lr_scheduler_params)
        lr_scheduler_iter = None
    else:
        lr_scheduler = None
        lr_scheduler_iter = config.lr_scheduler_iter_func(len(train_iterator), optimizer)

    ############ engine setup ##############
    engine_args = dict(gpu_ids=args.gpu_ids,
                       network=model,
                       criterion=config.loss_func,
                       train_iterator=train_iterator,
                       validate_iterator=val_iterator,
                       optimizer=optimizer,
                       )
    engine = construct_engine(engine_args,
                              log_freq=args.log_freq,
                              log_dir=args.log_dir,
                              checkpoint_dir=checkpoint_dir,
                              checkpoint_freq=args.checkpoint_freq,
                              lr_scheduler=lr_scheduler,
                              lr_scheduler_iter=lr_scheduler_iter,
                              metric_dict=config.metric_dict,
                              query_iterator=query_iterator,
                              gallary_iterator=gallary_iterator,
                              id_feature_params=config.id_feature_params,
                              id_iterator=id_iterator,
                              test_params=config.test_params
                              )

    engine.resume(args.maxepoch, args.resume_epoch, args.resume_iteration)


if __name__ == '__main__':
    main()
