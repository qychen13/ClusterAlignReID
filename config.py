from collections import namedtuple, defaultdict
import os
import numpy as np
import torch
import torch.nn.functional as functional


def get_config(args):
    config = {}

    # TODO: add more configuration functions
    if args.flag == 'train':
        param_funcs = [get_dataset_config, get_loss_config,
                       get_metrics_config, get_optimizer_config, get_model_config, get_test_config]
    elif args.flag == 'test':
        param_funcs = [get_dataset_config, get_model_config]
    else:
        raise NotImplementedError

    params = {}
    for func in param_funcs:
        params.update(func(args))

    Configuration = namedtuple('Configuration', params.keys())
    config = Configuration(**params)

    return config


def get_dataset_config(args):
    # data pre-processing
    params = {}
    params = {'mean': [0.485, 0.456, 0.406],
              'std': [0.229, 0.224, 0.225],
              'resize_size': [256, 128],
              'crop_size': [256, 128],
              'pad_size': 10,
              'resize_interpolation': 3}

    if 'external-abd' in args.model_name:
        del params['resize_size'], params['pad_size']
        params['random_scale'] = True
        params['crop_size'] = (384, 128)

    elif 'external-mgn' in args.model_name: #TODO: validate the effectiveness
        del params['pad_size']
        del params['crop_size']
        params['resize_size'] = (384, 128)
        """
        del params['resize_size'], params['pad_size']
        params['random_scale'] = True
        params['crop_size'] = (384, 128)
        """

    elif 'pcb' in args.model_name:
        del params['pad_size']
        del params['crop_size']
        params['resize_size'] = [384, 192]

    if 'duke-large-input' in args.version:
        params['crop_size'] = [384, 192]
        params['resize_size'] = [384, 192]

    # sampler
    if 'triplet' in args.version:
        params['sampler'] = 'triplet'
        params['sampler_paras'] = dict(num_instances=4, batch_size=args.batch_size)
    else:
        params['sampler'] = 'random'

    # random erasing
    if 'erasing' in args.version:
        params['random_erasing_prob'] = 0.5
        params['random_erasing_mean'] = (0.0, 0.0, 0.0)
        if 'erasing-v1' in args.version:
            params['random_erasing_mean'] = (0.4914, 0.4822, 0.4465)
        elif 'erasing-v2' in args.version:
            params['random_erasing_mean'] = params['mean']

    if 'interpo-2' in args.version:
        params['resize_interpolation'] = 2


    return {'dataset_parameters': params}


def get_model_config(args):
    if args.dataset_name == 'market1501':
        # TODO: add num_classes
        num_classes = 751
    elif args.dataset_name == 'dukemtm':
        num_classes = 702
    elif args.dataset_name == 'msmt17':
        num_classes = None
    elif 'cuhk03' in args.dataset_name:
        num_classes = 767
    else:
        raise NotImplementedError

    if 'external' in args.model_name:
        if 'layumi' in args.model_name:
            if 'resnet50' in args.model_name:
                external_model_paras = dict(class_num=num_classes, stride=1)
            elif 'pcb' in args.model_name:
                external_model_paras = dict(class_num=num_classes)
            else:
                raise NotImplementedError
            return dict(external_model_paras=external_model_paras)
        else:
            external_model_paras = dict(num_classes=num_classes)
            return dict(external_model_paras=external_model_paras)

    else:
        if 'resnet50' in args.model_name:
            last_stride = 2
            in_planes = 2048
            if 'ls1' in args.version:
                last_stride = 1
            feature_extractor_paras = dict(
                num_classes=1000, last_stride=last_stride)
            if args.model_name == 'resnet50-neck-v1':
                in_planes = 512
        else:
            raise NotImplementedError

        classifier_paras = dict(num_classes=num_classes, in_planes=in_planes)

        return dict(feature_extractor_paras=feature_extractor_paras, classifier_paras=classifier_paras)

def get_test_config(args):
    test_params = dict(method='euclidean', flips=False, reranking=False)
    if 'test-bnneck' in args.version:
        test_params['method'] = 'euclidean-normal'
    if 'flips' in args.version:
        test_params['flips'] = True

    return {'test_params':test_params}


def get_metrics_config(args):
    import utils.metrics

    # default metrics
    metric_dict = {'CrossEntropyLoss': utils.metrics.KeyLossMetric(None),
                   'Top1ACC': utils.metrics.ClassAccuracy(top_k=1)}
    if 'triplet' in args.version:
        metric_dict = {'TripletLoss': utils.metrics.KeyLossMetric('triplet_loss'),
                       'CrossEntropyLoss': utils.metrics.KeyLossMetric('cross_entropy_loss'),
                       'TotalLoss': utils.metrics.KeyLossMetric('total_loss'),
                       'Top1ACC': utils.metrics.ClassAccuracy(top_k=1),
                       'DistAP': utils.metrics.KeyLossMetric('dist_ap'),
                       'DistAN': utils.metrics.KeyLossMetric('dist_an')}

    if 'id' in args.version:
        metric_dict.update({
            'IDLoss': utils.metrics.KeyLossMetric('id_loss'),
            'CrossEntropyLoss': utils.metrics.KeyLossMetric('cross_entropy_loss'),
            'TotalLoss': utils.metrics.KeyLossMetric('total_loss'),
        })
        if 'intra-inter-id' in args.version:
            metric_dict.update({
                'IDAp': utils.metrics.KeyLossMetric('id_ap'),
                'IDAn': utils.metrics.KeyLossMetric('id_an'),
            })

    if 'external-abd' in args.model_name:
        del metric_dict['Top1ACC']
        metric_dict['GlobalCrossEntropyLoss'] = utils.metrics.KeyLossMetric(
            'global_xe_loss')
        metric_dict['ABDCrossEntropyLoss1'] = utils.metrics.KeyLossMetric(
            'abd_xe_loss_1')
        metric_dict['ABDCrossEntropyLoss2'] = utils.metrics.KeyLossMetric(
            'abd_xe_loss_2')
        metric_dict['GlobalTop1ACC'] = utils.metrics.ClassAccuracy(
            top_k=1, index=0)
        metric_dict['ABDTop1ACC1'] = utils.metrics.ClassAccuracy(
            top_k=1, index=1)
        metric_dict['ABDTop1ACC2'] = utils.metrics.ClassAccuracy(
            top_k=1, index=2)
    elif 'external-layumi-pcb' in args.model_name:
        # TODO: add part accuracy
        pcb_metric_dict = {}
        for key, val in metric_dict.items():
            if isinstance(val, utils.metrics.KeyLossMetric):
                for i in range(6):
                    pcb_metric_dict['{}_{}'.format(key, i)] = utils.metrics.KeyLossMetric(
                        '{}_{}'.format(val.key, i))
        metric_dict.update(pcb_metric_dict)
    elif 'pcb' in args.model_name:
        pcb_metric_dict = {}
        for key, val in metric_dict.items():
            if isinstance(val, utils.metrics.KeyLossMetric):
                for i in range(3):
                    pcb_metric_dict['{}_{}'.format(key, i)] = utils.metrics.KeyLossMetric(
                        '{}_{}'.format(val.key, i))
        metric_dict = pcb_metric_dict
    elif 'external-mgn' in args.model_name:
        metric_dict = {'TripletLoss': utils.metrics.KeyLossMetric('triplet_loss'),
                       'CrossEntropyLoss': utils.metrics.KeyLossMetric('cross_entropy_loss'),
                       'TotalLoss': utils.metrics.KeyLossMetric('total_loss'),
                       'Top1ACC': utils.metrics.ClassAccuracy(top_k=1)}
        if 'id' in args.version:
            if 'triplet' not in args.version:
                del metric_dict['TripletLoss']
            metric_dict['IDLoss'] = utils.metrics.KeyLossMetric('id_loss')
            if 'intra-inter-id' in args.version:
                metric_dict.update({
                    'IDAp': utils.metrics.KeyLossMetric('id_ap'),
                    'IDAn': utils.metrics.KeyLossMetric('id_an'),
                })

    return dict(metric_dict=metric_dict)


def get_loss_config(args):
    import utils.loss
    # defualt loss function

    if 'external-abd' in args.model_name:
        def cross_entropy_loss_func(output, target):
            losses = {}
            assert len(output['logits']) == 3
            losses['global_xe_loss'] = functional.cross_entropy(
                output['logits'][0], target['pid'])
            for i in range(1, 3):
                losses['abd_xe_loss_{}'.format(i)] = functional.cross_entropy(
                    output['logits'][i], target['pid'])
            losses['total_loss'] = (
                losses['global_xe_loss'] + losses['abd_xe_loss_1'] + losses['abd_xe_loss_2'])/3
            return losses
    elif 'label-smooth' in args.version:
        if args.dataset_name == 'market1501':
            # TODO: add num_classes
            num_classes = 751
        elif args.dataset_name == 'dukemtm':
            num_classes = 702
        elif args.dataset_name == 'msmt17':
            num_classes = None
        elif 'cuhk03' in args.dataset_name:
            num_classes = 767
        else:
            raise NotImplementedError
        label_smooth_loss = utils.loss.CrossEntropyLabelSmooth(num_classes=num_classes)
        def cross_entropy_loss_func(output, target):
            return label_smooth_loss(output['logits'], target['pid'])

    else:
        def cross_entropy_loss_func(output, target):
            return functional.cross_entropy(output['logits'], target['pid'])

    loss_func = cross_entropy_loss_func

    if 'triplet' in args.version:
        triplet_loss_func = utils.loss.TripletLoss(margin=0.3)
        if 'triplet0' in args.version:
            triplet_loss_func = utils.loss.TripletLoss(margin=0)

        def triplet_cross_entropy_loss(output, target):
            triplet_loss = triplet_loss_func(output['features'], target['pid'])
            cross_entropy_loss = cross_entropy_loss_func(output, target)
            total_loss = triplet_loss['loss'] + cross_entropy_loss

            return dict(triplet_loss=triplet_loss['loss'], cross_entropy_loss=cross_entropy_loss, total_loss=total_loss,
                        dist_ap=triplet_loss['dist_ap'], dist_an=triplet_loss['dist_an'])
        loss_func = triplet_cross_entropy_loss

    if 'mgn' in args.model_name:
        def mgn_loss(output, target):
            total_loss = 0
            # cross entropy loss
            cross_entropy_loss = 0
            for i in range(8):
                output_i = {'logits': output['logits'][i]}
                cross_entropy_loss += cross_entropy_loss_func(output_i, target)
            triplet_loss = 0
            for i in range(3):
                triplet_loss+= triplet_loss_func(output['features'][:,:,i], target['pid'])['loss']

            total_loss = triplet_loss + cross_entropy_loss

            return dict(triplet_loss=triplet_loss, cross_entropy_loss=cross_entropy_loss, total_loss=total_loss)
        loss_func = mgn_loss

    if 'id' in args.version:
        id_weight = 1.0

        # ID loss function
        id_loss_func = functional.mse_loss
        if 'id-l1' in args.version:
            id_loss_func = functional.l1_loss
        elif 'intra-inter-id' in args.version:
            if 'intra-inter-id-v1' in args.version:
                id_loss_func = utils.loss.TripletIDLoss(margin=0, ranking_loss=False)
            elif 'intra-inter-id-v2' in args.version:
                id_loss_func = utils.loss.TripletIDLoss(ranking_loss=True)
            elif 'intra-inter-id-v3' in args.version:
                id_loss_func = utils.loss.TripletIDLoss(margin=0.5, ranking_loss=True)
            elif 'intra-inter-id-v4' in args.version:
                id_loss_func = utils.loss.TripletIDLoss(margin=0.3, ranking_loss=True)
            else:
                id_loss_func = utils.loss.TripletIDLoss(margin=0)
            if 'id-on' in args.version:
                id_loss_func.id_loss=True

        # ID loss weight
        if 'id-w001' in args.version:
            id_weight = 0.01
        elif 'id-w01' in args.version:
            id_weight = 0.1
        elif 'id-w100' in args.version:
            id_weight = 100
        elif 'id-w10' in args.version:
            id_weight = 10
        elif 'id-w0' in args.version: # comparison only
            id_weight = 0

        def cross_entropy_id_loss(output, target):
            xe_loss = cross_entropy_loss_func(output, target)
            if isinstance(xe_loss, dict):
                loss_dict = xe_loss
                loss_dict['cross_entropy_loss'] = loss_dict['total_loss']
            else:
                loss_dict = dict()
                loss_dict['cross_entropy_loss'] = xe_loss

            if 'id_features' in target or 'id_feature_dict' in target:
                if 'intra-inter-id' in args.version:
                    id_loss = id_loss_func(
                    output['features'], target['pid'], target['id_feature_dict'])
                    loss_dict['id_ap'] = id_loss['id_ap']
                    loss_dict['id_an'] = id_loss['id_an']
                    id_loss = id_loss['id_loss']
                else:
                    id_loss = id_loss_func(
                        output['features'], target['id_features'])
                loss_dict['total_loss'] = id_weight * \
                    id_loss + loss_dict['cross_entropy_loss']
                if 'id-only' in args.version:
                    loss_dict['total_loss'] = id_weight * id_loss
            else:
                id_loss = np.array([0])  # NAN
                loss_dict['total_loss'] = loss_dict['cross_entropy_loss']
            loss_dict['id_loss'] = id_loss

            return loss_dict
        if 'triplet' in args.version:
            def cross_entropy_triplet_id_loss(output, target):
                loss_dict = cross_entropy_id_loss(output, target)
                triplet_loss = triplet_loss_func(
                    output['features'], target['pid'])
                loss_dict['triplet_loss'] = triplet_loss['loss']
                loss_dict['total_loss'] += triplet_loss['loss']
                loss_dict['dist_ap'] = triplet_loss['dist_ap']
                loss_dict['dist_an'] = triplet_loss['dist_an']
                return loss_dict
            loss_func = cross_entropy_triplet_id_loss
        else:
            loss_func = cross_entropy_id_loss

        if 'mgn' in args.model_name:
            def id_mgn_loss(output, target):
                loss_dict = defaultdict(lambda: 0)
                # cross entropy loss
                for i in range(8):
                    output_i = {'logits': output['logits'][i]}
                    loss_dict['cross_entropy_loss'] += cross_entropy_loss_func(output_i, target)

                total_id_loss = 0
                for i in range(3):
                    if 'id_features' in target or 'id_feature_dict' in target:
                        if 'intra-inter-id' in args.version:
                            id_loss = id_loss_func(
                            output['features'][:,:,i], target['pid'], target['id_feature_dict'][:,:,i])
                            loss_dict['id_ap'] += id_loss['id_ap']
                            loss_dict['id_an'] += id_loss['id_an']
                            id_loss = id_loss['id_loss']
                        else:
                            id_loss = id_loss_func(output['features'][:,:,i], target['id_features'][:,:,i])

                        total_id_loss+=id_loss
                    else:
                        total_id_loss = np.array([0])  # NAN
                        break
                loss_dict['id_loss'] = total_id_loss
                loss_dict['total_loss'] = id_weight * total_id_loss + loss_dict['cross_entropy_loss']
                if 'id-only' in args.version:
                    loss_dict['total_loss'] = id_weight * total_id_loss


                return loss_dict
            loss_func = id_mgn_loss

    if 'pcb' in args.model_name:
        def pcb_callback(output, target):
            num_parts = len(output['logits'])
            loss_dict = defaultdict(lambda: 0)
            id_features = target['id_features'] if 'id_features' in target else None
            for i in range(num_parts):
                if id_features is not None:
                    target['id_features'] = id_features[:, :, i]
                output_i = {key: val[i] if isinstance(
                    val, list) else val[:, :, i] for key, val in output.items()}
                loss_dict_i = loss_func(output_i, target)
                # cumulate the loss
                for key in loss_dict_i:
                    loss_dict[key] += loss_dict_i[key]
                loss_dict_i = {'{}_{}'.format(
                    key, i): val for key, val in loss_dict_i.items()}
                loss_dict.update(loss_dict_i)
            """
            # normalize each part features
            output['logits'] = functional.softmax(torch.stack(output['logits'], dim=2), dim=2)
            output['logits'] = torch.view(output['logits'].shape[0], output['logits'].shape[1], -1)
            loss_dict.update(loss_func(output, target))
            """
            return loss_dict

        return dict(loss_func=pcb_callback)



    return dict(loss_func=loss_func)


def get_optimizer_config(args):
    import torch.optim
    import utils.lr_scheduler

    # defualt optimizer and scheduler
    if 'nesterov' in args.version:
        nesterov = True
    else:
        nesterov = False
    optimizer_params = dict(lr=args.learning_rate, momentum=0.9,
                            weight_decay=5e-4, nesterov=nesterov)  # fix issue of weight decay

    optimizer = torch.optim.SGD
    if 'adam' in args.version:
        optimizer = torch.optim.Adam
        optimizer_params = dict(lr=args.learning_rate, weight_decay=5e-4)

    if 'diff-lr' in args.version:
        if 'diff-lr-v1' in args.version:
            if 'layumi' in args.model_name:
                if 'resnet50' in args.version:
                    def construct_optimzier(model):
                        ignored_params = list(
                            map(id, model.classifier.parameters()))
                        base_params = filter(lambda p: id(
                            p) not in ignored_params, model.parameters())
                        return optimizer([
                            {'params': base_params, 'lr': 0.1 *
                                optimizer_params['lr']},
                            {'params': model.classifier.parameters(
                            ), 'lr': optimizer_params['lr']}
                        ], **optimizer_params)
                elif 'pcb' in args.model_name:
                    def construct_optimzier(model):
                        ignored_params = list(
                            map(id, model.model.fc.parameters()))
                        ignored_params += (list(map(id, model.classifier0.parameters()))
                                           + list(map(id, model.classifier1.parameters()))
                                           + list(map(id, model.classifier2.parameters()))
                                           + list(map(id, model.classifier3.parameters()))
                                           + list(map(id, model.classifier4.parameters()))
                                           + list(map(id, model.classifier5.parameters()))
                                           #+list(map(id, model.classifier6.parameters() ))
                                           #+list(map(id, model.classifier7.parameters() ))
                                           )
                        base_params = filter(lambda p: id(
                            p) not in ignored_params, model.parameters())
                        optimizer_ft = optimizer([
                            {'params': base_params, 'lr': 0.1 *
                                optimizer_params['lr']},
                            {'params': model.model.fc.parameters(
                            ), 'lr': optimizer_params['lr']},
                            {'params': model.classifier0.parameters(
                            ), 'lr': optimizer_params['lr']},
                            {'params': model.classifier1.parameters(
                            ), 'lr': optimizer_params['lr']},
                            {'params': model.classifier2.parameters(
                            ), 'lr': optimizer_params['lr']},
                            {'params': model.classifier3.parameters(
                            ), 'lr': optimizer_params['lr']},
                            {'params': model.classifier4.parameters(
                            ), 'lr': optimizer_params['lr']},
                            {'params': model.classifier5.parameters(
                            ), 'lr': optimizer_params['lr']},
                            #{'params': model.classifier6.parameters(), 'lr': 0.01},
                            #{'params': model.classifier7.parameters(), 'lr': 0.01}
                        ], **optimizer_params)
                        return optimizer_ft
            else:
                def construct_optimzier(model):
                    ignored_params = list(
                        map(id, model.feature_extractor.fc.parameters()))
                    base_params = filter(lambda p: id(
                        p) not in ignored_params, model.feature_extractor.parameters())
                    return optimizer([
                        {'params': base_params, 'lr': 0.1 *
                            optimizer_params['lr']},
                        {'params': model.classifier.parameters(
                        ), 'lr': optimizer_params['lr']},
                        {'params': model.feature_extractor.fc.parameters(), 'lr': optimizer_params['lr']}], **optimizer_params)
        else:
            def construct_optimzier(model):
                return optimizer([{'params': model.feature_extractor.parameters(), 'lr': 0.1*optimizer_params['lr']},
                                  {'params': model.classifier.parameters(), 'lr': optimizer_params['lr']}], **optimizer_params)
    else:
        def construct_optimzier(model):
            return optimizer(model.parameters(), **optimizer_params)

    optimizer_func = construct_optimzier
    lr_scheduler_func = torch.optim.lr_scheduler.StepLR

    if args.resume_iteration == 0 or 'new-optim' in args.version:
        last_epoch = -1
        last_iteration = -1
    else:
        last_epoch = args.resume_epoch
        last_iteration = args.resume_iteration
    lr_scheduler_params = dict(
        step_size=40, gamma=0.1, last_epoch=last_epoch)
    lr_scheduler_iter_func = None

    if 'const-lr' in args.version:
        # const learning rate implemeted with a large step size
        lr_scheduler_params['step_size'] = 1000
    elif 'warm-up-v1' in args.version:
        lr_scheduler_func = utils.lr_scheduler.WarmupMultiStepLR
        lr_scheduler_params = dict(milestones=[
                                   40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=10, warmup_method='linear', last_epoch=last_epoch)

    elif 'warm-up-v2' in args.version:
        lr_scheduler_func = utils.lr_scheduler.WarmupMultiStepLR
        lr_scheduler_params = dict(milestones=[
                                   40, 80], gamma=0.1, warmup_factor=0.1, warmup_iters=5, warmup_method='linear', last_epoch=last_epoch)

    elif 'warm-up-v3' in args.version:
        lr_scheduler_func = None
        lr_scheduler_params = None

        def lr_scheduler_iter(iterations_per_epoch, *args, **kwarg):
            lr_scheduler_iter_params = dict(milestones=[40 * iterations_per_epoch, 80 * iterations_per_epoch],
                                            gamma=0.1, warmup_factor=0.1, warmup_iters=5 * iterations_per_epoch,
                                            warmup_method='linear', last_epoch=last_iteration)
            kwarg.update(lr_scheduler_iter_params)
            return utils.lr_scheduler.WarmupMultiStepLR(*args, **kwarg)

        lr_scheduler_iter_func = lr_scheduler_iter
    elif 'cont-id-lr' in args.version:
        lr_scheduler_func = torch.optim.lr_scheduler.MultiStepLR
        lr_scheduler_params = dict(milestones=[140], gamma=0.1)

    id_feature_params = dict(warm_up_epochs=None)
    if 'id' in args.version:
        # id loss type: impact the injection of extra info in targets
        if 'intra-inter-id' in args.version:
            id_feature_params['id_loss'] = 'intra-inter'
        else:
            id_feature_params['id_loss'] = 'intra'

        # id warm up epoch
        if 'id-e10' in args.version:
            id_feature_params['warm_up_epochs'] = 10
        elif 'id-e20' in args.version:
            id_feature_params['warm_up_epochs'] = 20
        elif 'id-e40' in args.version:
            id_feature_params['warm_up_epochs'] = 40
        elif 'id-e0' in args.version:
            id_feature_params['warm_up_epochs'] = 0
        else:
            id_feature_params['warm_up_epochs'] = -1

        # id calculation method
        if 'weight-prob' in args.version:
            id_feature_params['method'] = 'weight-prob'
        else:
            id_feature_params['method'] = 'avg'

        # how frequently to update the id anchor
        if 'id-update-iter' in args.version:
            id_feature_params['update_freq'] = 'iteration'
        elif 'id-update-first' in args.version:
            id_feature_params['update_freq'] = 'first'
        else: # default option
            id_feature_params['update_freq'] = 'epoch'


    return dict(optimizer_func=optimizer_func,
                lr_scheduler_func=lr_scheduler_func,
                lr_scheduler_params=lr_scheduler_params,
                lr_scheduler_iter_func=lr_scheduler_iter_func,
                id_feature_params=id_feature_params)
