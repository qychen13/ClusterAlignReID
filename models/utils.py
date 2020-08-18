from .model_base import PersonReidModel, BaselineClassifier, BNNeckClassifer, BottleNeckClassifier, PersonReidModelNeck, resnet50_feature_extractor, resnet50_feature_extractor_v1
from . import model_layumi
from . import model_abd
from . import model_strong_baseline
from . import model_mgn

def construct_model(args, config):
    if 'external' in args.model_name:
        model = external_model_factory[args.model_name](**config.external_model_paras)
        return model

    feacture_extractor = feature_extractor_factory[args.model_name](
        **config.feature_extractor_paras)
    classifier = classifier_factory[args.model_name](**config.classifier_paras)
    if 'neck-fs' in args.version:  # test use only
        model = PersonReidModelNeck(feacture_extractor, classifier)
    else:
        model = PersonReidModel(feacture_extractor, classifier)

    return model


feature_extractor_factory = {'resnet50': resnet50_feature_extractor,
                             'resnet50-bnneck': resnet50_feature_extractor,
                             'resnet50-neck': resnet50_feature_extractor,
                             'resnet50-neck-v1': resnet50_feature_extractor_v1,
                             }
classifier_factory = {'resnet50': BaselineClassifier,
                      'resnet50-bnneck': BNNeckClassifer,
                      'resnet50-neck': BottleNeckClassifier,
                      'resnet50-neck-v1': BaselineClassifier}
external_model_factory = {
    'external-layumi-resnet50': model_layumi.ft_net,
    'external-layumi-pcb': model_layumi.PCB,
    'external-abd-resnet50': model_abd.resnet50,
    'external-bnneck': model_strong_baseline.resnet50,
    'external-bnneckv1': model_strong_baseline.resnet50v1,
    'external-before': model_strong_baseline.resnet50v2,
    'external-bnneck-pcb': model_strong_baseline.resnet50_pcb,
    'external-bnneck-pcbv1': model_strong_baseline.resnet50_pcb_v1,
    'external-mgn': model_mgn.MGN,
    'external-bnneck-ibn-a': model_strong_baseline.resnet50_ibn_a,
    'external-bnneck-ibn-a-v1': model_strong_baseline.resnet50_ibn_av1
}
