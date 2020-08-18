from .baseline import Baseline


def resnet50(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50-19c8e357.pth', 'bnneck', 'after', 'resnet50', 'imagenet')


def resnet50v1(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50-19c8e357.pth', 'bnneck', 'after-v1', 'resnet50', 'imagenet')

def resnet50v2(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50-19c8e357.pth', 'bnneck', 'before', 'resnet50', 'imagenet')

def resnet50_pcb(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50-19c8e357.pth', 'bnneck-pcb', 'after', 'resnet50', 'imagenet')

def resnet50_pcb_v1(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50-19c8e357.pth', 'bnneck-pcb-v1', 'after', 'resnet50', 'imagenet')

def resnet50_ibn_a(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50_ibn_a_01.pth.tar', 'bnneck', 'after', 'resnet50_ibn_a', 'imagenet')

def resnet50_ibn_av1(num_classes):
    return Baseline(num_classes, 1, '../modelzoo/resnet50_ibn_a_01.pth.tar', 'bnneck', 'after-v1', 'resnet50_ibn_a', 'imagenet')
