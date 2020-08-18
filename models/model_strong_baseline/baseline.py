# Modified from reid-strong-baseline
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .resnet import ResNet, BasicBlock, Bottleneck
from .senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class PCBClassifier(nn.Module):
    def __init__(self, in_planes, num_classes, parts=6):
        super(PCBClassifier, self).__init__()
        self.parts = parts
        for i in range(self.parts):
            name = 'classifier'+str(i)
            setattr(self, name, nn.Linear(in_planes, num_classes, bias=False))

    def forward(self, x):
        predict = []
        for i in range(self.parts):
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict.append(c(x[:,:,i]))
        return predict

class SqueezeNeck(nn.Module):
    def __init__(self, in_planes, out_planes, parts=6):
        super(SqueezeNeck, self).__init__()
        for i in range(6):
            name = 'block{}'.format(i)
            setattr(self, name, nn.Sequential(
                nn.Linear(in_planes, out_planes, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.LeakyReLU(0.1)))
        self.parts = parts
        self.apply(weights_init_kaiming)
    def forward(self, x):
        y = [getattr(self, 'block{}'.format(i))(x[:,:,i]) for i in range(self.parts)]
        return torch.stack(y, 2)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        if neck == 'bnneck-pcb':
            self.gap=nn.AdaptiveAvgPool2d((3, 1))
        elif neck == 'bnneck-pcb-v1':
            self.gap=nn.AdaptiveAvgPool2d((6, 1))
        else:
            self.gap=nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'bnneck-pcb':
            print('Training with pcb......')
            self.bottleneck=nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier=PCBClassifier(self.in_planes, self.num_classes, parts=3)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        elif self.neck == 'bnneck-pcb-v1':
            print('Training with pcb v1......')
            self.squeezeneck=SqueezeNeck(self.in_planes, 256, parts=6)
            self.bottleneck=nn.BatchNorm1d(256)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier=PCBClassifier(256, self.num_classes, parts=6)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        before_gap = self.base(x)
        global_feat = self.gap(before_gap)  # (b, 2048, 1, 1)
        #global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if global_feat.shape[3] == 1:
            global_feat = global_feat.squeeze(3)
        if global_feat.shape[2] == 1:
            global_feat = global_feat.squeeze(2)
        if self.neck == 'bnneck-pcb-v1':
            global_feat = self.squeezeneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif 'bnneck' in self.neck:
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        cls_score = self.classifier(feat)
        if self.neck_feat == 'after-v1':
            return dict(logits=cls_score, features=feat, before_gap=before_gap)
        elif self.neck_feat == 'after':
            return dict(logits=cls_score, features=global_feat, features_test=feat, before_gap=before_gap)
        elif self.neck_feat == 'before':
            return dict(logits=cls_score, features=global_feat, before_gap=before_gap)
        else:
            raise NotImplementedError

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
