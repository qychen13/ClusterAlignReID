import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PersonReidModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(PersonReidModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return dict(features=features, logits=logits)

# test use only


class PersonReidModelNeck(PersonReidModel):
    def __init__(self, *args, **kwargs):
        super(PersonReidModelNeck, self).__init__(*args, **kwargs)

    def forward(self, x):
        fs = self.feature_extractor(x)
        fs = self.classifier.bottleneck(fs)
        fs = self.classifier.bn(fs)

        return fs


def weight_initialization(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
        if m.affine is not None:
            # nn.init.constant_(m.weight, 1.0)
            nn.init.normal_(m.weight.data, 1.0, 0.02) # change to person reid strong baseline
            nn.init.constant_(m.bias, 0.0)

# copy from person reid strong baseline
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def resnet50_feature_extractor(last_stride, *args, **kwargs):
    model = tv_models.resnet50(*args, **kwargs)
    # delete fc layer
    model.fc = Identity()
    if last_stride == 1:
        model.layer4[0].downsample[0].stride = (1, 1)
        model.layer4[0].conv2.stride = (1, 1)
    return model

def resnet50_feature_extractor_v1(last_stride, *args, **kwargs):
    model = resnet50_feature_extractor(last_stride, *args, **kwargs)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5)
    )
    model.fc.apply(weight_initialization)
    return model


class BaselineClassifier(nn.Module):
    def __init__(self, num_classes, in_planes):
        super(BaselineClassifier, self).__init__()
        self.fc = nn.Linear(in_planes, num_classes)
        # self.apply(weight_initialization)
        self.apply(weights_init_classifier)

    def forward(self, x):
        return self.fc(x)


class BNNeckClassifer(nn.Module):
    # from reid-strong-baseline
    def __init__(self, num_classes, in_planes):
        super(BNNeckClassifer, self).__init__()
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.fc = nn.Linear(in_planes, num_classes, bias=False)
        self.bottleneck.apply(weight_initialization)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        y = self.bottleneck(x)
        y = self.fc(y)

        return y


class BottleNeckClassifier(nn.Module):
    def __init__(self, num_classes, in_planes, num_channels=512):
        super(BottleNeckClassifier, self).__init__()
        self.bottleneck = nn.Linear(in_planes, num_channels, bias=False)
        self.bn = nn.BatchNorm1d(num_channels)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_channels, num_classes)
        self.apply(weight_initialization)

    def forward(self, x):
        y = self.bottleneck(x)
        y = self.bn(y)
        # y = F.leaky_relu(y, 0.1)
        y = self.dropout(y)
        y = self.fc(y)

        return y
