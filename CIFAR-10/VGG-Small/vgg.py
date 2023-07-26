'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import ir_1w1a
import copy
import torch
import torch.nn.functional as F


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg_small', 'vgg_small_1w1a'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.classifier(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Hardtanh(inplace=True)]
            else:
                layers += [conv2d, nn.Hardtanh(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model


class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)

        self.conv1 = ir_1w1a.SKDIRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = ir_1w1a.SKDIRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = ir_1w1a.SKDIRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = ir_1w1a.SKDIRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = ir_1w1a.SKDIRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.SKDIRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)

        L1_Binaryout, L1_Realout = self.conv1(x, x.detach())
        L1_Binaryout = self.pooling(L1_Binaryout)
        L1_Binaryout = self.bn1(L1_Binaryout)
        L1_Binaryout = self.nonlinear(L1_Binaryout)
        # copy重复使用的层，并进行冻结
        realbn1 = copy.deepcopy(self.bn1)
        with torch.no_grad():
            L1_Realout = self.pooling(L1_Realout)
            L1_Realout = realbn1(L1_Realout)
            L1_Realout = self.nonlinear(L1_Realout)

        L2_Binaryout, L2_Realout = self.conv2(L1_Binaryout, L1_Realout)
        L2_Binaryout = self.bn2(L2_Binaryout)
        L2_Binaryout = self.nonlinear(L2_Binaryout)
        # copy重复使用的层，并进行冻结
        realbn2 = copy.deepcopy(self.bn2)
        with torch.no_grad():
            L2_Realout = realbn2(L2_Realout)
            L2_Realout = self.nonlinear(L2_Realout)

        L3_Binaryout, L3_Realout = self.conv3(L2_Binaryout, L2_Realout)
        L3_Binaryout = self.pooling(L3_Binaryout)
        L3_Binaryout = self.bn3(L3_Binaryout)
        L3_Binaryout = self.nonlinear(L3_Binaryout)
        # copy重复使用的层，并进行冻结
        realbn3 = copy.deepcopy(self.bn3)
        with torch.no_grad():
            L3_Realout = self.pooling(L3_Realout)
            L3_Realout = realbn3(L3_Realout)
            L3_Realout = self.nonlinear(L3_Realout)

        L4_Binaryout, L4_Realout = self.conv4(L3_Binaryout, L3_Realout)
        L4_Binaryout = self.bn4(L4_Binaryout)
        L4_Binaryout = self.nonlinear(L4_Binaryout)
        # copy重复使用的层，并进行冻结
        realbn4 = copy.deepcopy(self.bn4)
        with torch.no_grad():
            L4_Realout = realbn4(L4_Realout)
            L4_Realout = self.nonlinear(L4_Realout)

        L5_Binaryout, L5_Realout = self.conv5(L4_Binaryout, L4_Realout)
        L5_Binaryout = self.pooling(L5_Binaryout)
        L5_Binaryout = self.bn5(L5_Binaryout)
        L5_Binaryout = self.nonlinear(L5_Binaryout)
        # copy重复使用的层，并进行冻结
        realbn5 = copy.deepcopy(self.bn5)
        with torch.no_grad():
            L5_Realout = self.pooling(L5_Realout)
            L5_Realout = realbn5(L5_Realout)
            L5_Realout = self.nonlinear(L5_Realout)

        out = L5_Binaryout.view(L5_Binaryout.size(0), -1)
        penul_Binaryout = out
        out = self.fc(out)

        return out, penul_Binaryout, \
               L1_Binaryout, L1_Realout, \
               L2_Binaryout, L2_Realout, \
               L3_Binaryout, L3_Realout, \
               L4_Binaryout, L4_Realout, \
               L5_Binaryout, L5_Realout


def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model
