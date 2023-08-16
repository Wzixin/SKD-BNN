'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import os
import math
import copy
import torch
import torch.nn as nn
import ir_1w1a

__all__ = ['ResNet', 'resnet18', 'resnet34']

# you need to donwload the models to models_dir
# resnet18: https://download.pytorch.org/models/resnet18-5c106cde.pth
# resnet34: https://download.pytorch.org/models/resnet34-333f7ec4.pth
models_dir = './Torch_Models'
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
}


def conv3x3Binary(in_planes, out_planes, stride=1):
    "3x3 convolution with padding , with Binary"
    return ir_1w1a.IRConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3Binary(inplanes, planes, stride)  # conv3x3Binary or conv3x3
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear = nn.Hardtanh(inplace=True)  # Hardtanh or ReLU
        self.conv2 = conv3x3Binary(planes, planes)  # conv3x3Binary or conv3x3
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, realx):
        Binaryout1, Realout1 = self.conv1(x, realx)
        Binaryout1 = self.bn1(Binaryout1)
        if self.downsample is not None:
            x = self.downsample(x)
        Binaryout1 += x
        Binaryout1 = self.nonlinear(Binaryout1)
        realbn1 = copy.deepcopy(self.bn1)
        realdownsample = copy.deepcopy(self.downsample)
        with torch.no_grad():
            Realout1 = realbn1(Realout1)
            if self.downsample is not None:
                realx = realdownsample(realx)
            Realout1 += realx
            Realout1 = self.nonlinear(Realout1)

        Binaryout2, Realout2 = self.conv2(Binaryout1, Realout1)
        Binaryout2 = self.bn2(Binaryout2)
        Binaryout2 += Binaryout1
        Binaryout2 = self.nonlinear(Binaryout2)
        realbn2 = copy.deepcopy(self.bn2)
        with torch.no_grad():
            Realout2 = realbn2(Realout2)
            Realout2 += Realout1
            Realout2 = self.nonlinear(Realout2)

        return Binaryout1, Realout1, Binaryout2, Realout2


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False, avg_down=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.deep_stem = deep_stem
        self.avg_down = avg_down

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.Hardtanh(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.Hardtanh(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.maxpool(x)

        L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout = self.layer1[0](x, x.detach())
        L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout = self.layer1[1](L1_2_Binaryout, L1_2_Realout)
        L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout = self.layer2[0](L1_4_Binaryout, L1_4_Realout)
        L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout = self.layer2[1](L2_2_Binaryout, L2_2_Realout)
        L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout = self.layer3[0](L2_4_Binaryout, L2_4_Realout)
        L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout = self.layer3[1](L3_2_Binaryout, L3_2_Realout)
        L4_1_Binaryout, L4_1_Realout, L4_2_Binaryout, L4_2_Realout = self.layer4[0](L3_4_Binaryout, L3_4_Realout)
        L4_3_Binaryout, L4_3_Realout, L4_4_Binaryout, L4_4_Realout = self.layer4[1](L4_2_Binaryout, L4_2_Realout)

        out = self.avgpool(L4_4_Binaryout)
        out = out.view(out.size(0), -1)
        penul_Binaryout = self.bn2(out)
        out = self.fc(penul_Binaryout)

        return out, penul_Binaryout, \
               L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout, \
               L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout, \
               L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout, \
               L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout, \
               L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout, \
               L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout, \
               L4_1_Binaryout, L4_1_Realout, L4_2_Binaryout, L4_2_Realout, \
               L4_3_Binaryout, L4_3_Realout, L4_4_Binaryout, L4_4_Realout


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    # if pretrained:
    #     weights_dict = torch.load(os.path.join(models_dir, model_name['resnet18']))
    #     load_weights_dict = {k: v for k, v in weights_dict.items()
    #                          if model.state_dict()[k].numel() == v.numel()}
    #     del load_weights_dict['fc.weight']
    #     del load_weights_dict['fc.bias']
    #     del weights_dict
    #     model.load_state_dict(load_weights_dict)
    # return model

    if pretrained:
        assert os.path.exists(models_dir)
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert os.path.exists(models_dir)
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model

