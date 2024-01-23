'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import ir_1w1a

__all__ = ['resnet20_1w1a', 'ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = ir_1w1a.SKDIRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w1a.SKDIRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     ir_1w1a.SKDIRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x, realx):
        # first convolutional block
        Binaryout1, Realout1 = self.conv1(x, realx)
        Binaryout1 = self.bn1(Binaryout1)
        Binaryout1 += self.shortcut(x)
        Binaryout1 = F.hardtanh(Binaryout1)
        # Copy reused layers and freeze them
        realbn1 = copy.deepcopy(self.bn1)
        realshortcut = copy.deepcopy(self.shortcut)
        with torch.no_grad():
            Realout1 = realbn1(Realout1)
            Realout1 += realshortcut(realx)
            Realout1 = F.hardtanh(Realout1)
        
        # second convolution block
        Binaryout2, Realout2 = self.conv2(Binaryout1, Realout1)
        Binaryout2 = self.bn2(Binaryout2)
        Binaryout2 += Binaryout1
        Binaryout2 = F.hardtanh(Binaryout2)
        # Copy reused layers and freeze them
        realbn2 = copy.deepcopy(self.bn2)
        with torch.no_grad():
            Realout2 = realbn2(Realout2)
            Realout2 += Realout1
            Realout2 = F.hardtanh(Realout2)

        return Binaryout1, Realout1, Binaryout2, Realout2

        # out = self.bn1(self.conv1(x))
        # out += self.shortcut(x)
        # out = F.hardtanh(out)
        # x1 = out
        # out = self.bn2(self.conv2(out))
        # out += x1
        # out = F.hardtanh(out)
        # return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))

        L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout = self.layer1[0](out, out.detach())
        L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout = self.layer1[1](L1_2_Binaryout, L1_2_Realout)
        L1_5_Binaryout, L1_5_Realout, L1_6_Binaryout, L1_6_Realout = self.layer1[2](L1_4_Binaryout, L1_4_Realout)

        L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout = self.layer2[0](L1_6_Binaryout, L1_6_Realout)
        L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout = self.layer2[1](L2_2_Binaryout, L2_2_Realout)
        L2_5_Binaryout, L2_5_Realout, L2_6_Binaryout, L2_6_Realout = self.layer2[2](L2_4_Binaryout, L2_4_Realout)

        L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout = self.layer3[0](L2_6_Binaryout, L2_6_Realout)
        L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout = self.layer3[1](L3_2_Binaryout, L3_2_Realout)
        L3_5_Binaryout, L3_5_Realout, L3_6_Binaryout, L3_6_Realout = self.layer3[2](L3_4_Binaryout, L3_4_Realout)

        out = F.avg_pool2d(L3_6_Binaryout, L3_6_Binaryout.size()[3])
        out = out.view(out.size(0), -1)
        penul_Binaryout = self.bn2(out)
        out = self.linear(penul_Binaryout)

        return out, penul_Binaryout, \
               L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout, L1_3_Binaryout, L1_3_Realout, \
               L1_4_Binaryout, L1_4_Realout, L1_5_Binaryout, L1_5_Realout, L1_6_Binaryout, L1_6_Realout, \
               L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout, L2_3_Binaryout, L2_3_Realout, \
               L2_4_Binaryout, L2_4_Realout, L2_5_Binaryout, L2_5_Realout, L2_6_Binaryout, L2_6_Realout, \
               L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout, L3_3_Binaryout, L3_3_Realout, \
               L3_4_Binaryout, L3_4_Realout, L3_5_Binaryout, L3_5_Realout, L3_6_Binaryout, L3_6_Realout


def resnet20_1w1a():
    return ResNet(BasicBlock_1w1a, [3, 3, 3])


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
