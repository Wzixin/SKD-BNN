'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import ir_1w1a
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        # init.kaiming_uniform_(m.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ir_1w1a.SKDIRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ir_1w1a.SKDIRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, realx):
        # first convolutional block
        Binaryout1, Realout1 = self.conv1(x, realx)
        Binaryout1 = self.bn1(Binaryout1)
        Binaryout1 = F.hardtanh(Binaryout1)
        # Copy reused layers and freeze them
        realbn1 = copy.deepcopy(self.bn1)
        with torch.no_grad():
            Realout1 = realbn1(Realout1)
            Realout1 = F.hardtanh(Realout1)

        # second convolution block
        Binaryout2, Realout2 = self.conv2(Binaryout1, Realout1)
        Binaryout2 = self.bn2(Binaryout2)
        Binaryout2 += self.shortcut(x)
        Binaryout2 = F.hardtanh(Binaryout2)
        # Copy reused layers and freeze them
        realbn2 = copy.deepcopy(self.bn2)
        realshortcut = copy.deepcopy(self.shortcut)
        with torch.no_grad():
            Realout2 = realbn2(Realout2)
            Realout2 += realshortcut(realx)
            Realout2 = F.hardtanh(Realout2)

        # out = F.hardtanh(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        # out = F.hardtanh(out)
        # return out

        return Binaryout1, Realout1, Binaryout2, Realout2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512*block.expansion)
        self.apply(_weights_init)
        # self.softmax = nn.LogSoftmax()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.bn1(self.conv1(x))
        out = F.hardtanh(self.bn1(self.conv1(x)))

        L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout = self.layer1[0](out, out.detach())
        L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout = self.layer1[1](L1_2_Binaryout, L1_2_Realout)

        L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout = self.layer2[0](L1_4_Binaryout, L1_4_Realout)
        L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout = self.layer2[1](L2_2_Binaryout, L2_2_Realout)

        L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout = self.layer3[0](L2_4_Binaryout, L2_4_Realout)
        L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout = self.layer3[1](L3_2_Binaryout, L3_2_Realout)

        L4_1_Binaryout, L4_1_Realout, L4_2_Binaryout, L4_2_Realout = self.layer4[0](L3_4_Binaryout, L3_4_Realout)
        L4_3_Binaryout, L4_3_Realout, L4_4_Binaryout, L4_4_Realout = self.layer4[1](L4_2_Binaryout, L4_2_Realout)

        out = F.avg_pool2d(L4_4_Binaryout, 4)
        out = out.view(out.size(0), -1)
        penul_Binaryout = self.bn2(out)
        out = self.linear(penul_Binaryout)
        # out = self.softmax(out)

        return out, penul_Binaryout, \
               L1_1_Binaryout, L1_1_Realout, L1_2_Binaryout, L1_2_Realout, \
               L1_3_Binaryout, L1_3_Realout, L1_4_Binaryout, L1_4_Realout, \
               L2_1_Binaryout, L2_1_Realout, L2_2_Binaryout, L2_2_Realout, \
               L2_3_Binaryout, L2_3_Realout, L2_4_Binaryout, L2_4_Realout, \
               L3_1_Binaryout, L3_1_Realout, L3_2_Binaryout, L3_2_Realout, \
               L3_3_Binaryout, L3_3_Realout, L3_4_Binaryout, L3_4_Realout, \
               L4_1_Binaryout, L4_1_Realout, L4_2_Binaryout, L4_2_Realout, \
               L4_3_Binaryout, L4_3_Realout, L4_4_Binaryout, L4_4_Realout


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
