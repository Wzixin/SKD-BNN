import torch.nn as nn
import torch.nn.functional as F
import binaryfunction
import torch
import math
from SKD_BNN import Rough_Normalized_scale


def PFRecover(input, k, t):
    return k * torch.tanh(input * t)


def PFRecover2(input):
    mask1 = input < -1
    mask2 = input < 0
    mask3 = input < 1
    out1 = (-1) * mask1.type(torch.float32) + (input * input + 2 * input) * (1 - mask1.type(torch.float32))
    out2 = out1 * mask2.type(torch.float32) + (-input * input + 2 * input) * (1 - mask2.type(torch.float32))
    out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
    return out3


class SKDIRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(SKDIRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input, realinput):
        w = self.weight
        a = input

        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()

        with torch.no_grad():
            Ra = realinput
            Rw = bw

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw

        # binary convolution output
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)

        with torch.no_grad():
            # Full precision convolution output
            RealConv_output = F.conv2d(Ra, Rw, self.bias, self.stride, self.padding, self.dilation, self.groups)
            RealConv_output = torch.abs(output) * torch.sign(RealConv_output)

        return output, RealConv_output