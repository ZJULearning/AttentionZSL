import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=True):
        super(ConvLayer, self).__init__()
        if isinstance(kernel_size, int):
            reflection_padding = kernel_size // 2
        elif isinstance(kernel_size, tuple):
            assert(len(kernel_size) == 2)
            # (paddingLeft, paddingRight, paddingTop, paddingBottom)
            reflection_padding = (kernel_size[1], kernel_size[1], kernel_size[0], kernel_size[0])
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.padding = padding

    def forward(self, x):
        if self.padding:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels, kernels=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernels, stride=stride)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernels, stride=stride)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

def weight_init(*ms): 
    for m in ms:
        if isinstance(m, torch.nn.Linear):
            size = m.weight.size()
            fan_out = size[0] # number of rows
            fan_in = size[1] # number of columns
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(0.0, variance)
            
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
