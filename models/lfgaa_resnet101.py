import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

from .basic import weight_init

class LFGAAResNet101(torch.nn.Module):
    def __init__(self, k):
        super(LFGAAResNet101, self).__init__()
        self.k = k

        resnet101 = models.resnet101(pretrained=True)
        self.layers = list(resnet101.children())

        self.conv1_layer = torch.nn.Sequential(*self.layers[:3]) # 1 x 64  x 112x 112
        self.conv2_layer = torch.nn.Sequential(*self.layers[3:5]) # 1 x 256 x 56 x 56
        self.conv3_layer = self.layers[5] # 1 x 512 x 28 x 28
        self.conv4_layer = self.layers[6] # 1 x 1024 x 14 x 14

        self.tail_layer = torch.nn.Sequential(*self.layers[7:-1])

        self.extract_0 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=8, stride=8),
                torch.nn.Conv2d(64, self.k, kernel_size=1, stride=1)
            )
        self.extract_1 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=4, stride=4),
                torch.nn.Conv2d(256, self.k, kernel_size=1, stride=1)
            )
        self.extract_2 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(512, self.k, kernel_size=1, stride=1)
            )
        self.extract_3 = torch.nn.Sequential(
                torch.nn.Conv2d(1024, self.k, kernel_size=1, stride=1)
            )
        
        self.fc0 = torch.nn.Linear(196, 1, bias=True)
        self.fc1 = torch.nn.Linear(196, 1, bias=True)
        self.fc2 = torch.nn.Linear(196, 1, bias=True)
        self.fc3 = torch.nn.Linear(196, 1, bias=True)
        
        self.fc4 = torch.nn.Linear(2048, 2 * k, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(k)
        self.bn2 = torch.nn.BatchNorm1d(k)

        weight_init(self.fc0,
                    self.fc1,
                    self.fc2,
                    self.fc3,
                    self.fc4)

    def attention_sublayers(self, feats, embedding_layers, latent):
        feats = feats.view((feats.size(0), self.k, -1))
        feats = feats.transpose(dim0=1, dim1=2)
        feats = feats + latent.unsqueeze(1)
        feats = feats.transpose(dim0=1, dim1=2)

        feats = embedding_layers(feats).squeeze(-1)
        p = F.softmax(feats, dim=1)
        return p

    def forward(self, x):
        conv1_layer = self.conv1_layer(x)
        conv2_layer = self.conv2_layer(conv1_layer)
        conv3_layer = self.conv3_layer(conv2_layer)
        conv4_layer = self.conv4_layer(conv3_layer)

        x = F.relu(self.fc4(self.tail_layer(conv4_layer).view(-1, 2048)))
        attr = self.bn1(x[:, :self.k])
        latent = self.bn2(x[:, self.k:])

        feats_0 = self.extract_0(conv1_layer)
        feats_1 = self.extract_1(conv2_layer)
        feats_2 = self.extract_2(conv3_layer)
        feats_3 = self.extract_3(conv4_layer) # N x k x 14 x 14
        
        p_0 = self.attention_sublayers(feats_0, self.fc0, latent)
        p_1 = self.attention_sublayers(feats_1, self.fc1, latent)
        p_2 = self.attention_sublayers(feats_2, self.fc2, latent)
        p_3 = self.attention_sublayers(feats_3, self.fc3, latent) # N x k 

        p = p_0 + p_1 + p_2 + p_3 

        return attr * p, latent

