import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

from .basic import weight_init

class LFGAAVGG19(torch.nn.Module):
    def __init__(self, k):
        super(LFGAAVGG19, self).__init__()
        self.k = k
        vgg19 = models.vgg19_bn(pretrained=True)

        features_list = list(vgg19.features.children())
        self.conv2_2 = torch.nn.Sequential(*features_list[:13])      # 1 x 128 x 112x112
        self.conv3_4 = torch.nn.Sequential(*features_list[13:26])     # 1 x 256 x 56 x 56
        self.conv4_4 = torch.nn.Sequential(*features_list[26: 39])  # 1 x 512 x 28 x 28
        self.conv5_4 = torch.nn.Sequential(*features_list[39:-1])   # 1 x 512 x 14 x 14

        self.tail_layer = features_list[-1]
        self.fc_layers = list(vgg19.classifier.children())[:-2]
        self.fc_layers = torch.nn.Sequential(*list(self.fc_layers))

        self.extract_0 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=8, stride=8),
                torch.nn.Conv2d(128, self.k, kernel_size=1, stride=1)
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
                torch.nn.Conv2d(512, self.k, kernel_size=1, stride=1)
            )

        self.fc0 = torch.nn.Linear(196, 1, bias=True)
        self.fc1 = torch.nn.Linear(196, 1, bias=True)
        self.fc2 = torch.nn.Linear(196, 1, bias=True)
        self.fc3 = torch.nn.Linear(196, 1, bias=True)
        
        self.fc4 = torch.nn.Linear(4096, 2 * k, bias=True)
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
        conv2_2 = self.conv2_2(x)
        conv3_4 = self.conv3_4(conv2_2)
        conv4_4 = self.conv4_4(conv3_4)
        conv5_4 = self.conv5_4(conv4_4)

        x = F.relu(self.fc4(self.fc_layers(self.tail_layer(conv5_4).view(-1, 25088))))
        attr = self.bn1(x[:, :self.k])
        latent = self.bn2(x[:, self.k:])
        
        feats_0 = self.extract_0(conv2_2)
        feats_1 = self.extract_1(conv3_4)
        feats_2 = self.extract_2(conv4_4)
        feats_3 = self.extract_3(conv5_4) # N x k x 14 x 14
        
        p_0 = self.attention_sublayers(feats_0, self.fc0, latent)
        p_1 = self.attention_sublayers(feats_1, self.fc1, latent)
        p_2 = self.attention_sublayers(feats_2, self.fc2, latent)
        p_3 = self.attention_sublayers(feats_3, self.fc3, latent) # N x k 

        p = p_0 + p_1 + p_2 + p_3 

        return attr * p, latent
        
