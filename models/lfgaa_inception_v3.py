import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models as th_models

from .basic import weight_init

model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

class LFGAAGoogleNet(torch.nn.Module):
    def __init__(self, k):
        super(LFGAAGoogleNet, self).__init__()
        self.k = k
    
        inception_v3 = th_models.Inception3(aux_logits=False, transform_input=False)
        state_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        state_dict_rm_aux = {k: v for k, v in state_dict.items() if 'AuxLogits' not in k}
        inception_v3.load_state_dict(state_dict_rm_aux)

        layers = list(inception_v3.children())[:-1]
        layers.insert(3, torch.nn.MaxPool2d(3, 2))
        layers.insert(6, torch.nn.MaxPool2d(3, 2))
        layers.append(torch.nn.AvgPool2d(8))

        self.layers_1 = torch.nn.Sequential(*layers[:3]) # 64 x 147 x 147
        self.layers_2 = torch.nn.Sequential(*layers[3:6]) # 192 x 71 x 71
        self.layers_3 = torch.nn.Sequential(*layers[6:10]) # 288 x 35 x 35
        self.layers_4 = torch.nn.Sequential(*layers[10:15]) # 768 x 17 x 17
        self.tail_layers   = torch.nn.Sequential(*layers[15:])

        self.extract_1 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=8, stride=8, padding=-2),
                torch.nn.Conv2d(64, self.k, kernel_size=1, stride=1)
            )
        self.extract_2 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=4, stride=4),
                torch.nn.Conv2d(192, self.k, kernel_size=1, stride=1)
            )
        self.extract_3 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(288, self.k, kernel_size=1, stride=1)
            )
        self.extract_4 = torch.nn.Sequential(
                torch.nn.Conv2d(768, self.k, kernel_size=1, stride=1)
            )
        
        self.fc1 = torch.nn.Linear(289, 1, bias=True)
        self.fc2 = torch.nn.Linear(289, 1, bias=True)
        self.fc3 = torch.nn.Linear(289, 1, bias=True)
        self.fc4 = torch.nn.Linear(289, 1, bias=True)

        self.fc5 = torch.nn.Linear(2048, 2 * k, bias=True)
        self.bn1 = torch.nn.BatchNorm1d(k)
        self.bn2 = torch.nn.BatchNorm1d(k)

        weight_init(self.fc1,
                    self.fc2,
                    self.fc3,
                    self.fc4,
                    self.fc5)

    def attention_sublayers(self, feats, embedding_layers, latent):
        feats = feats.view((feats.size(0), self.k, -1))
        feats = feats.transpose(dim0=1, dim1=2)
        feats = feats + latent.unsqueeze(1)
        feats = feats.transpose(dim0=1, dim1=2)

        feats = embedding_layers(feats).squeeze(-1)
        p = F.softmax(feats, dim=1)
        return p

    def forward(self, x):
        feats_1 = self.layers_1(x)
        feats_2 = self.layers_2(feats_1)
        feats_3 = self.layers_3(feats_2)
        feats_4 = self.layers_4(feats_3)

        x = F.relu(self.fc5(self.tail_layers(feats_4).view(-1, 2048)))
        attr = self.bn1(x[:, :self.k])
        latent = self.bn2(x[:, self.k:])

        feats_1 = self.extract_1(feats_1)
        feats_2 = self.extract_2(feats_2)
        feats_3 = self.extract_3(feats_3)
        feats_4 = self.extract_4(feats_4) # N x k x 17 x 17
        
        p_1 = self.attention_sublayers(feats_1, self.fc1, latent)
        p_2 = self.attention_sublayers(feats_2, self.fc2, latent)
        p_3 = self.attention_sublayers(feats_3, self.fc3, latent)
        p_4 = self.attention_sublayers(feats_4, self.fc4, latent) # N x k

        p = p_1 + p_2 + p_3 + p_4

        return attr * p, latent
