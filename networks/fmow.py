from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
import math

IMG_HEIGHT = 224
NUM_CLASSES = 62

class FMoWNetwork(nn.Module):
    def __init__(self, args, weights=None, ssl_training=False):
        super(FMoWNetwork, self).__init__()
        self.args = args
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))
        # SimCLR projection head
        if self.args.method == 'simclr':
            from lightly.models.modules.heads import SimCLRProjectionHead
            self.projection_head = SimCLRProjectionHead(1024, 1024, 128)
        # SwaV: projection head and prototypes
        elif self.args.method == 'swav':
            from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
            self.projection_head = SwaVProjectionHead(1024, 1024, 128)
            self.prototypes = SwaVPrototypes(128, n_prototypes=1024)
        self.ssl_training = ssl_training

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError('nan in input x')
        features = self.enc(x)
        if torch.isnan(features).any():
            raise ValueError('nan in features')
        out = F.relu(features, inplace=True)
        if torch.isnan(out).any():
            raise ValueError('nan in relu')
        out = F.adaptive_avg_pool2d(out, (1, 1))
        if torch.isnan(out).any():
            raise ValueError('nan in adaptive avg pool2d')
        out = torch.flatten(out, 1)
        if torch.isnan(out).any():
            raise ValueError('nan in flatten')

        if self.args.method == 'simclr' and self.ssl_training:
            return self.projection_head(out)
        elif self.args.method == 'swav' and self.ssl_training:
            if torch.isnan(out).any():
                raise ValueError('error in early part')
            out = self.projection_head(out)
            if torch.isnan(out).any():
                raise ValueError('error in projection_head')
            out = nn.functional.normalize(out, dim=1, p=2)
            if torch.isnan(out).any():
                raise ValueError('error in normalize')
            out = self.prototypes(out)
            if torch.isnan(out).any():
                raise ValueError('error in prototypes')
            return out
        else:
            return self.classifier(out)