import torch
import torch.nn as nn
from lightly.models.modules.heads import SimCLRProjectionHead

class YearbookNetwork(nn.Module):
    def __init__(self, args, num_input_channels, num_classes, ssl_training=False):
        super(YearbookNetwork, self).__init__()
        self.args = args
        self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.hid_dim = 32
        self.classifier = nn.Linear(32, num_classes)
        # Projection head for SimCLR
        self.projection_head = SimCLRProjectionHead(128, 128, 128)
        self.ssl_training = ssl_training

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.enc(x)

        if self.args.method == 'simclr' and self.ssl_training:
            h = x.flatten(start_dim=1)
            return self.projection_head(h)
        else:
            h = torch.mean(x, dim=(2, 3))
            return self.classifier(h)