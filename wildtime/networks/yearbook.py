import torch
import torch.nn as nn
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.heads import SimCLRProjectionHead

def define_model(config):
    layers_dict = {}
    nn_space = config.copy()
    n_convs = 4
    pre_flat_size = 32
    in_channels = 3
    out_kernel = None
    layers = []
    
    for i in range(n_convs):
        out_channels = nn_space.get("n_conv_channels_c{}".format(i+1))
        kernel_size = nn_space.get("kernel_size_c{}".format(i+1))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        pre_flat_size = pre_flat_size - kernel_size+1
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        in_channels = out_channels
        out_kernel = kernel_size
    layers_dict["enc"] = nn.Sequential(*layers)

    layers = []
    in_features = out_channels
    layers_dict["classifier"] = nn.Sequential(*layers)
    return layers_dict
       
class YearbookNetwork(nn.Module):
    def __init__(self, args, num_input_channels, num_classes, ssl_training=False):
        super(YearbookNetwork, self).__init__()
        self.use_config = args.use_config
        use_config = self.use_config
        if use_config is None:
            self.args = args
            self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                    self.conv_block(32, 32), self.conv_block(32, 32))
            self.hid_dim = 32
            self.classifier = nn.Linear(32, num_classes)

            # SimCLR: projection head
            if self.args.method == 'simclr':
                self.projection_head = SimCLRProjectionHead(32, 32, 128)
            # SwaV: projection head and prototypes
            elif self.args.method == 'swav':
                self.projection_head = SwaVProjectionHead(32, 32, 128)
                self.prototypes = SwaVPrototypes(128, n_prototypes=32)
            self.ssl_training = ssl_training
        else:
            layers_dict = define_model(use_config)
            self.enc = layers_dict["enc"]
            self.classifier = layers_dict["classifier"]


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        if self.use_config is None:
            x = self.enc(x)
            x = torch.mean(x, dim=(2, 3))
            if self.args.method == 'simclr' and self.ssl_training:
                return self.projection_head(x)
            elif self.args.method == 'swav' and self.ssl_training:
                x = self.projection_head(x)
                x = nn.functional.normalize(x, dim=1, p=2)
                return self.prototypes(x)
            else:
                # [32,32]
                x = self.classifier(x)
                return x
        else:
            x = self.enc(x)
            x = torch.mean(x, dim=(2, 3))
            x = self.classifier(x)
            return x
