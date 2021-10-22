# import statements
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


''' Create the basic block architecture'''
class Block(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3):
        super().__init__()
        # kernel_size = 3
        self.conv_in = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # kernel_size = 3
        self.conv_out = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1)

    def forward(self,x):
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return self.relu(x)


''' similar to standard CNN '''
class Encoder(nn.Module):
    def __init__(self, num_channels = (3,64,128)):
        super().__init__()
        # creates a block for each pair of channels 
        self.enc_blocks = nn.ModuleList([Block(num_channels[i], num_channels[i+1]) for i in range(len(num_channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = []
        for block in self.enc_blocks:
            x = block(x)
            out.append(x)
            x = self.pool(x)
        return out


class Decoder(nn.Module):
    ''' up convolution part '''
    def __init__(self, num_channels=(128,64)):
        super().__init__()
        self.num_channels = num_channels
        # conv transpose for decreasing pairs of channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(num_channels[i], num_channels[i+1], 2, 2) for i in range(len(num_channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(num_channels[i], num_channels[i+1]) for i in range(len(num_channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.num_channels)-1):
            x = self.upconvs[i](x)
            encoder_features = self.crop(encoder_features[i],x)
            # concat the unconvs and encoder features
            x = torch.cat([x, encoder_features], dim=1)
            x = self.dec_blocks[i](x)
        return x 

    ''' Downsizing and creating the correct output shape '''
    def crop(self, encoder_features, x):
        _, _, H, W = x.shape
        encoder_features = torchvision.transforms.CenterCrop([H,W])(encoder_features)
        return encoder_features


class UNet(nn.Module):
    def __init__(self, encoder_channels=(3,64,128), decoder_channels=(128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            # interpolates output if the current output is not the correct output size
            out = F.interpolate(out, out_sz)
        return out
