# import statements
import torch
import torch.nn as nn
import torchvision


# encoder doubles the number of channels at every step and halves spatial dimension
# max pool reduces the height and width of the image by half


# decoder upsamples the feature maps, doubles the spatial dimensions and halves the number of channels
# upsample using conv transpose 2d, doubles the height and width

# relu activation between each layer

#Convolution operations have kernel size of 3, and no padding
#the output feature map doesn’t have the same Height and Width as the input feature map

class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3)

    def forward(self,x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    ''' DownSampling Part'''
    def __init__(self, channels = (3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs =[]
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    ''' up convolution part '''
    def __init__(self, channels=(1024, 512, 256, 128,64)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i],x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x 

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H,W])(enc_ftrs)
        return enc_ftrs


encoder = Encoder()
# input image
x    = torch.randn(1, 3, 572, 572)
ftrs = encoder(x)
decoder = Decoder()
x = torch.randn(1, 1024, 28, 28)
decoder(x, ftrs[::-1][1:]).shape

