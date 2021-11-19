import torch
import numpy as np
import torchvision
from PIL import Image
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class ImageProcessing:
    """
    Processes images for proper training specifications 
    Prevents issues with size and shape of all images in a dataset
    Normalizes values in images to prevent training issues

    Parameters
    ----------
    image_list : list 
        holds the list of images to transform with following methods
    """

    def __init__(self, image_list):
        self.image_list = image_list

    def smallest_image_size(self):
        min_height = 10e4
        min_length = 10e4

        for s in self.image_list:
            shape = s.shape
            height = shape[0]
            length = shape[1]
            if height < min_height:
                min_height = height
            if length < min_length:
                min_length = length
        return min_height, min_length

    def resize(self, image, height, length):
        image = Image.fromarray(image)
        resized_image = image.resize((length - 1, height - 1))
        resized_image = np.asarray(resized_image)
        return resized_image

    def normalize(self):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        transformed_images = []
        for im in self.image_list:
            transformed_images.append(transform(im))
        return transformed_images


class Block(nn.Module):
    """ 
    Create the basic block architecture with conv2d in and out and RELU activation
    Assumes kernel size is 3, stride is 1, and padding is 1
    
    Parameters
    ----------
    input_channels : int
        The number of channels in the input image
        Set to 1 if no channels necessary
    output_channels : int
        The number of channels in the output image
        Set to 1 if no channels necessary
    """

    def __init__(self, input_channels=1, output_channels=3):
        super().__init__()
        self.input_layer = nn.Conv2d(
            input_channels, output_channels, 3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.output_layer = nn.Conv2d(
            output_channels, output_channels, 3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    """ 
    Creates the Encoder block using the Block class that contracts images
    The encoder is similar to a standard (Convolutional Neural Network CNN)
    Assumes kernel size is 3, stride is 1, and padding is 1 from class Block
    
    Parameters
    ----------
    num_channels : list, tuple
        As the image trains the number of channels will be increased
        Later to be decreased by the conv transpose
    """

    def __init__(self, num_channels=(1, 64, 128)):
        super().__init__()
        block_list = []
        for i in range(len(num_channels) - 1):
            block_list.append(Block(num_channels[i], num_channels[i + 1]))

        self.encoder_blocks = nn.ModuleList(block_list)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        output = []
        for block in self.encoder_blocks:
            x = block(x)
            output.append(x)
            x = self.maxpool(x)
        return output


class Decoder(nn.Module):
    """ 
    Creates the Decoder block using the Block class to expand images
    Up samples image at each step to increase image from low to high resolution
    Assumes kernel size is 3, stride is 1, and padding is 1 from class Block
    
    Parameters
    ----------
    num_channels : list, tuple
        As the image trains the number of channels will be decreased from 
        the maximum size from the encoder
    Methods
    -------
    crop - Downsizes x to ensure correct output shape
    """

    def __init__(self, num_channels=(128, 64)):
        super().__init__()
        self.num_channels = num_channels
        # conv transpose for decreasing pairs of channels
        upconv_list = []
        decblock_list = []
        for i in range(len(num_channels) - 1):
            upconv_list.append(
                nn.ConvTranspose2d(num_channels[i], num_channels[i + 1], 2, 2)
            )
            decblock_list.append(Block(num_channels[i], num_channels[i + 1]))
        self.upconv_modules = nn.ModuleList(upconv_list)
        self.decoder_blocks = nn.ModuleList(decblock_list)

    def forward(self, x, encoder_features):
        for i in range(len(self.num_channels) - 1):
            x = self.upconv_modules[i](x)
            encoder_features = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_features], dim=1)
            x = self.decoder_blocks[i](x)
        return x

    def crop(self, encoder_features, x):
        _, _, H, W = x.shape
        encoder_features = torchvision.transforms.CenterCrop([H, W])(encoder_features)
        return encoder_features


class UNet(nn.Module):
    """ 
    Creates a Unet model using the Block, Encoder, and Decoder classes
    
    Parameters
    ----------
    encoder_channels : list, tuple
        same channels as the encoder class
    decoder_channels : list, tuple
        same channels as the decoder class
    groups : int
        default is 1 so that all inputs are convolved to all outputs
    """

    def __init__(
        self, encoder_channels=(1, 64, 128), decoder_channels=(128, 64), groups=1,
    ):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], groups, 1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.head(out)
        return out
