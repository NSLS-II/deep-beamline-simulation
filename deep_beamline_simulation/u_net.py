import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.nn import Upsample, ConvTranspose2d, Conv2d, MaxPool2d, Module, ModuleList, ReLU, Dropout
from torchvision.transforms import CenterCrop


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
        res = cv2.resize(
            image, dsize=(length - 1, height - 3), interpolation=cv2.INTER_CUBIC
        )
        return res

    def normalize_image(self, image):
        im_mean = np.mean(image)
        im_std = np.std(image)
        return (image - im_mean) / im_std

    def loss_crop(self, image):
        image = (image[0, 0, :, :]).numpy()
        cropped_image = []
        for row in image:
            crop = row[17:25]
            cropped_image.append(crop)
        cropped_image = np.asarray(cropped_image[55:80])
        cropped_image = torch.from_numpy(cropped_image.astype("f"))
        return cropped_image


class UNet(Module):
    def __init__(self, input_size=136, output_size=40):
        super().__init__()
        self.input_layer = Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # for going down the U
        self.conv_inx64 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv_64x128 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_128x256 = Conv2d(128, 256, kernel_size=3, stride = 1, padding=1)
        self.conv_256x512 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv_512x1024 = Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        self.relu_1 = ReLU()
        self.relu_2 = ReLU()
        self.relu_3 = ReLU()
        self.relu_4 = ReLU()
        self.relu_5 = ReLU()
        self.relu_6 = ReLU()
        self.relu_7 = ReLU()
        self.relu_8 = ReLU()
        self.relu_9 = ReLU()

        self.dropout_1 = Dropout(0.5)
        self.dropout_2 = Dropout(0.5)
        self.dropout_3 = Dropout(0.5)
        self.dropout_4 = Dropout(0.5)
        self.dropout_5 = Dropout(0.5)
        self.dropout_6 = Dropout(0.5)
        self.dropout_7 = Dropout(0.5)
        self.dropout_8 = Dropout(0.5)
        self.dropout_9 = Dropout(0.5)

        self.maxpool_1 = MaxPool2d(2)
        self.maxpool_2 = MaxPool2d(2)
        self.maxpool_3 = MaxPool2d(2)
        self.maxpool_4 = MaxPool2d(2)
        self.maxpool_5 = MaxPool2d(2)

        # for going up the U
        self.upconv1024 = ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv512 = ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv256 = ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv128 = ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.conv_1024x512 = Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv_512x256 = Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv_256x128 = Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv_128x64 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1)


        # used for blocks 
        self.conv64 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv128 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv256 = Conv2d(256, 256, kernel_size=3, stride = 1, padding=1)
        self.conv512 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1024 = Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # used for blocks 
        self.conv64_1 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv128_1 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv256_1 = Conv2d(256, 256, kernel_size=3, stride = 1, padding=1)
        self.conv512_1 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1024_1 = Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)


        self.output_layer = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        # down
        # encoder block 1
        x = self.input_layer(inputs)
        x = self.conv_inx64(x)
        x = self.dropout_1(x)
        x = self.relu_1(x)
        x = self.conv64(x)
        #x = self.maxpool_1(x)

        # encoder block 2
        x = self.conv_64x128(x)
        x = self.dropout_2(x)
        x = self.relu_2(x)
        x = self.conv128(x)
        #x = self.maxpool_2(x)

        # encoder block 3
        x = self.conv_128x256(x)
        x = self.dropout_3(x)
        x = self.relu_3(x)
        x = self.conv256(x)
        #x = self.maxpool_3(x)

        # encoder block 4
        x = self.conv_256x512(x)
        x = self.dropout_4(x)
        x = self.relu_4(x)
        x = self.conv512(x)
        #x = self.maxpool_4(x)

        # # encoder block 5
        # x = self.conv_512x1024(x)
        # x = self.dropout_5(x)
        # x = self.relu_5(x)
        # x = self.conv1024(x)
        # x = self.maxpool_5(x)

        # # up
        # # decoder block 1
        # x = self.upconv1024(x)
        # #x = self.conv_1024x512(x)
        # x = self.dropout_6(x)
        # x = self.relu_6(x)
        # x = self.conv512_1(x)

        # decoder block 2
        x = self.upconv512(x)
        #x = self.conv_512x256(x)
        x = self.dropout_7(x)
        x = self.relu_7(x)
        x = self.conv256_1(x)

        # decoder block 3
        x = self.upconv256(x)
        #x = self.conv_256x128(x)
        x = self.dropout_8(x)
        x = self.relu_8(x)
        x = self.conv128_1(x)

        # decoder block 4
        x = self.upconv128(x)
        #x = self.conv_128x64(x)
        x = self.dropout_9(x)
        x = self.relu_9(x)
        x = self.conv64_1(x)

        output = self.output_layer(x)
        return output

        '''
        Going down the U
        conv 2d 64 
        relu, dropout, maxpool (to reduce dims)
        conv 2d 128
        relu, dropout, maxpool (to reduce dims)
        conv 2d 256
        relu, dropout, maxpool (to reduce dims)
        conv 2d 512
        relu, dropout, maxpool (to reduce dims)
        conv 2d 1024

        Going up the U
        ConvTranspose2d 1024, 512
        conv 2d 1024, 512
        relu, dropout
        ConvTranspose2d 512, 256
        conv 2d 512, 256
        relu, dropout
        ConvTranspose2d 256, 128
        conv 2d 256, 128
        relu, dropout
        ConvTranspose2d 256, 128
        conv 2d 256, 128
        relu, dropout
        ConvTranspose2d 128, 64
        conv 2d 128, 64
        relu, dropout

        output layer conv2d (64 ...)
        '''





