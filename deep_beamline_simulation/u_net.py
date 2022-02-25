import os
import cv2
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.nn import (
    Upsample,
    ConvTranspose2d,
    Conv2d,
    MaxPool2d,
    Module,
    ModuleList,
    ReLU,
    Dropout,
)
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
        # stores a list of images to process for training or testing
        self.image_list = image_list

    def smallest_image_size(self):
        # set to values that will be instantly changed
        min_height = 10e4
        min_length = 10e4
        # loop through the images to get the minimum and maximum to resize
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
        # use CV2 to resize images to desired shape and interpolate if needed
        res = cv2.resize(
            image, dsize=(length - 1, height - 3), interpolation=cv2.INTER_CUBIC
        )
        return res

    def normalize_image(self, image):
        # normalize image by the mean and standard deviation to improve training
        im_mean = np.mean(image)
        im_std = np.std(image)
        return (image - im_mean) / im_std

    def loss_crop(self, image):
        # crop images to recompute the loss to avoid all of th extra empty space
        image = (image[0, 0, :, :]).numpy()
        cropped_image = []
        for row in image:
            crop = row[17:25]
            cropped_image.append(crop)
        cropped_image = np.asarray(cropped_image[55:80])
        cropped_image = torch.from_numpy(cropped_image.astype("f"))
        return cropped_image


class UNet(Module):
    """
    Defines the UNet architecture
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        # define input layer
        self.input_layer = Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # for going down the U
        self.conv_inx64 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv_64x128 = Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_128x256 = Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_256x512 = Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.relu = ReLU()

        self.dropout = Dropout(0.5)

        self.maxpool = MaxPool2d(2)

        self.upsample = Upsample(scale_factor=2, mode="nearest")
        self.upsample_final = Upsample(size=(input_size, output_size))

        # for going up the U
        self.upconv1024 = ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv512 = ConvTranspose2d(513, 256, kernel_size=3, stride=1, padding=1)
        self.upconv256 = ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv128 = ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)

        # used for down blocks
        self.conv64 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv128 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv256 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv512 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1024 = Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # used for up blocks
        self.conv64_1 = Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv128_1 = Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv256_1 = Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv512_1 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1024_1 = Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # define output layer
        self.output_layer = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # define parameter layer for exapanding parameters of the change
        self.param_layer = torch.nn.Linear(2, 850)

    def forward(self, inputs, input_params):
        # down
        # encoder block 1
        x = self.input_layer(inputs)
        x = self.conv_inx64(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv64(x)
        x = self.maxpool(x)

        # encoder block 2
        x = self.conv_64x128(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv128(x)
        x = self.maxpool(x)

        # encoder block 3
        x = self.conv_128x256(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv256(x)
        x = self.maxpool(x)

        # encoder block 4
        x = self.conv_256x512(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv512(x)

        # the aperature horizonal/vertical position
        parameters = torch.tensor(input_params)
        parameters = self.param_layer(parameters)

        # flatten original tensor out of bottom of u_net
        flat = torch.flatten(x)
        # concatenate
        x = torch.cat((flat, parameters))
        # reshape with one more value than before to preserve final shape
        x = torch.reshape(x, (10, 513, 17, 5))

        x = torch.squeeze(x)

        # up
        # decoder block 1
        x = self.upconv512(x)
        # x = self.conv_512x256(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv256_1(x)

        # decoder block 2
        x = self.upconv256(x)
        # x = self.conv_256x128(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv128_1(x)

        # decoder block 3
        x = self.upconv128(x)
        # x = self.conv_128x64(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.upsample_final(x)
        x = self.conv64_1(x)

        output = self.output_layer(x)
        return output


class ImageDataset(Dataset):
    """
    Allows pytorch to access images for training
    """

    def __init__(self, path):
        self.data_path = path
        self.file = None
        self.beamIntensities = None
        self.parameters = None
        self.image_count = 0

    def find_images(self):
        self.file = h5py.File(self.data_path)
        self.beamIntensities = file["/beamIntensities"]
        self.image_count = len(self.beamIntensities)
        self.parameters = file["paramVals"][:]

    def __del__(self):
        self.file.close()

    def __getitem__(self, index):
        """
        Returns initial intensity (for beam), parameters and intensity for given index
        """
        return (self.parameters[index], self.beamIntensities[index])

    def __len__(self):
        return len(self.image_count)
