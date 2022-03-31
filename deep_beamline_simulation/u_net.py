import os
import cv2
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

import deep_beamline_simulation

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

    def preprocess(self, filename):
        dbs_path = Path(deep_beamline_simulation.__file__)

        dbs_repository = dbs_path.parent.parent

        # get data from file
        # example: filename = "NSLS-II-TES-beamline-rsOptExport-2/rsopt350/datasets/results.h5"
        train_file = dbs_repository / filename

        # read the results.h5 file
        with h5py.File(train_file) as f:
            beam_intensities = f['beamIntensities']
            self.image_list = beam_intensities
            min_height, min_length = self.smallest_image_size()

            # identify parameter shapes and counts
            print(f"Parameter Shape: {f['params'].shape}" )
            _parameter_count = f["params"].shape[0]

            # can call loss_crop or crop manually
            h = beam_intensities.shape[1]
            w = beam_intensities.shape[2]
            hi = 0 + (h // 3)
            hj = h - (h // 3)
            wi = 0 + (w // 3)
            wj = w - (w // 3)

            cropped_beam_intensities = beam_intensities[:, hi:hj, wi:wj]
            plt.figure()
            plt.imshow(beam_intensities[0], aspect='auto')
            plt.title('Cropped Image')

            log_cropped_beam_intensities = np.log(cropped_beam_intensities + 1e-10)
            plt.figure()
            plt.hist(log_cropped_beam_intensities.flatten(), bins=100)
            plt.title("log transformed cropped image data")
            plt.show()

            normalized_log_cropped_beam_intensities = (log_cropped_beam_intensities - np.mean(
            log_cropped_beam_intensities)) / np.std(log_cropped_beam_intensities)
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].hist(normalized_log_cropped_beam_intensities.flatten(), bins=300)
            axs[0].set_title("Normalized Log Transformed Cropped Image Data")

            axs[1].hist(np.std(normalized_log_cropped_beam_intensities, axis=(1, 2)), bins=300)
            axs[1].set_title("STD")
            plt.show()

            # determine if blank images occur and remove them
            bad_image_indices = []
            good_image_indices = []
            resized_images = []

            for i in range(normalized_log_cropped_beam_intensities.shape[0]):
                std = np.std(normalized_log_cropped_beam_intensities[i])
                if 1e-10 < std:
                    good_image_indices.append(i)
                else:
                    print(f"rejecting image {i} with std {std:.3e}")
                    bad_image_indices.append(i)
                    # don't plot all bad images, there are about 50
                    if len(bad_image_indices) < 3:
                        plt.figure()
                        plt.imshow(
                            ip.resize(
                                normalized_log_cropped_beam_intensities[i],
                                height=128 + 3,
                                length=128 + 1
                            ),
                            aspect="equal"
                        )
                        plt.show()
                resized_images.append(
                    self.resize(
                        normalized_log_cropped_beam_intensities[i],
                        height=128 + 3,
                        length=128 + 1
                    )
                )

            print(f"bad image count: {len(bad_image_indices)}")

            # this is the input image or the intial intensity
            initial_beam_intensity_csv_path = dbs_repository / "NSLS-II-TES-beamline-rsOptExport-2/tes_init.csv"

            initial_beam_intensity = pd.read_csv(initial_beam_intensity_csv_path, skiprows=1).to_numpy()
            min_initial_beam_intensity = np.min(initial_beam_intensity)
            print(f"min initial beam intensity {min_initial_beam_intensity}")

            if min_initial_beam_intensity > 0:
                e = 0.0
            elif min_initial_beam_intensity == 0.0:
                e = 1e-10
            else:
                e = 1e-10 + np.abs(min_initial_beam_intensity)

            log_initial_beam_intensity = np.log(
                initial_beam_intensity + e
            )
            plt.figure()
            plt.hist(log_initial_beam_intensity.flatten(), bins=100)
            plt.title("log_initial_beam_intensity")
            plt.show()

            normalized_initial_beam_intensity = (log_initial_beam_intensity - np.mean(log_initial_beam_intensity)) / np.std(
                log_initial_beam_intensity)
            resized_initial_beam_intensity = self.resize(
                normalized_initial_beam_intensity,
                height=128 + 3,
                length=128 + 1
            )

            with h5py.File("preprocessed_results.h5", mode="w") as preprocessed_results:
                good_image_count = len(good_image_indices)

                pi_ds = preprocessed_results.create_dataset(
                    "preprocessed_initial_beam_intensity",
                    (128, 128)
                )
                pi_ds[:] = resized_initial_beam_intensity

                params_ds = preprocessed_results.create_dataset_like("params", f["params"])
                for i, param in enumerate(f["params"]):
                    params_ds[i] = param

                pbi_ds = preprocessed_results.create_dataset(
                    "preprocessed_beam_intensities",
                    (good_image_count, 128, 128)
                )

                normalized_param_vals_ds = preprocessed_results.create_dataset(
                    "preprocessed_param_vals",
                    (good_image_count, f["paramVals"].shape[1])
                )

                normalized_param_vals = (f["paramVals"] - np.mean(f["paramVals"])) / np.std(f["paramVals"])
                print(f"normalized_param_vals\n{normalized_param_vals}")

                for i, good_i in enumerate(good_image_indices):
                    normalized_param_vals_ds[i] = normalized_param_vals[good_i]
                    pbi_ds[i] = resized_images[good_i]

                for good_i in good_image_indices[:10]:
                    print(f"std: {np.std(pbi_ds[good_i])}")
                    f, ax = plt.subplots(nrows=1, ncols=3)
                    ax[0].imshow(beam_intensities[good_i], aspect="equal")
                    ax[1].imshow(normalized_log_cropped_beam_intensities[good_i], aspect="equal")
                    ax[2].imshow(resized_images[good_i], aspect="equal")
                    plt.title(f"{params_ds[:]}\n{normalized_param_vals[good_i, :]}")
                    plt.show()

            with h5py.File("preprocessed_results.h5", mode="r") as preprocessed_results:
                print(preprocessed_results.keys())
                print(preprocessed_results["params"])
                print(preprocessed_results["params"][:])
                print(preprocessed_results["preprocessed_param_vals"])
                # print(preprocessed_results["preprocessed_initial_beam_intensity"][0:2, :10])

        return _parameter_count, resized_images


class UNet(Module):
    """
    Defines the UNet architecture
    """

    def __init__(self, input_size, output_size, parameter_count):
        super().__init__()
        # define input layer
        self.input_layer = Conv2d(in_channels = 1, out_channels = 1, kernel_size=3, stride=1, padding=1)
        # for going down the U
        self.conv_inx64 = Conv2d(in_channels = 1, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.conv_64x128 = Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.conv_128x256 = Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv_256x512 = Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, stride=1, padding=1)

        self.relu = ReLU()

        self.dropout = Dropout(0.5)

        self.maxpool = MaxPool2d(2)

        self.upsample = Upsample(scale_factor=2, mode="nearest")
        self.upsample_final = Upsample(size=(input_size, output_size))

        # for going up the U
        self.upconv1024 = ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size=3, stride=1, padding=1)
        self.upconv512 = ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.upconv256 = ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.upconv128 = ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=3, stride=1, padding=1)

        # used for down blocks
        self.conv64 = Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.conv128 = Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.conv256 = Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv512 = Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, stride=1, padding=1)
        self.conv1024 = Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3, stride=1, padding=1)

        # used for up blocks
        self.conv64_1 = Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.conv128_1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1, padding=1)
        self.conv256_1 = Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1, padding=1)
        self.conv512_1 = Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, stride=1, padding=1)
        self.conv1024_1 = Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3, stride=1, padding=1)

        # define output layer
        self.output_layer = Conv2d(in_channels = 64, out_channels = 1, kernel_size=3, stride=1, padding=1)

        # define parameter layer for exapanding parameters of the change
        # add more to this for shaping issues
        self.param_layer = torch.nn.Linear(parameter_count, 128)
        self.param_layer_256 = torch.nn.Linear(128, 256)
        self.param_layer_512 = torch.nn.Linear(256, 512)

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
        #print(parameters.shape)
        parameters = self.param_layer(parameters)
        #print(parameters.shape)
        #parameters = torch.permute(parameters, (1, 0))
        #print(parameters.shape)

        x = self.param_layer_256(parameters)
        x = self.param_layer_512(x)

        # flatten original tensor out of bottom of u_net
        #flat = torch.flatten(x)
        # concatenate
        #x = torch.cat((flat, parameters))
        # reshape with one more value than before to preserve final shape
        #x = torch.reshape(x, (513, 17, 5))
        #print(x.shape)
        x = x[:, :, None, None]

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


class IntensityImageDataset:
    '''
    Create dataloaders for training and testing on intensity image data
    '''
    def __init__(self, beam_intensities, initial_beam_intensity, params, param_vals):
        self.beam_intensities = np.expand_dims(beam_intensities, axis = 1)
        self.initial_beam_intensity = np.expand_dims(initial_beam_intensity, axis=0)
        self.params = params 
        self.param_vals = param_vals.astype("float32")

    def __getitem__(self, index):
        '''
        Returns the specified intensity image at the index given
        '''
        return self.beam_intensities[index], self.initial_beam_intensity, self.param_vals[index]

    def __len__(self):
        return self.beam_intensities.shape[0]

    def report(self):
        '''
        Print information about the current data
        '''
        print(f'length: {len(self)}')
        print(f"initial beam intensity.shape:\n{self.initial_beam_intensity.shape}\n")
        print(f"data shape:\n{self.beam_intensities.shape}\n")
        print(f"data at index 0:\n{self[0]}\n")
        print(f'beamline parameters dtype:\n\t{self.params.dtype}\n')
        print(f'beamline parameters:\n\t{self.params}\n')


def build_dataloaders(data_path, batch_size):
    with h5py.File(data_path, mode='r') as preprocessed_results:
        '''
        Parse the results that have been preprocessed and make datasets and dataloaders
        '''
        initial_beam_intensity_ds = preprocessed_results['preprocessed_initial_beam_intensity']
        initial_beam_intensity = np.zeros_like(initial_beam_intensity_ds)
        initial_beam_intensity[:] = initial_beam_intensity_ds[:]

        beam_intensities_ds = preprocessed_results['preprocessed_beam_intensities']
        beam_intensities = np.zeros_like(beam_intensities_ds)
        beam_intensities[:] = beam_intensities_ds[:]

        beam_parameters_ds = preprocessed_results['params']
        beam_parameters = np.zeros_like(beam_parameters_ds)
        beam_parameters[:] = beam_parameters_ds[:]

        beam_param_values_ds = preprocessed_results['preprocessed_param_vals']
        beam_param_values = np.zeros_like(beam_param_values_ds)
        beam_param_values[:] = beam_param_values_ds[:]

        # get two thirds of the data to use for training
        training_size = 2 * (beam_intensities.shape[0] // 3)

        training_intensity_dataset = IntensityImageDataset(
            beam_intensities= beam_intensities[:training_size],
            initial_beam_intensity=initial_beam_intensity,
            params = beam_parameters,
            param_vals = beam_param_values[:training_size]
            )

        training_intensity_dataloader = DataLoader(
            training_intensity_dataset,
            batch_size = batch_size,
            shuffle=True
            )

        testing_intensity_dataset = IntensityImageDataset(
            beam_intensities = beam_intensities[training_size:],
            initial_beam_intensity=initial_beam_intensity,
            params=beam_parameters,
            param_vals = beam_param_values[training_size:]
            )

        testing_intensity_dataloader = DataLoader(
            testing_intensity_dataset,
            batch_size = batch_size,
            shuffle = True
            )

    return training_intensity_dataloader, testing_intensity_dataloader



