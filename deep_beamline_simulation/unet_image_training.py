import os
import torch
import pandas
import numpy as np
from pathlib import Path
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from u_net import Block, Encoder, Decoder, UNet, ImageProcessing


# file path for images
train_file = "../image_data/Initial-Intensity-33-1798m.csv"
output_file = "../image_data/Intensity-At-Sample-63-3m.csv"

# read csv using pandas, skip first row for headers
train_numpy = pandas.read_csv(train_file, skiprows=1).to_numpy()
output_numpy = pandas.read_csv(output_file, skiprows=1).to_numpy()

# create a list to store in ImageProcessing class
image_list = [train_numpy, output_numpy]
ip = ImageProcessing(image_list)

# find the size of the smallest image
height, length = ip.smallest_image_size()

# resize all images bigger than the smallest image size
resized_output_image = ip.resize(output_numpy, height, length)

# normalize all training data
train_numpy = ip.normalize_image(train_numpy)
resized_output_image = ip.normalize_image(resized_output_image)

# convert each numpy array into a tensor
train_image = torch.from_numpy(train_numpy.astype("f"))
output_image = torch.from_numpy(resized_output_image.astype("f"))

# create model
model = UNet()

'''
Sanity check for image sizes
print("Train Image Size")
print(train_image.size())
print("Output Image Size")
print(output_image.size())
'''

train_image = train_image[None, None, :]
output_image = output_image[None, None, :]

'''
Sanity check for image sizes
print("Train Image Size After Adding Channels")
print(train_image.shape)
print("Output Image Size After Adding Channels")
print(output_image.shape)
'''

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.L1Loss()

# loop through many epochs
for e in range(1, 4001):
    predictions = model(train_image)
    loss = loss_func(predictions, output_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e%1000 == 0:
        # output the loss
        print('Training Loss:' + str(loss.data.numpy()))


# test the model (same image for now)
test_predictions = None
with torch.no_grad():
    model.eval()
    test_predictions = model(train_image)
    test_loss = loss_func(test_predictions, output_image)

# output test loss
print("Test Loss: " + str(test_loss.data.numpy()))

# remove extra dimenations to plot and view output
prediction_im = test_predictions[0,0,:,:]
actual_im = output_image[0,0,:,:]

# plot output
plt.subplot(121)
plt.imshow(actual_im, extent=[0,100,0,1], aspect='auto')
plt.subplot(122)
plt.imshow(prediction_im, extent=[0,100,0,1], aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.show()



