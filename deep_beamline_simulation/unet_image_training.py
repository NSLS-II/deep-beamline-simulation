import os
import torch
import pandas
import numpy as np
from pathlib import Path
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from u_net import Block, Encoder, Decoder, UNet, ImageProcessing, ParamUnet
#from u_net import Block, Encoder, Decoder, UNet, ImageProcessing
from u_net import UNet, ImageProcessing


# file path for train images
train_file = "image_data/Initial-Intensity-33-1798m.csv"
output_file = "image_data/Intensity-At-Sample-63-3m.csv"

#file path for test images
test_file = "image_data/initialInt_262.csv"
test_output_file = "image_data/sample_555.csv"

# read csv using pandas, skip first row for headers
train_numpy = pandas.read_csv(train_file, skiprows=1).to_numpy()
output_numpy = pandas.read_csv(output_file, skiprows=1).to_numpy()

test_numpy = pandas.read_csv(test_file, skiprows=1).to_numpy()
test_output_numpy = pandas.read_csv(test_output_file, skiprows=1).to_numpy()

# create a list to store in ImageProcessing class
image_list = [train_numpy, output_numpy, test_numpy, test_output_numpy]
ip = ImageProcessing(image_list)

# find the size of the smallest image
height, length = ip.smallest_image_size()


# resize all images bigger than the smallest image size
resized_train_image = ip.resize(train_numpy, height, length)
resized_output_image = ip.resize(output_numpy, height, length)
resized_test_image = ip.resize(test_numpy, height, length)
resized_test_output_image = ip.resize(test_output_numpy, height, length)


# normalize all training and testing data
train_numpy = ip.normalize_image(resized_train_image)
output_image = ip.normalize_image(resized_output_image)
test_numpy = ip.normalize_image(resized_test_image)
test_output_numpy = ip.normalize_image(resized_test_output_image)

# convert each numpy array into a tensor
train_image = torch.from_numpy(train_numpy.astype("f"))
output_image = torch.from_numpy(output_image.astype("f"))
test_image = torch.from_numpy(test_numpy.astype("f"))
test_output_image = torch.from_numpy(test_output_numpy.astype("f"))


# create model
#model = ParamUnet()
model = UNet()

'''
Sanity check for image sizes
print("Train Image Size")
print(train_image.size())
print("Output Image Size")
print(output_image.size())
'''
print(train_image.shape)
train_image = train_image[:, :, None, None]
output_image = output_image[:, :, None, None]

test_image = test_image[:, :, None, None]
test_output_image = test_output_image[:, :, None, None]


#Sanity check for image sizes
print("Train Image Size After Adding Channels")
print(train_image.shape)
print("Output Image Size After Adding Channels")
print(output_image.shape)


# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()


# loop through many epochs
for e in range(1, 3001):
    predictions = model(train_image)
    crop_pred = predictions.detach()
    crop_train = ip.loss_crop(train_image)
    crop_out = ip.loss_crop(crop_pred)
    loss = loss_func(predictions, output_image)
    cropped_train_loss = loss_func(crop_train, crop_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e%1000 == 0:
        # output the loss
        print('Training Loss: ' + str(loss.data.numpy()))
        print('Cropped Training Loss: ' + str(cropped_train_loss.numpy()))


# test the model (same image for now)
test_predictions = None
with torch.no_grad():
    model.eval()
    test_predictions = model(test_image)
    test_loss = loss_func(test_predictions, test_output_image)
    test_pred_cropped = ip.loss_crop(test_predictions)
    output_cropped = ip.loss_crop(test_output_image)
    cropped_test_loss = loss_func(test_pred_cropped, output_cropped)

# output test loss
print("Test Loss: " + str(test_loss.data.numpy()))
# cropped output test loss
print("Cropped Test Loss: " + str(cropped_test_loss.data.numpy()))


# remove extra dimenations to plot and view output
prediction_im = test_predictions[0,0,:,:]
actual_im = test_output_image[0,0,:,:]


# plot output
plt.subplot(121)
plt.imshow(actual_im, extent=[0,100,0,1], aspect='auto')
plt.subplot(122)
plt.imshow(prediction_im, extent=[0,100,0,1], aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.show()
