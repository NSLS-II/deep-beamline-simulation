import os
import torch
import pandas
import numpy as np
from pathlib import Path
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from u_net import Block, Encoder, Decoder, UNet, ImageProcessing


train_file = "image_data/Initial-Intensity-33-1798m.csv"
output_file = "image_data/Intensity-At-Sample-63-3m.csv"

train_numpy = pandas.read_csv(train_file, skiprows=1).to_numpy()
output_numpy = pandas.read_csv(output_file, skiprows=1).to_numpy()

image_list = [train_numpy, output_numpy]

ip = ImageProcessing(image_list)
height, length = ip.smallest_image_size()
resized_output_image = ip.resize(output_numpy, height, length)

train_image = torch.from_numpy(train_numpy.astype("f"))
output_image = torch.from_numpy(resized_output_image.astype("f"))

model = UNet()

print("Train Image Size")
print(train_image.size())
print("Output Image Size")
print(output_image.size())

train_image = train_image[None, None, :]
output_image = output_image[None, None, :]

print("Train Image Size After Adding Channels")
print(train_image.shape)
print("Output Image Size After Adding Channels")
print(output_image.shape)


optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

epochs = 5
loss_list = []
training_loss = 0.0

for e in range(0, 10000):
    predictions = model(train_image)
    loss = loss_func(predictions, output_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.data.numpy())

test_predictions = None
with torch.no_grad():
    model.eval()
    test_predictions = model(train_image)
    test_loss = loss_func(test_predictions, output_image)

print("Test Loss: " + str(test_loss.data.numpy()))
