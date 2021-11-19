import os
import pandas
#import deep_beamline_simulation
from u_net import Block, Encoder, Decoder, UNet
#from deep_beamline_simulation.u_net import Block, Encoder, Decoder, UNet
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchinfo import summary
from torch.utils.data import DataLoader


train_file = 'image_data/Initial-Intensity-33-1798m.csv'
output_file = 'image_data/Intensity-At-Sample-63-3m.csv'
# torch.from_numpy(np.random.rand(3, 20, 20).astype("f"))

train_numpy = pandas.read_csv(train_file, skiprows=1).to_numpy()
output_numpy = pandas.read_csv(output_file, skiprows=1).to_numpy()

train_image = torch.from_numpy(train_numpy.astype('f'))
output_image = torch.from_numpy(output_numpy.astype('f'))

print('Train Image Size')
print(train_image.size())
print('Output Image Size')
print(output_image.size())

model = UNet()

train_image = train_image[None, None, :]
#output_image = output_image[None, None, :]
print('Train Image Size After Adding Channels')
print(train_image.shape)
#print('Output Image Size After Adding Channels')
#print(output_image.shape)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

epochs = 5
loss_list = []
training_loss = 0.0

for e in range(0, 10):
    predictions = model(train_image)
    print(predictions.shape)
    # print(outputs.shape)
    loss = loss_func(predictions, output_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("Test Loss: " + str(loss))
test_predictions = None
with torch.no_grad():
    model.eval()
    test_predictions = model(test_in)
    test_loss = loss_func(test_predictions, test_out)
    test_plot = test_predictions.data.numpy()[0]
print("Test Loss: " + str(test_loss))
print(test_predictions)
