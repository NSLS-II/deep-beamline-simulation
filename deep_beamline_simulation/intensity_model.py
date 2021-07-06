import re
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import *
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.preprocessing import PolynomialFeatures


# use for normalization
def min_max(lists):
    # large numbers to override in search process
    minimum = 1e20
    maximum = 1e-20
    for l in lists:
        for e in l:
            if e < minimum:
                minimum = e
            if e > maximum:
                maximum = e
    return minimum, maximum


# define neural network
class Neural_Net(torch.nn.Module):
    def __init__(self):
        super(Neural_Net, self).__init__()
        # define layers
        self.hidden1 = torch.nn.Linear(3, 256)
        self.hidden2 = torch.nn.Linear(256, 512)
        # define dropout
        self.dropout1 = torch.nn.Dropout(0.6)
        self.hidden3 = torch.nn.Linear(512, 512)
        self.dropout2 = torch.nn.Dropout(0.6)
        self.hidden4 = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, 70)

    def forward(self, x):
        # call layers and use relu activation function
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden3(x))
        x = self.dropout2(x)
        x = F.relu(self.hidden4(x))
        x = self.output(x)
        return x


# data organization - file names and parameters
# to be automated
data = {}
data["210301_horpos.csv"] = [21.0, 0.3, 0.1]
data["210102_horpos.csv"] = [21.0, 0.1, 0.2]
data["210305_hospos.csv"] = [21.0, 0.3, 0.5]
data["210304_horpos.csv"] = [21.0, 0.3, 0.4]
data["210205_hospos.csv"] = [21.0, 0.2, 0.5]
data["210201_horpos.csv"] = [21.0, 0.2, 0.1]
data["210303_horpos.csv"] = [21.0, 0.3, 0.3]
data["210105_horpos.csv"] = [21.0, 0.1, 0.5]
data["210203_horpos.csv"] = [21.0, 0.2, 0.3]
data["210204_hospos.csv"] = [21.0, 0.2, 0.4]
data["210103_horpos.csv"] = [21.0, 0.1, 0.3]
data["210104_horpos.csv"] = [21.0, 0.1, 0.4]
data["210202_horpos.csv"] = [21.0, 0.2, 0.2]
data["210101_horpos.csv"] = [21.0, 0.1, 0.1]
data["210302_horpos.csv"] = [21.0, 0.3, 0.2]

# read data from file
pos_int = {}
path_to_file = "Data_Collection/"
for file, parameters in data.items():
    position_intensity = np.genfromtxt(path_to_file + file, delimiter=",")
    # remove headers
    position_intensity = np.array(position_intensity[1:])
    pos_int[file] = position_intensity
# strip file names and obtain all data
trainx = list(data.values())
trainy = []
pos_ints_vals = pos_int.values()
# turn into proper formatting for training
for lst in pos_ints_vals:
    ints = lst[:, 1]
    trainy.append(np.array(ints))

min_y, max_y = min_max(trainy)
norm_trainy = []
for y in trainy:
    # normalize train y
    norm_trainy.append((y - min_y) / (max_y - min_y))

# define writer for tensorboard implementation
writer = SummaryWriter()

# create model object
model = Neural_Net()
# display model
print("Model Structure")
print(model)

# convert numpy array to tensor of input shape and convert to float type
x = torch.from_numpy(np.array(trainx)).float()
y = torch.from_numpy(np.array(norm_trainy)).float()

# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

# change type to Variable (contains gradients)
inputs = Variable(x)
outputs = Variable(y)

# define a loss list
loss_plot = []

# define data location for tensorboard access
file_path = '/vagrant/tensorboard.txt'
# open file
file = open(file_path, "w")

# start training time
t_start = time.time()

# begin training
for i in range(5000):
    prediction = model(inputs)
    loss = loss_func(prediction, outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # collect loss
    loss_plot.append(loss.data.numpy())

    if i % 1000 == 0:
        # plot and show learning process
        plt.cla()
        for i in range(len(trainy)):
            plt.cla()
            plt.plot(
                np.linspace(-200, 200, len(trainy[0])), norm_trainy[i], label="original"
            )
            plt.title(i)
            plt.plot(
                np.linspace(-200, 200, len(norm_trainy[0])),
                prediction.data.numpy()[i],
                label="predicted",
            )
            plt.text(0, 0, "Loss=%.4f" % loss.data.numpy())
            plt.legend()
            plt.pause(0.5)
            plt.plot()

# note training has ended
print("training completed")

# end training time
t_end = time.time()

# print time in minutes elapsed
print("Time elapsed: " + str((t_end - t_start) / 60.0))

for element in loss_plot:
    file.write(str(element)+'\n')
file.close()
