import time
import torch
from torch.autograd import Variable
from torch.nn import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# define neural network
class Neural_Net(torch.nn.Module):
    def __init__(self):
        super(Neural_Net, self).__init__()
        # define layers
        self.hidden1 = torch.nn.Linear(7, 256)
        self.hidden2 = torch.nn.Linear(256, 512)
        self.hidden3 = torch.nn.Linear(512, 512)
        self.hidden4 = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, 1)

    def forward(self, x):
        # call layers and use relu activation function
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.output(x)
        return x


# create model object
model = Neural_Net()
# display model
print("Model Structure")
print(model)

# get data from csv file
data = np.genfromtxt(
    "Intensity-At-BPM1-38-6904m-Horizontal-Position.csv", delimiter=","
)
data = np.array(data[1:])

position = data[:, 0]
intensity = data[:, 1]

# find max and min for normalization process
min_ints = min(intensity)
print("Min" + str(min_ints))
max_ints = max(intensity)
print("Max" + str(max_ints))
# normalize the intensity
norm_intensity = (intensity - min_ints) / (max_ints - min_ints)

# apply polynomial preprocessing
poly = PolynomialFeatures(6)
x_expanded = np.expand_dims(position, axis=1)
x_expanded = poly.fit_transform(x_expanded)

# convert numpy array to tensor of input shape
x_poly = torch.from_numpy(x_expanded).float()
y_norm = torch.from_numpy(norm_intensity.reshape(-1, 1)).float()

# optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

# change type to Variable (contains gradients)
inputs = Variable(x_poly)
outputs = Variable(y_norm)

# define a loss list
loss_plot = []
x_loss = []

t_start = time.time()
# begin training
for i in range(10000):
    prediction = model(inputs)
    loss = loss_func(prediction, outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # collect loss per epoch
    if i % 500 == 0:
        loss_plot.append(loss.data.numpy())
        x_loss.append(i)

    if i % 1000 == 0:
        # plot and show learning process
        plt.cla()
        plt.plot(position, y_norm.data.numpy())
        plt.title(i)
        plt.plot(position, prediction.data.numpy())
        plt.text(0, 0, "Loss=%.4f" % loss.data.numpy())
        plt.pause(0.1)
t_end = time.time()

print("training completed")
print("Time elapsed: " + str(t_end - t_start))
plt.show()
plt.plot(x_loss, loss_plot)
plt.title("Loss plot")
plt.show()
