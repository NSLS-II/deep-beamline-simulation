import torch
from torch.autograd import Variable
from torch.nn import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# define neural network
class Neural_Net(torch.nn.Module):
    def __init__(self):
        super(Neural_Net, self).__init__()
        # define layers
        self.hidden1 = torch.nn.Linear(1, 256)
        self.hidden2 = torch.nn.Linear(256, 512)
        self.hidden3 = torch.nn.Linear(512, 512)
        self.hidden4 = torch.nn.Linear(512, 1024)
        self.hidden5 = torch.nn.Linear(1024, 1024)
        self.hidden6 = torch.nn.Linear(1024, 1024)
        self.hidden7 = torch.nn.Linear(1024, 1024)
        self.hidden8 = torch.nn.Linear(1024, 1024)
        self.hidden9 = torch.nn.Linear(1024, 512)
        self.hidden10 = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, 1)

    def forward(self, x):
        # call layers and use relu activation function
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = F.relu(self.hidden10(x))
        x = self.output(x)
        return x

# create model object
model = Neural_Net()
# display model
print(model)

# get data from csv file
data = np.genfromtxt('Intensity-At-BPM1-38-6904m-Horizontal-Position.csv', delimiter=',')
data = list(data)
data.pop(0)
data = np.array(data)
# split np array
position, intensity = np.split(data, 2, axis=1)
# find max and min for normalization process
min_ints = min(intensity)
max_ints = max(intensity)
# normalize the intensity
norm_intensity = (intensity - min_ints) / (max_ints-min_ints)
# convert numpy array to tensor of input shape
x_pos = torch.from_numpy(position.reshape(-1, 1)).float()
y_norm = torch.from_numpy(norm_intensity.reshape(-1, 1)).float()
y_plot = torch.from_numpy(intensity.reshape(-1,1)).float()
# optimizer and loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()
# change type to Variable (contains gradients)
inputs = Variable(x_pos)
outputs = Variable(y_norm)
#define a loss list
loss_plot = []
x_loss = []
# begin training
for i in range(5000):
    prediction = model(inputs)
    # un-normalize predictions
    np_predictions = prediction.detach().numpy()
    norm_predictions = np_predictions * max_ints
    norm_predictions = Variable(torch.from_numpy(np_predictions.reshape(-1, 1)).float())
    loss = loss_func(prediction, outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # collect loss per epoch
    if i%50 == 0:
        loss_plot.append(loss.data.numpy())
        x_loss.append(i)

    if i % 1000 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x_pos.data.numpy(), y_plot.data.numpy())
        plt.title(i)
        plt.plot(x_pos.data.numpy(), norm_predictions, 'r-', lw =0.5)
        #plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=0.1)
        #plt.text(0, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
        plt.text(0, 0, 'Loss=%.4f' %loss.data.numpy())
        plt.pause(0.1)

plt.show()
plt.scatter(x_loss, loss_plot)
plt.title('Loss plot')
plt.show()
