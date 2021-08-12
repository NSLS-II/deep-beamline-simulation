import time
import torch
import numpy as np
from torch.nn import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from deep_beamline_simulation.neuralnet import Neural_Net

srx_writer = SummaryWriter()
srx_model = Neural_Net(15,560)
print("Model Structure")
print(srx_model)

training_data = {}
training_data["Intensity-At-Sample-63-3m-Horizontal-Position.csv"] = [
    33.1798, 2, 1,
    34.2608, 0, 0,
    35.6678, 2.4, 1.5,
    50.6572, 0.005, 3,
    62.488, 3, 0.875
]

trainx = list(training_data.values())
trainy = srx_model.parse_data(training_data)

trainx = torch.from_numpy(np.array(trainx)).float()
trainy = torch.from_numpy(np.array(trainy)).float()

train_inputs = Variable(trainx)
train_outputs = Variable(trainy)

optimizer = torch.optim.Adam(srx_model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

loss_plot = []
accuracy_plot = []
file_path1 = "/vagrant/loss.txt"
file1 = open(file_path1, "w")
file_path2 = "/vagrant/accuracy.txt"
file2 = open(file_path2, "w")

t_start = time.time()
for e in range(3000):
    prediction = srx_model(train_inputs)
    loss = loss_func(prediction, train_outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_plot.append(loss.data.numpy())
    points = np.linspace(-200, 200, len(trainy[0]))
    if e % 1000 == 0:
        plt.cla()
        for i in range(len(trainy)):
            plt.cla()
            plt.plot(points, trainy[i], label="original")
            plt.title(e)
            plt.plot(points, prediction.data.numpy()[i], label="predicted")
            plt.text(-200, 1, "Loss=%.4f" % loss.data.numpy())
            plt.text(
                -200,
                0.89,
                "Accuracy=%.4f"
                % srx_model.accuracy(prediction.data.numpy()[i], trainy.numpy()[i]),
            )
            plt.legend(loc="upper right")
            plt.pause(1)
    for i in range(len(trainy)):
        accuracy_plot.append(
            srx_model.accuracy(prediction.data.numpy()[i], trainy.numpy()[i])
        )
print("training completed")
t_end = time.time()
print("Time elapsed: " + str((t_end - t_start) / 60.0))
for element in loss_plot:
    file1.write(str(element) + "\n")
for i in accuracy_plot:
    file2.write(str(i) + "\n")
file1.close()
file2.close()
