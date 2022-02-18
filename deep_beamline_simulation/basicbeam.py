import time
import torch
import numpy as np
from torch.nn import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from deep_beamline_simulation.neuralnet import Neural_Net

writer = SummaryWriter()
model = Neural_Net(3, 70)
print("Model Structure")
print(model)

training_data = {}
# training_data["210301_horpos.csv"] = [21.0, 0.3, 0.1]
training_data["../test_data/210051_horpos.csv"] = [21.0, 0.05, 1]

trainx = list(training_data.values())
trainy = model.parse_data(training_data)

testing_data = {}
testing_data["../test_data/2100202_horpos.csv"] = [21.0, 0.02, 0.2]
testx = list(testing_data.values())
testy = model.parse_data(testing_data)

trainx = torch.from_numpy(np.array(trainx)).float()
trainy = torch.from_numpy(np.array(trainy)).float()
testx = torch.from_numpy(np.array(testx)).float()
testy = torch.from_numpy(np.array(testy)).float()

train_inputs = Variable(trainx)
train_outputs = Variable(trainy)
test_inputs = Variable(testx)
test_outputs = Variable(testy)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

loss_plot = []
accuracy_plot = []

file_path1 = "/vagrant/loss.txt"
file_path2 = "/vagrant/accuracy.txt"
with open(file_path1, "r+") as f:
    f.truncate(0)
with open(file_path2, "r+") as f:
    f.truncate(0)

file1 = open(file_path1, "w")
file2 = open(file_path2, "w")

t_start = time.time()
for e in range(4500):
    prediction = model(train_inputs)
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
                % model.accuracy(prediction.data.numpy()[i], trainy.numpy()[i]),
            )
            plt.legend(loc="upper right")
            plt.pause(1)
    for i in range(len(trainy)):
        accuracy_plot.append(
            model.accuracy(prediction.data.numpy()[i], trainy.numpy()[i])
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

test_plot = []
with torch.no_grad():
    model.eval()
    model_pred = model(testx)
    test_loss = loss_func(model_pred, testy)
test_plot = model_pred.data.numpy()[0]
plt.show()
plt.plot(points, test_plot, label="predicted")
plt.plot(points, testy[0], label="actual")
plt.text(-200, 0.88, "Loss=%.4f" % test_loss)
plt.text(-200, 0.75, "Accuracy=%.4f" % model.accuracy(test_plot, testy[0]))
plt.title("Model Predictions on Test Data")
plt.legend(loc="upper right")
plt.show()
