import re
import os
import time
import torch
import torchvision
import numpy as np
from torch.nn import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import PolynomialFeatures

class Neural_Net(torch.nn.Module):
    def __init__(self):
        ''' Define Neural Network '''
        super(Neural_Net,self).__init__()
        self.input = torch.nn.Linear(3,256)
        self.hidden1 = torch.nn.Linear(256, 512)
        self.dropout1 = torch.nn.Dropout(0.6)
        self.hidden2 = torch.nn.Linear(512,512)
        self.dropout2 = torch.nn.Dropout(0.6)
        self.hidden3 = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, 70)

    def forward(self, x):
        ''' Call layers and use ReLU activation function '''
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x

    def parse_data(self, data):
        ''' Fix data to be in neural network input format '''
        pos_int = {}
        for file, parameters in data.items():
            position_intensity = np.genfromtxt(file, delimiter=',')
            position_intensity = np.array(position_intensity[1:])
            # remove headers
            pos_int[file] = position_intensity
        parsed = []
        pos_ints_values = pos_int.values()
        for lst in pos_ints_values:
            ints = lst[:, 1]
            parsed.append(np.array(ints))
        return parsed

    def min_max(self, l):
        ''' Use for normalizing data '''
        minimum = 1e20
        maximum = -1e20
        for i in l:
            if i < minimum:
                minimum = i
            if i > maximum:
                maximum = i
        return minimum, maximum

    def normalize_data(self, data):
        ''' Normalize data to be [0,1] '''
        norm_data = []
        minimum, maximum = self.min_max(data)
        for i in data:
            norm_data.append((i-minimum)/ (maximum-minimum))
        return norm_data

    def accuracy(self, predicted, actual):
        correct = 0
        for i in range(0, len(predicted)):
             if predicted[i]-actual[i] < 0.01:
                 correct+=1
        acc = (correct/len(predicted))*100
        return acc


def main():
    writer =  SummaryWriter()
    model = Neural_Net()
    print('Model Structure')
    print(model)

    training_data = {}
    #training_data["210301_horpos.csv"] = [21.0, 0.3, 0.1]
    training_data['210051_horpos.csv'] = [21.0, 0.05, 1]

    trainx = list(training_data.values())
    temp_trainy = model.parse_data(training_data)
    trainy = []
    for lst in temp_trainy:
        d = model.normalize_data(lst)
        trainy.append(d)

    testing_data = {}
    testing_data['2100202_horpos.csv'] = [21.0, 0.02, 0.2]
    testx = list(testing_data.values())
    temp_testy = model.parse_data(testing_data)
    testy = []
    for lst in temp_testy:
        d = model.normalize_data(lst)
        testy.append(d)

    trainx = torch.from_numpy(np.array(trainx)).float()
    trainy = torch.from_numpy(np.array(trainy)).float()
    testx = torch.from_numpy(np.array(testx)).float()
    testy = torch.from_numpy(np.array(testy)).float()

    optimizer = torch.optim.Adam(model.parameters(), lr =0.1)
    loss_func = torch.nn.MSELoss()

    train_inputs = Variable(trainx)
    train_outputs =  Variable(trainy)
    test_inputs = Variable(testx)
    test_outputs =  Variable(testy)

    loss_plot = []
    accuracy_plot = []
    file_path1='/vagrant/loss.txt'
    file1 = open(file_path1, 'w')
    file_path2='/vagrant/accuracy.txt'
    file2 = open(file_path2, 'w')

    t_start = time.time()
    for e in range(5000):
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
                plt.plot(points, trainy[i], label = 'original')
                plt.title(e)
                plt.plot(points, prediction.data.numpy()[i], label='predicted')
                plt.text(-200, 1, 'Loss=%.4f' % loss.data.numpy())
                plt.text(-200, 0.89, 'Accuracy=%.4f' % model.accuracy(prediction.data.numpy()[i], trainy.numpy()[i]))
                plt.legend(loc = 'upper right')
                plt.pause(1)
        for i in range(len(trainy)):
            accuracy_plot.append(model.accuracy(prediction.data.numpy()[i], trainy.numpy()[i]))
    print('training completed')
    t_end = time.time()
    print('Time elapsed: ' + str((t_end-t_start)/60.0))
    for element in loss_plot:
        file1.write(str(element)+'\n')
    for i in accuracy_plot:
        file2.write(str(i)+'\n')
    file1.close()
    file2.close()

    test_plot = []
    with torch.no_grad():
        model.eval()
        model_pred = model(testx)
        test_loss = loss_func(model_pred, testy)
    test_plot = model_pred.data.numpy()[0]
    plt.show()
    plt.plot(points, test_plot, label='predicted')
    plt.plot(points, testy[0], label='actual')
    plt.text(-200,0.88, 'Loss=%.4f' % test_loss)
    plt.text(-200,0.75, 'Accuracy=%.4f' % model.accuracy(test_plot, testy[0]))
    plt.title('Model Predictions on Test Data')
    plt.legend(loc = 'upper right')
    plt.show()

if __name__ == '__main__':
    main()

