import torch
import torchvision
import numpy as np
from torch.nn import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import PolynomialFeatures


class Neural_Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        """Define Neural Network"""
        super(Neural_Net, self).__init__()
        self.input = torch.nn.Linear(input_size, 256)
        self.hidden1 = torch.nn.Linear(256, 512)
        self.dropout1 = torch.nn.Dropout(0.6)
        self.hidden2 = torch.nn.Linear(512, 512)
        self.dropout2 = torch.nn.Dropout(0.6)
        self.hidden3 = torch.nn.Linear(512, 256)
        self.output = torch.nn.Linear(256, output_size)

    def forward(self, x):
        """Call layers and use ReLU activation function"""
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x

    def parse_data(self, data):
        """Fix data to be in neural network input format"""
        pos_int = {}
        for file, parameters in data.items():
            position_intensity = np.genfromtxt(file, delimiter=",")
            position_intensity = np.array(position_intensity[1:])
            # remove headers
            pos_int[file] = position_intensity
        parsed = []
        pos_ints_values = pos_int.values()
        for lst in pos_ints_values:
            ints = lst[:, 1]
            parsed.append(np.array(ints))
        trainy = []
        for lst in parsed:
            d = self.normalize_data(lst)
            trainy.append(d)
        return trainy

    def min_max(self, l):
        """Use for normalizing data"""
        minimum = 1e20
        maximum = -1e20
        for i in l:
            if i < minimum:
                minimum = i
            if i > maximum:
                maximum = i
        return minimum, maximum

    def normalize_data(self, data):
        """Normalize data to be [0,1]"""
        norm_data = []
        minimum, maximum = self.min_max(data)
        for i in data:
            norm_data.append((i - minimum) / (maximum - minimum))
        return norm_data

    def accuracy(self, predicted, actual):
        correct = 0
        for i in range(0, len(predicted)):
            if predicted[i] - actual[i] < 0.01:
                correct += 1
        acc = (correct / len(predicted)) * 100
        return acc
