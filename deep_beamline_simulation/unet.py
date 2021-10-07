import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import numpy as np

class DownConvBlock(nn.Module):
    ###
    # Down-Conv Layer
    # 1. 2d Convolution
    # 2. Normalization
    # 3. Leaky ReLU Activation
    ###
    def __init__(self, input_size, output_size, norm=True, dropout=0.0):
        super(DownConvBlock, self).__init__()
        self.layers = [nn.Conv1d(
            in_channels = input_size, out_channels = output_size, kernel_size=1, stride=1)]
        if norm:
            self.layers.append(nn.InstanceNorm2d(output_size))
        self.layers += [nn.LeakyReLU(0.2)]
        if dropout:
            self.layers += [nn.Dropout(dropout)]

    def forward(self, x):
        # *layers unpacks into sequential layers
        op = nn.Sequential(*(self.layers))(x)
        return op


class UpConvBlock(nn.Module):
    ###
    # Up-Conv Layer
    # 1. 2D Transposed Convolution
    # 2. Normalization
    # 3. Leaky ReLU Activation
    ###
    def __init__(self, input_size, output_size, dropout=0.0):
        super(UpConvBlock, self).__init__()
        self.layers = [
            nn.ConvTranspose1d(
                input_size, output_size, kernel_size=1, stride=1
            ),
            nn.InstanceNorm2d(output_size),
            nn.ReLU(),
        ]
        if dropout:
            self.layers += [nn.Dropout(dropout)]

    def forward(self, x, encrypted_inputs):
        x = nn.Sequential(*(self.layers))(x)
        output = torch.cat((x, encrypted_inputs), 1)
        return output


class UNet(nn.Module):
    ###
    # UpSampling and Conv Layer
    # 1. Nearest Neighbor based Upsampling
    # 2. Convolution
    # 3. Tanh Activation
    ###
    # why is the input and output 3?
    def __init__(self, inputs, outputs):
        super(UNet, self).__init__()
        self.DC_l1 = DownConvBlock(41, 143, norm=False)
        self.DC_l2 = DownConvBlock(143,143)
        self.DC_l3 = DownConvBlock(143,143, dropout=0.5)
        self.DC_l4 = DownConvBlock(143,143, dropout=0.5)
        self.DC_l5 = DownConvBlock(143,143, dropout=0.5)
        self.UC_l1 = UpConvBlock(143,143, dropout=0.5)
        self.UC_l2 = UpConvBlock(143,143, dropout=0.5)
        self.UC_l3 = UpConvBlock(143,143, dropout=0.5)
        self.UC_l4 = UpConvBlock(143,143, dropout=0.5)
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv_layer = nn.Conv2d(128, outputs, kernel_size=3, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        encrypted_1 = self.DC_l1(x)
        encrypted_2 = self.DC_l2(encrypted_1)
        encrypted_3 = self.DC_l3(encrypted_2)
        encrypted_4 = self.DC_l4(encrypted_3)
        final = self.DC_l5(encrypted_4)
        encrypted_5 = self.DC_l5(encrypted_4)
        decrypted_1 = self.UC_l1(encrypted_5, encrypted_4)
        decrypted_2 = self.UC_l2(decrypted_1, encrypted_3)
        decrypted_3 = self.UC_l3(decrypted_2, encrypted_2)
        decrypted_4 = self.UC_l4(decrypted_3, encrypted_1)
        final_output = self.upsample_layer(decrypted_4)
        final_output = self.zero_pad(final_output)
        final_output = self.conv_layer(final)
        return self.activation(final)


# kernel_size defined the field of view of the convolution
#kernel_size = 4 # 4x4 pixels
# stride defines the step size of the kernel when traversing an image
#stride = 2 

df = pd.read_csv('test_data/Initial-Intensity-33-1798m.csv', skiprows = 1)
train_data = df.to_numpy()
print(train_data.shape)

df1 = pd.read_csv('test_data/Intensity-At-Sample-63-3m.csv', skiprows = 1)
test_data = df1.to_numpy()
print(test_data.shape)

train_data = Variable(torch.from_numpy(train_data).float())
test_data = Variable(torch.from_numpy(test_data).float())
train_data = train_data.unsqueeze(0)
print(train_data.shape)
train_data = train_data.permute(1,2,0)
print(train_data.shape)

# define model here
model = UNet(1, 10)
print('Model Structure')
print(model)

# load data into dataloader
# using batch size 100, not necessary if single element needed to be used
optimizer = optim.Adam(model.parameters(), lr = 0.01)
loss_function =  nn.MSELoss()

# should be 
# num channels = 1
# batch_size = 1
# [batch_size, num channels, height, width]


for e in range(0,100):
        trained_predictions = model(train_data)
        loss = loss_function(trained_predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
