import torch
import torch.nn as nn
import torch.nn.functional as F


class DownConvBlock(nn.Module):
    ###
    # Down-Conv Layer
    # 1. 2d Convolution
    # 2. Normalization
    # 3. Leaky ReLU Activation
    ###
    def __init__(self, input_size, output_size, k_size, stride, norm=True, dropout=0.0):
        super(DownConvBlock, self).__init__()
        self.layers = nn.Conv2D(
            inputs, outputs, kernel_size=k_size, stride=stride, padding=1
        )
        if norm:
            self.layers.append(nn.InstanceNorm2d(output_size))
        self.layers += [nn.LeakyReLU(0.2)]
        if dropout:
            self.layers += nn.Dropout(dropout)

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
    def __init__(self, input_size, output_size, kernel_size, stride, dropout=0.0):
        super(UpConvBlock, self).__init__()
        self.layers = [
            nn.ConvTranspose2d(
                inputs, outputs, kernel_size=kernel_size, stride=stride, padding=1
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
    def __init__(self, inputs, outputs, kern_size):
        super(UNet, self).__init__()
        self.DC_l1 = DownConvBlock(inputs, 64, norm=False)
        self.DC_l2 = DownConvBlock(64, 128)
        self.DC_l3 = DownConvBlock(128, 256, dropout=0.5)
        self.DC_l4 = DownConvBlock(256, 512, dropout=0.5)
        self.DC_l5 = DownConvBlock(512, 512, dropout=0.5)
        self.UC_l1 = UpConvBlock(512, 512, dropout=0.5)
        self.UC_l2 = UpConvBlock(512, 256, dropout=0.5)
        self.UC_l3 = UpConvBlock(256, 128, dropout=0.5)
        self.UC_l4 = UpConvBlock(128, 64, dropout=0.5)
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv_layer = nn.Conv2d(128, outputs, kernel_size=kern_size, padding=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        encrypted_1 = self.DC_l1(x)
        encrypted_2 = self.DC_l2(encrypted_1)
        encrypted_3 = self.DC_l3(encrypted_2)
        encrypted_4 = self.DC_l4(encrypted_3)
        encrypted_5 = self.DC_l5(encrypted_4)
        decrypted_1 = self.UC_l1(encrypted_5, encrypted_4)
        decrypted_2 = self.UC_l2(decrypted_1, encrypted_3)
        decrypted_3 = self.UC_l3(decrypted_2, encrypted_2)
        decrypted_4 = self.UC_l4(decrypted_3, encrypted_1)
        final_output = self.upsample_layer(decrypted_4)
        final_output = self.zero_pad(final_output)
        final_output = self.conv_layer(final)
        return self.activation(final)
