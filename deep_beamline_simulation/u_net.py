# import statements
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary
import matplotlib.pyplot as plt

class Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3)

    def forward(self,x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

enc_block = Block(1, 64)
''' input shape '''
x = torch.randn(1, 1, 128, 128)
''' output shape '''
print('block shape')
print(enc_block(x).shape)

class Encoder(nn.Module):
    ''' DownSampling Part'''
    def __init__(self, channels = (3,64,128)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs =[]
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

encoder = Encoder()
# input image
x    = torch.randn(1, 3, 128, 128)
ftrs = encoder(x)
print('Encoder Output')
for ftr in ftrs: print(ftr.shape)

class Decoder(nn.Module):
    ''' up convolution part '''
    def __init__(self, channels=(128,64)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i],x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x 

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H,W])(enc_ftrs)
        return enc_ftrs

decoder = Decoder()
x = torch.randn(1, 128, 28, 28)
print('Decoder Output')
print(decoder(x, ftrs[::-1][1:]).shape)

class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128), dec_chs=(128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out


model = UNet()
print("Model Structure")
print(model)
print('Model Summary')
summary(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

inputs = torch.randn(1, 3, 20, 20)
outputs = torch.randn(1, 1, 4, 4)

test_in = torch.randn(1, 3, 20, 20)
test_out = torch.randn(1, 1, 4, 4)

# sanity check
#print(model(inputs).shape)
#print(outputs.shape)

for e in range(0,10):
	predictions = model(inputs)
	loss = loss_func(predictions, outputs)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

print('Test Loss: ' + str(loss))

with torch.no_grad():
	model.eval()
	test_predictions = model(test_in)
	test_loss = loss_func(test_predictions, test_out)
	test_plot = test_predictions.data.numpy()[0]

print('Test Loss: ' + str(test_loss))

