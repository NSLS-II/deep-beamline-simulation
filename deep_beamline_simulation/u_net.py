# import statements
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


''' Create the basic block architecture'''
class Block(nn.Module):
    def __init__(self, input_channels = 3, output_channels = 3):
        super().__init__()
        # kernel_size = 3
        self.conv_in = nn.Conv2d(input_channels, output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # kernel_size = 3
        self.conv_out = nn.Conv2d(output_channels, output_channels, 3, stride=1, padding=1)

    def forward(self,x):
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return self.relu(x)


''' similar to standard CNN '''
class Encoder(nn.Module):
    def __init__(self, num_channels = (3,64,128)):
        super().__init__()
        # creates a block for each pair of channels 
        self.enc_blocks = nn.ModuleList([Block(num_channels[i], num_channels[i+1]) for i in range(len(num_channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = []
        for block in self.enc_blocks:
            x = block(x)
            out.append(x)
            x = self.pool(x)
        return out


class Decoder(nn.Module):
    ''' up convolution part '''
    def __init__(self, num_channels=(128,64)):
        super().__init__()
        self.num_channels = num_channels
        # conv transpose for decreasing pairs of channels
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(num_channels[i], num_channels[i+1], 2, 2) for i in range(len(num_channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(num_channels[i], num_channels[i+1]) for i in range(len(num_channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.num_channels)-1):
            x = self.upconvs[i](x)
            encoder_features = self.crop(encoder_features[i],x)
            # concat the unconvs and encoder features
            x = torch.cat([x, encoder_features], dim=1)
            x = self.dec_blocks[i](x)
        return x 

    ''' Downsizing and creating the correct output shape '''
    def crop(self, encoder_features, x):
        _, _, H, W = x.shape
        encoder_features = torchvision.transforms.CenterCrop([H,W])(encoder_features)
        return encoder_features


class UNet(nn.Module):
    def __init__(self, encoder_channels=(3,64,128), decoder_channels=(128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        encoder_features = self.encoder(x)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            # interpolates output if the current output is not the correct output size
            out = F.interpolate(out, out_sz)
        return out


'''
# Check the shape of each block of the unet depending on the input
enc_block = Block(1, 64)
# input shape
x = torch.randn(1, 1, 128, 128)
# output shape
print('block shape')
print(enc_block(x).shape)

# check the shape of the encoder block depending on the inputs
encoder = Encoder()
# input image
x    = torch.randn(1, 3, 128, 128)
ftrs = encoder(x)
print('Encoder Output')
for ftr in ftrs: print(ftr.shape)

decoder = Decoder()
x = torch.randn(1, 128, 28, 28)
print('Decoder Output')
print(decoder(x, ftrs[::-1][1:]).shape)

model = UNet()
print("Model Structure")
print(model)
print('Model Summary')
summary(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

# (number of lists, sub lists, num rows, num columns)
inputs = torch.randn(10, 3, 20, 20)
#print(inputs)
outputs = torch.randn(10, 1, 4, 4)
#print(outputs)

test_in = torch.randn(2, 3, 20, 20)
test_out = torch.randn(2, 1, 4, 4)

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

test_predictions = None
with torch.no_grad():
	model.eval()
	test_predictions = model(test_in)
	test_loss = loss_func(test_predictions, test_out)
	test_plot = test_predictions.data.numpy()[0]

print('Test Loss: ' + str(test_loss))
print(test_predictions)
'''

# Dataloader
random_dataset = []
for i in range(0,100):
    random_dataset.append((torch.from_numpy(np.random.rand(3,20,20).astype('f')), (torch.from_numpy(np.random.rand(1,20,20).astype('f')))))

random_dataset_loader = DataLoader(random_dataset, batch_size=10, shuffle=True)
# Display image and label

'''
train_features, train_labels = next(iter(random_dataset_loader))
print(f"Feature batch shape: {train_features.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

'''

# Training
model = UNet()
print(model)
summary(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

epochs = 500
loss_list = []
for e in range(epochs):
    training_loss = 0.0
    for img, label in random_dataset_loader:
        prediction = model(img)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.data.item()
        loss_list.append(training_loss)
    if e % 10 == 0:
        print('Epoch: {}, Training Loss: {:.2f}'.format(e, training_loss))

#plt.plot(loss_list)
#plt.show()




