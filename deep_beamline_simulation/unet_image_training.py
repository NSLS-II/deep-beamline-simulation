import os
import torch
import torch.optim
import pandas
import numpy as np
from pathlib import Path
from torchinfo import summary
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from u_net import UNet, ImageProcessing, build_dataloaders
from torchinfo import summary

# process the images with image processing class
ip = ImageProcessing([])
filename = "NSLS-II-TES-beamline-rsOptExport-2/rsopt350/datasets/results.h5"
param_count, resized_images = ip.preprocess(filename)

# dataloaders and dataset
training_intensity_dataloader, testing_intensity_dataloader = build_dataloaders('preprocessed_results.h5', 10)

# define model
model = UNet(128, 128, param_count)

# get model summary with 128x128 size images
summary(model, input_data = (torch.ones(2, 1, 128, 128), torch.ones(2, param_count)),
        col_names=('input_size', 'output_size', 'num_params'))


# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

def train(model, optimizer, loss_function, train_dataloader, test_dataloader, epochs):
    '''
    Function for training the model
    '''
    train_loss = []
    test_loss = []

    if torch.cuda.is_available():
        device  = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    for e in range(0, epochs):
        training_loss = 0.0 
        model.train()
        for correct_images, images, input_params in train_dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            correct_images = correct_images.to(device)

            predicted_images  = model(images, input_params)

            loss = loss_function(predicted_images, correct_images)
            loss.backward()
            optimizer.step()

            training_loss += loss.data.item()
        train_loss.append(training_loss)

        testing_loss = 0.0
        model.eval()

        for correct_images, images, input_params in test_dataloader:
            images = images.to(device)
            correct_images = correct_images.to(device)
            predicted_images = model(images, input_params)
            loss = loss_function(predicted_images, correct_images)
            testing_loss += loss.data.item()

        test_loss.append(testing_loss)

        if e % 100 == 0:
            print(
                'Epoch:{}, Training Loss: {:.5f}, Test Loss: {:.5f}'.format(e, training_loss, testing_loss)
                )
    return train_loss, test_loss

sim_count = len(resized_images)
beamline = 'TES'

epoch_count = 200

print(f"training_intensity_dataloader length {len(training_intensity_dataloader)}")
print(f"testing_intensity_dataloader length {len(testing_intensity_dataloader)}")

print(list(training_intensity_dataloader))

train_loss, test_loss = train(
    model,
    optimizer,
    loss_func,
    training_intensity_dataloader,
    testing_intensity_dataloader,
    epochs = epoch_count
    )


plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.title(f"{beamline} {sim_count} Simulations")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
