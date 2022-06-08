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
filename = "NSLS-II-TES-beamline-rsOptExport-2/rsopt50/datasets/results.h5"
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

    # want to visualize images and output
    num_rows = round((epochs-1)/50)
    fig, axs = plt.subplots(nrows=num_rows+1, ncols=3)
    count  = 0

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


        if e % 50 == 0:
            print(
                'Epoch:{}, Training Loss: {:.5f}, Test Loss: {:.5f}'.format(e, training_loss, testing_loss)
                )
            images = images.numpy()
            #print(images.shape)
            axs[count, 0].imshow(images[0, 0, :, :], origin="lower")
            axs[count, 0].set_title("initial, e= "+str(e))
            axs[count,0].axis("off")
            axs[count, 1].imshow(correct_images[0, 0, :, :], origin="lower")
            axs[count, 1].set_title("target, e= "+str(e))
            axs[count, 1].axis("off")
            #axs[1].set_xlabel(radius_scale_factor[0, :])
            predicted_images = predicted_images.cpu().detach().numpy()
            axs[count, 2].imshow(predicted_images[0, 0, :, :], origin="lower")
            axs[count, 2].set_title("output, e= "+str(e))
            axs[count, 2].axis("off")
            count += 1
    return train_loss, test_loss

sim_count = len(resized_images)
beamline = 'TES'

epoch_count = 301

print(f"training_intensity_dataloader length {len(training_intensity_dataloader)}")
print(f"testing_intensity_dataloader length {len(testing_intensity_dataloader)}")

train_loss, test_loss = train(
    model,
    optimizer,
    loss_func,
    training_intensity_dataloader,
    testing_intensity_dataloader,
    epochs = epoch_count
    )

plt.figure()
plt.plot(train_loss)
plt.title(f"{beamline} {sim_count} Simulations Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(test_loss)
plt.title(f"{beamline} {sim_count} Simulations Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
