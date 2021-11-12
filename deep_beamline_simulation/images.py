import os
import pandas
import deep_beamline_simulation
from deep_beamline_simulation.u_net import Block, Encoder, Decoder, UNet
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchinfo import summary
from torch.utils.data import DataLoader


# file dictionary for importing
im_data = (
    Path(deep_beamline_simulation.__path__[0]).parent
    / "deep_beamline_simulation/image_data"
)

'''
images = []
for filename in os.listdir(im_data):
    im_data_path = im_data / filename
    images.append(im_data_path)

fig, axes = plt.subplots(1, 8)
fig.suptitle("Varying Radius of Surface Curvature", fontsize=20)

image_data = []
for i in range(0, len(images)):
    image = pandas.read_csv(images[i], skiprows=1)
    image_data.append( (torch.tensor(image.to_numpy()), torch.from_numpy(np.random.rand(1, 20, 20).astype("f"))) )
    axes[i].imshow(image, aspect="auto")
#plt.show()

image_dataset = DataLoader(image_data, batch_size=1, shuffle=True)
#print(image_data)
'''
train_file = im_data / 'Initial-Intensity-33-1798m.csv'
output_file = im_data / 'Intensity-At-Sample-63-3m.csv'
train_image = torch.tensor( pandas.read_csv(train_file, skiprows=1).to_numpy() )
output_image = torch.tensor( pandas.read_csv(output_file, skiprows=1).to_numpy() )

model = UNet()
#print(model)
#summary(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

epochs = 5
loss_list = []
training_loss = 0.0

for e in range(0, 10):
    predictions = model(train_image)
    loss = loss_func(predictions, output_image)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Test Loss: " + str(loss))
test_predictions = None
with torch.no_grad():
    model.eval()
    test_predictions = model(test_in)
    test_loss = loss_func(test_predictions, test_out)
    test_plot = test_predictions.data.numpy()[0]
print("Test Loss: " + str(test_loss))
print(test_predictions)






