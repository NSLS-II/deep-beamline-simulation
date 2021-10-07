import torch
from skimage.io import imread
from torch.utils import data
import deep_beamline_simulation
from pathlib import Path

class createDataset(data.Dataset):
    '''
    Class to create a data generator
    inputs: list of inputs
    target: list of target values
    '''
    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype =  torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    '''
    Used to load, transform, and preprocess data
    index should be an integer
    '''
    def __getitem__(self, index):
        input_value = self.inputs[index]
        target_value = self.targets[index]
        # turns data into np arrays
        x,y = imread(input_value), imread(target_value)
        if self.transform is not None:
            x,y = self.transform(x,y)
        # verify data is tensor type
        x,y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        return x,y


# image paths
# file dictionary for importing
images_dir = (Path(deep_beamline_simulation.__path__[0]).parent / "images")
target_dir = (Path(deep_beamline_simulation.__path__[0]).parent / "targets")

in_im1 = str(images_dir / "0cdf5b5d0ce1_01.png")
t_im1 = str(target_dir / "0cdf5b5d0ce1_01.png")
in_im2 = str(images_dir / "0cdf5b5d0ce1_02.png")
t_im2 = str(target_dir / "0cdf5b5d0ce1_02.png")

input_data = [in_im1, in_im2]
target_data = [t_im1, t_im2]

training_dataset = createDataset(inputs=input_data,targets = target_data, transform=None)
training_dataloader = data.DataLoader(dataset=training_dataset, batch_size = 2, shuffle = True)

x,y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')
