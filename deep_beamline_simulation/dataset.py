import numpy as np
from typing import List, Callable
from skimage.io import imread
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import pathlib
from functools import partial
import deep_beamline_simulation

def dense_target(target):
    classes = np.unique(target)
    dense = np.zeros_like(target)
    for index, value in enumerate(classes):
        mask = np.where(target == value)
        dense[mask] = index
    return dense

def normalize(inputs):
    """ Normalize input value range [0, 1] """
    norm_inputs = (inputs - np.min(inputs)) / np.ptp(inputs)
    return norm_inputs

class String_Representation:
    """ string representation of an object """
    def __String_Representation__(self): return f'{self.__class__.__name__}: {self.__dict__}'

class FunctionWrapperDouble(String_Representation):
    """A function wrapper that returns a partial for an input-target pair."""
    def __init__(self, function, input = True, target = False, *args, **kwargs):
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, input_array, target_dict):
        if self.input: input_array = self.function(input_array)
        if self.target: target_dict = self.function(target_dict)
        return input_array, target_dict

class Compose:
    """Baseclass - composes several transforms together."""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    def __repr__(self): return str([transform for transform in self.transforms])

class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""
    def __call__(self, input_array, target_dict):
        for t in self.transforms:
            input_array, target_dict = t(input_array, target_dict)
        return input_array, target_dict


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
        x,y = imread(str(input_value)), imread(str(target_value))
        if self.transform is not None:
            x,y = self.transform(x,y)
        # verify data is tensor type
        x,y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        return x,y

# find root directory using Pathlib
root = pathlib.Path.cwd() / 'Carvana'

# get individual file names and return the list
def get_filenames_of_path(path, ext: str = '*'):
    return [file for file in path.glob(ext) if file.is_file()]
    
input_files = get_filenames_of_path(root / 'inputs')
target_files = get_filenames_of_path(root / 'targets')

# training transformations and augmentations
transforms = ComposeDouble([
    FunctionWrapperDouble(dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize)
])

random_seed = 42
train_size = 0.8

inputs_train, inputs_valid = train_test_split(
    input_files,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    target_files,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

# dataset training
dataset_train = createDataset(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms)

# dataset validation
dataset_valid = createDataset(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)

# verify we have the correct format
batch = dataset_train[0]
x, y = batch

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

