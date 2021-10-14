import numpy as np
from typing import List, Callable
from skimage.io import imread
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from pathlib import Path
import deep_beamline_simulation

def create_dense_target(tar: np.ndarray):
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx
    return dummy


def normalize(inp: np.ndarray):
    """Squash image input to the value range [0, 1] (no clipping)"""
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out


class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar


class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])



class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


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


# image paths
# file dictionary for importing
images_dir = (Path(deep_beamline_simulation.__path__[0]).parent / "inputs")
target_dir = (Path(deep_beamline_simulation.__path__[0]).parent / "targets")

def get_filenames_of_path(path, ext: str = '*'):
    return [str(file) for file in path.glob(ext) if file.is_file()]
    
inputs = get_filenames_of_path(images_dir)
#print(inputs)
targets = get_filenames_of_path(target_dir)
#print(targets)

# training transformations and augmentations
transforms = ComposeDouble([
    FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
    FunctionWrapperDouble(normalize)
])

random_seed = 42
train_size = 0.8

inputs_train, inputs_valid = train_test_split(
    inputs,
    random_state=random_seed,
    train_size=train_size,
    shuffle=True)

targets_train, targets_valid = train_test_split(
    targets,
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

