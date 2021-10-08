import torch
from skimage.io import imread
from torch.utils import data
import deep_beamline_simulation
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split

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
images_dir = (Path(deep_beamline_simulation.__path__[0]).parent / "images")
target_dir = (Path(deep_beamline_simulation.__path__[0]).parent / "targets")

def get_filenames_of_path(path, ext: str = '*'):
    return [file for file in path.glob(ext) if file.is_file()]
    
inputs = get_filenames_of_path(images_dir)
targets = get_filenames_of_path(target_dir)

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
                                    transform=None)

# dataset validation
dataset_valid = createDataset(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=None)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)
