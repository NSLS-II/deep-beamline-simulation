import pandas
import numpy as np
import pylab as pl
from PIL import Image
import matplotlib.pyplot as plt

train_file = 'image_data/Initial-Intensity-33-1798m.csv'
output_file = 'image_data/Intensity-At-Sample-63-3m.csv'


train_numpy = pandas.read_csv(train_file, skiprows=1).to_numpy()
output_numpy = pandas.read_csv(output_file, skiprows=1).to_numpy()

dataset = [train_numpy, output_numpy]

print(train_numpy.shape)
print(output_numpy.shape)

min_height = 10e4
min_length = 10e4

for s in dataset:
    shape = s.shape
    height = shape[0]
    length = shape[1]
    if height < min_height:
        min_height = height 
    if length < min_length:
        min_length = length

def resize(image, height, length):
    image = Image.fromarray(image)
    resized_image = image.resize((length, height))
    resized_image =  np.asarray(resized_image)
    return resized_image

new_image = resize(output_numpy, min_height ,min_length)
print(new_image.shape)

plt.imshow(output_numpy)
plt.show()
plt.imshow(new_image)
plt.show()
