import os
import torch
import pytest
import pandas
from pathlib import Path
import deep_beamline_simulation
from deep_beamline_simulation.u_net import UNet, ImageProcessing

def test_predictions():
	'''
	Simple test function to ensure the model returns an 
	output image when given and input image
	'''
	dbs_dir = (Path(deep_beamline_simulation.__path__[0]).parent / 'deep_beamline_simulation/image_data')
	train_file = dbs_dir / 'Initial-Intensity-33-1798m.csv'
	output_file = dbs_dir / "Intensity-At-Sample-63-3m.csv"

	# read csv using pandas, skip first row for headers
	train_numpy = pandas.read_csv(train_file, skiprows=1).to_numpy()
	output_numpy = pandas.read_csv(output_file, skiprows=1).to_numpy()

	# create a list to store in ImageProcessing class
	image_list = [train_numpy, output_numpy]
	ip = ImageProcessing(image_list)

	# find the size of the smallest image
	height, length = ip.smallest_image_size()

	# resize all images bigger than the smallest image size
	train_numpy = ip.resize(train_numpy, height, length)
	resized_output_image = ip.resize(output_numpy, height, length)

	# normalize all training data
	train_numpy = ip.normalize_image(train_numpy)
	resized_output_image = ip.normalize_image(resized_output_image)

	# convert each numpy array into a tensor
	train_image = torch.from_numpy(train_numpy.astype("f"))
	output_image = torch.from_numpy(resized_output_image.astype("f"))

	# create model
	model = UNet()

	train_image = train_image[:, :, None, None]
	output_image = output_image[:, :, None, None]

	# define optimizer and loss function
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	loss_func = torch.nn.MSELoss()

	predictions = []
	# loop through many epochs
	for e in range(0, 3):
	    predictions = model(train_image)
	    loss = loss_func(predictions, output_image)
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()

	assert predictions.size != 0
