import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def open_beam(filename):
    data = np.load(filename)
    plt_data = data[0, :, :]
    return data


def open_dat(filename):
    first_ten_lines = pd.read_csv(filename, nrows=9)
    horizontal_position = str(first_ten_lines.loc[5])
    vertical_position = str(first_ten_lines.loc[8])
    horizontal = re.findall("[0-9]{2,300}", horizontal_position)
    horizontal_shape = int(horizontal[0])
    vertical = re.findall("[0-9]{2,300}", vertical_position)
    vertical_shape = int(vertical[0])

    data = pd.read_csv(filename, skiprows=10, header=None)
    data = data.values.copy()
    data.resize(vertical_shape, horizontal_shape)
    shaped_data = pd.DataFrame(data)
    return shaped_data


def load_params(filename):
    params = np.load(filename)
    p = params.tolist()
    return p


"""
data = open_dat('NSLS-II-CSX-1-beamline-rsOptExport/data_files/res_int_se_0.dat')
print('Dat')
print(data.head())

beam = open_beam('NSLS-II-CSX-1-beamline-rsOptExport/beams/beam_0.npy')
print('Beam')
print(beam)

params = load_params('NSLS-II-CSX-1-beamline-rsOptExport/datasets/parameters.npy')
print('Params')
print(params)
"""
