These files facilitate the generation and use of input data for machine learning for SRW. They include:

NSLS-II-CSX-1-beamline-rsOptExport.sh: a shell script that generates input data
NSLS-II-CSX-1-beamline-rsOptExport.py: a python script that runs SRW using the generated data
NSLS-II-CSX-1-beamline-rsOptExport.yml: a YML configuration file for rsopt
README.txt: this README

Also included are auxiliary files required for the specific SRW simulation:
['mirror_1d.dat']

To generate the data, run
    'bash NSLS-II-CSX-1-beamline-rsOptExport.sh'.

This creates a new directory based on the current time and stores the data in the numpy file
    NSLS-II-CSX-1-beamline-rsOptExport.npy

The script will prompt you to then run SRW with that data. To do so later, run
    'python NSLS-II-CSX-1-beamline-rsOptExport.py rsopt_run NSLS-II-CSX-1-beamline-rsOptExport.npy'

Please note:
    - rsopt does the job of running SRW with the stored input data
    - If the machine running SRW has N cores, rsopt will attempt execute in parallel on N - 1 cores
    - SRW will run 100 times; use caution

The final consolidated input and output data for use in ML is in the hdf5 file "results.h5" under the "datasets" folder.
It has the following structure:
    - "/params": the parameters in the format <element name>_<param name>, e.g.
        "Aperture_horizontalSize"
    - "/paramVals": array of size 100 containing the parameter values for each run
    - "/beamIntensities": array of size 100 containing the propagated single-electron intensity
        distribution vs horizontal and vertical position for each run

Intermediate output is also available:

data_files/res_int_pr_se_[n].dat:
    propagated single-e intensity distribution vs horizontal and vertical position

beams/beam_[n].npy:
    same information as in "data_files", but in numpy format

parameters/values_[n].npy:
    values of the parameters

