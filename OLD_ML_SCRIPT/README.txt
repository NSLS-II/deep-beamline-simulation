These files facilitate the generation and use of input data for machine learning for SRW. They include:

NSLS-II-CSX-1-beamline-rsOptExport.sh: a shell script that generates input data
NSLS-II-CSX-1-beamline-rsOptExport.py: a python script that runs SRW using the generated data
NSLS-II-CSX-1-beamline-rsOptExport.yml: a YML configuration file for rsopt
README.txt: this README

Also included are auxiliary files required for the specific SRW simulation:
['mirror_1d.dat']

To generate the data, run
    'bash NSLS-II-CSX-1-beamline-rsOptExport.sh'.

This stores the data in the numpy file
    NSLS-II-CSX-1-beamline-rsOptExport.npy

The script will prompt you to then run SRW with that data. To do so later, run
    'python NSLS-II-CSX-1-beamline-rsOptExport.py rsopt_run NSLS-II-CSX-1-beamline-rsOptExport.npy'

Please note:
    - rsopt does the job of running SRW with the stored input data
    - If the machine running SRW has N cores, rsopt will attempt execute in parallel on N - 1 cores
    - SRW will run 10 times; use caution

The final consolidated output data for use in ML is in two numpy files under the "datasets" folder. Each is an array
with 10 entries, one for each run.  
    - beam_intensities.npy: propagated single-electron intensity distribution vs horizontal and vertical position
    - parameters.npy: the inputs used for each run


Intermediate output is also available (note the files may be re-used by workers):

data_files/res_int_pr_se_[n].dat:
    propagated single-e intensity distribution vs horizontal and vertical position

beams/beam_[n].npy:
    same information as in "data_files", but in numpy format

parameters/values_[n].npy:
    values of the parameters

