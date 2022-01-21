from pykern.pkrunpy import run_path_as_module

input_dict = {

    'Aperture_size_0': 1.0045544256324548,

    'Aperture_size_1': 1.0035586898284456,


}

m = run_path_as_module('NSLS-II-CSX-1-beamline-rsOptExport.py')

m.set_rsopt_params_dummy(**input_dict)