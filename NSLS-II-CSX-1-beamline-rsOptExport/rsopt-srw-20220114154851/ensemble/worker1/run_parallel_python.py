from pykern.pkrunpy import run_path_as_module

input_dict = {

    'Aperture_horizontalSize': 1.0033340977809262,

    'Aperture_verticalSize': 1.0035035087008957,


}

m = run_path_as_module('NSLS-II-CSX-1-beamline-rsOptExport.py')

m.set_rsopt_params_dummy(**input_dict)