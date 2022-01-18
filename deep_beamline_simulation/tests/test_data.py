import pytest
import numpy as np
from pathlib import Path
import deep_beamline_simulation
from deep_beamline_simulation.data_collection import open_beam, open_dat, load_params

def test_beam():
    dbs_dir = (
        Path(deep_beamline_simulation.__path__[0]).parent
        / "deep_beamline_simulation/NSLS-II-CSX-1-beamline-rsOptExport"
    )
    beam = open_beam(dbs_dir / 'beams/beam_0.npy')
    actual = ((beam[0])[0])[0]
    expected = 14255822848.0
    assert actual == expected


def test_dat():
    dbs_dir = (
        Path(deep_beamline_simulation.__path__[0]).parent
        / "deep_beamline_simulation/NSLS-II-CSX-1-beamline-rsOptExport"
    )
    data = open_dat(dbs_dir / 'data_files/res_int_se_0.dat')
    assert len(data) != 0


def test_params():
    dbs_dir = (
        Path(deep_beamline_simulation.__path__[0]).parent
        / "deep_beamline_simulation/NSLS-II-CSX-1-beamline-rsOptExport"
    )
    actual_params = load_params(dbs_dir / 'datasets/parameters.npy')
    expected_params = [[0.9968614314798989, 1.0018781174363909], [0.9993364442216538, 0.9995388167340034], [0.9980048862325155, 1.0026704675101783], [0.9949587242051728, 0.9957203244934422], [1.0013248974747047, 0.9963023325726318], [0.9956001258122992, 0.9970923385947688], [1.0004611139658437, 0.9983455607270431], [1.0037590352826038, 1.0006852195003968], [1.0022301263525177, 1.0041981014890848], [1.0045544256324548, 1.0035586898284456]]
    assert np.array_equal(expected_params, actual_params)
