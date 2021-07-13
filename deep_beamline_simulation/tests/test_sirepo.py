from sirepo_bluesky.sirepo_bluesky import SirepoBluesky
from deep_beamline_simulation.sirepo_data import sirepo_data
import os
import pytest
import vcr

cassette_location = os.path.join(
    os.path.dirname(deep_beamline_simulation.tests.__file__), "vcr_cassettes"
)


def test_auth(sim_id, server_name):
    sb = SirepoBluesky("server_name")
    data, schema = sb.auth("srw", sim_id)
    assert "beamline" in data["models"]


@vcr.use_cassette(f"{cassette_location}/test_smoke_sirepo.yml")
def test_auth_vcr():
    test_auth(sim_id="avw4qnNw", server_name="http://10.10.10.10:8000")


def test_simids():
    sb = SirepoBluesky("http://localhost:8000")
    data, schema = sb.auth("srw", "avw4qnNw")
    sd = sirepo_data("avw4qnNw")
    id_dict = sd.get_simids()
    # sim_ids tested are the original created ones when srw is first opened
    # needs to be modififed so other sims can be added
    actual_dict = {
        "Bending Magnet Radiation": "jDcdb2Ny",
        "Boron Fiber (CRL with 3 lenses)": "z3KMvWYT",
        "Boron Fiber (CRL with 4 lenses)": "S7DcAc5g",
        "Compound Refractive Lens (CRL)": "gsp8i719",
        "Diffraction by an Aperture": "zFJ9LE0c",
        "Double Crystal Monochromator (selecting)": "dVkEEQIP",
        "Double Crystal Monochromator (truncating)": "PRzTDjYm",
        "Ellipsoidal Undulator Example": "uBVEFRWW",
        "Focusing Bending Magnet Radiation": "fihzhjmM",
        "Gaussian X-ray Beam Through Perfect CRL": "s8lmNptn",
        "Gaussian X-ray beam through a Beamline containing Imperfect Mirrors": "lBgxz6HM",
        "Idealized Free Electron Laser Pulse": "HQc82jbv",
        "LCLS SXR beamline": "ut25hqgV",
        "LCLS SXR beamline - Simplified": "Yu4BSCuT",
        "Mask example": "PkLAus8v",
        "NSLS-II 3PW QAS beamline": "Oe1KX5I8",
        "NSLS-II CHX beamline": "ZwLGbPrN",
        "NSLS-II CHX beamline (tabulated)": "bFrCjcwk",
        "NSLS-II CSX-1 beamline": "apItt8YH",
        "NSLS-II ESM beamline": "zbgAixO8",
        "NSLS-II ESM beamline (tabulated undulator)": "qfqHYNy8",
        "NSLS-II FMX beamline": "Xd0r57lH",
        "NSLS-II FMX beamline (tabulated)": "QW2CxGwo",
        "NSLS-II HXN beamline": "kpeCVLzR",
        "NSLS-II HXN beamline: SSA closer": "IVoFAyNk",
        "NSLS-II SMI beamline": "0dFQhOZb",
        "NSLS-II SRX beamline": "uaq7GPvv",
        "Polarization of Bending Magnet Radiation": "2IrhuKLs",
        "Sample from Image": "M0Ed8p4x",
        "Soft X-Ray Undulator Radiation Containing VLS Grating": "a0EaYV8Z",
        "Tabulated Undulator Example": "oy1s9g03",
        "Undulator Radiation": "f0s7zHqq",
        "Young's Double Slit Experiment": "9f5gONQs",
        "Young's Double Slit Experiment (green laser)": "l5NrcWqX",
        "Young's Double Slit Experiment (green laser, no lens)": "5jqPY7Wb",
        "example": "avw4qnNw",
    }
    assert id_dict == actual_dict


def test_components():
    sb = SirepoBluesky("http://localhost:8000")
    data, schema = sb.auth("srw", "avw4qnNw")
    sd = sirepo_data("avw4qnNw")
    component_dict = sd.get_components()
    actual_dict = {
        "Aperture": {
            "horizontalOffset": 0,
            "horizontalSize": 1,
            "id": 4,
            "position": 25,
            "shape": "r",
            "type": "aperture",
            "verticalOffset": 0,
            "verticalSize": 1,
        },
        "Watchpoint": {"id": 5, "position": 27, "type": "watch"},
    }
    assert component_dict == actual_dict
