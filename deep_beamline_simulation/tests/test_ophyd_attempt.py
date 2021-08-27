import pytest
import requests
import deep_beamline_simulation
from pathlib import Path


def test_connection():
    session = requests.Session()
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    assert response_sim_list.status_code == 200


def test_login():
    session = requests.Session()
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    response_auth_guest_login = session.post(
        "http://localhost:8000/auth-guest-login/srw"
    )
    expected = {
        "authState": {
            "avatarUrl": None,
            "displayName": "Guest User",
            "guestIsOnlyMethod": True,
            "isGuestUser": True,
            "isLoggedIn": True,
            "isLoginExpired": False,
            "jobRunModeMap": {"parallel": "1 cores (SMP)", "sequential": "Serial"},
            "method": "guest",
            "needCompleteRegistration": False,
            "paymentPlan": "enterprise",
            "roles": ["adm", "enterprise", "premium"],
            "uid": "78msjrcn",
            "upgradeToPlan": None,
            "userName": None,
            "visibleMethods": [],
        },
        "state": "ok",
    }
    expected["authState"].pop("uid")
    actual_response = response_auth_guest_login.json()
    actual_response["authState"].pop("uid")
    assert expected == actual_response


def test_simulationUpload():
    session = requests.Session()
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    response_auth_guest_login = session.post(
        "http://localhost:8000/auth-guest-login/srw"
    )
    sirepo_simulations_dir = (
        Path(deep_beamline_simulation.__path__[0]).parent / "sirepo_simulations"
    )
    simulation_zip_file_path = sirepo_simulations_dir / "sim_example.zip"
    files = {
        "file": open(str(simulation_zip_file_path), "rb"),
        "folder": (None, "/foo"),
    }
    response_import_file = session.post(
        "http://localhost:8000/import-file/srw", files=files
    )
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    # verify uploaded simulation is in simulation list
    is_sim_uploaded = False
    for sim in response_sim_list.json():
        if sim["name"] == "example":
            is_sim_uploaded = True
    assert is_sim_uploaded


def test_runSimulation():
    session = requests.Session()
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    response_auth_guest_login = session.post(
        "http://localhost:8000/auth-guest-login/srw"
    )
    sirepo_simulations_dir = (
        Path(deep_beamline_simulation.__path__[0]).parent / "sirepo_simulations"
    )
    simulation_zip_file_path = sirepo_simulations_dir / "sim_example.zip"
    files = {
        "file": open(str(simulation_zip_file_path), "rb"),
        "folder": (None, "/foo"),
    }
    response_import_file = session.post(
        "http://localhost:8000/import-file/srw", files=files
    )
    uploaded_sim = response_import_file.json()
    # get sim id from uploaded_sim
    sim_id = uploaded_sim["models"]["simulation"]["simulationId"]
    # set the sim_id to prevent 'simulationId not found' error
    uploaded_sim["simulationId"] = sim_id
    uploaded_sim["report"] = "intensityReport"
    # run simulation here
    response_run_simulation = session.post(
        "http://localhost:8000/run-simulation", json=uploaded_sim
    )
    for i in range(1000):
        state = (response_run_simulation.json())["state"]
        if state == "completed" or state == "error":
            state = (response_run_simulation.json())["state"]
        if state == "completed" or state == "error":
            break
        else:
            response_run_simulation = session.post(
                "http://localhost:8000/run-status",
                json=(response_run_simulation.json())["nextRequest"],
            )
    # compare the summary to avoid comparing the large amount of intensity data
    expected_summary_data = {
        "summaryData": {
            "fieldIntensityRange": [210561.34375, 1.8392213897609216e16],
            "fieldRange": [100.0, 20000.0, 10000, 0.0, 0.0, 1, 0.0, 0.0, 1],
        },
        "title": "On-Axis Spectrum from Filament Electron Beam",
        "x_label": "Photon Energy [eV]",
        "x_range": [100.0, 20000.0],
        "x_units": "eV",
        "y_label": "Intensity (ph/s/.1%bw/mm²)",
        "y_units": "ph/s/.1%bw/mm^2",
        "z_range": [210561.34375, 1.8392213897609216e16],
    }
    actual_summary_data = response_run_simulation.json()["summaryData"]


def test_find_sim_by_name():
    session = requests.Session()
    # call for cookies
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    response_auth_guest_login = session.post(
        "http://localhost:8000/auth-guest-login/srw"
    )
    # call for simulation list
    response_sim_list = session.post(
        "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
    )
    # example name
    given_name = "NSLS-II SRX beamline"
    actual_response = None
    for sim in response_sim_list.json():
        if sim["name"] == given_name:
            actual_response = sim
    # remove changing variables for expected response
    expected_response = {
        "simulationId": "Ralizjbm",
        "name": "NSLS-II SRX beamline",
        "folder": "/Light Source Facilities/NSLS-II/NSLS-II SRX beamline",
        "isExample": True,
        "simulation": {
            "distanceFromSource": 20,
            "documentationUrl": "https://www.bnl.gov/ps/beamlines/beamline.php?r=5-ID",
            "fieldUnits": 1,
            "folder": "/Light Source Facilities/NSLS-II/NSLS-II SRX beamline",
            "horizontalPointCount": 100,
            "horizontalPosition": 0,
            "horizontalRange": 2.5,
            "isExample": True,
            "lastModified": 1630077470514,
            "name": "NSLS-II SRX beamline",
            "notes": "",
            "outOfSessionSimulationId": "qPYbqYWl",
            "photonEnergy": 8000,
            "sampleFactor": 0.1,
            "samplingMethod": "1",
            "simulationId": "Ralizjbm",
            "simulationSerial": 1630077470518433,
            "sourceType": "t",
            "verticalPointCount": 100,
            "verticalPosition": 0,
            "verticalRange": 1.3,
        },
    }
    expected_response.pop("simulationId")
    expected_response["simulation"].pop("lastModified")
    expected_response["simulation"].pop("simulationId")
    expected_response["simulation"].pop("simulationSerial")
    # remove changing variables for actual response
    actual_response.pop("simulationId")
    actual_response["simulation"].pop("lastModified")
    actual_response["simulation"].pop("simulationId")
    actual_response["simulation"].pop("simulationSerial")
    assert expected_response == actual_response
