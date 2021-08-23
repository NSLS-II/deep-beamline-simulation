import pytest
import requests

def test_connection():
    session = requests.Session()
    response_sim_list = session.post("http://localhost:8000/simulation-list", json={"simulationType": "srw"})
    expected = '<Response [200]>'
    assert str(response_sim_list) == expected


def test_login():
    session = requests.Session()
    response_sim_list = session.post("http://localhost:8000/simulation-list", json={"simulationType": "srw"})
    response_auth_guest_login = session.post("http://localhost:8000/auth-guest-login/srw")
    expected = {'authState': {'avatarUrl': None,
               'displayName': 'Guest User',
               'guestIsOnlyMethod': True,
               'isGuestUser': True,
               'isLoggedIn': True,
               'isLoginExpired': False,
               'jobRunModeMap': {'parallel': '1 cores (SMP)',
                                 'sequential': 'Serial'},
               'method': 'guest',
               'needCompleteRegistration': False,
               'paymentPlan': 'enterprise',
               'roles': ['adm', 'enterprise', 'premium'],
               'uid': '78msjrcn',
               'upgradeToPlan': None,
               'userName': None,
               'visibleMethods': []},
               'state': 'ok'}
    assert expected['state'] == response_auth_guest_login.json()['state']


def test_simulationUpload():
    session = requests.Session()
    response_sim_list = session.post("http://localhost:8000/simulation-list", json={"simulationType": "srw"})
    response_auth_guest_login = session.post("http://localhost:8000/auth-guest-login/srw")
    files = {"file": open("../../sim_example.zip", "rb"), "folder": (None, "/foo")}
    response_import_file = session.post("http://localhost:8000/import-file/srw",files=files)
    expected = '<Response [200]>'
    assert expected == str(response_import_file)


def test_runSimulation():
    session = requests.Session()
    response_sim_list = session.post("http://localhost:8000/simulation-list", json={"simulationType": "srw"})
    response_auth_guest_login = session.post("http://localhost:8000/auth-guest-login/srw")
    files = {"file": open("../../sim_example.zip", "rb"), "folder": (None, "/foo")}
    response_import_file = session.post("http://localhost:8000/import-file/srw",files=files)
    expected = '<Response [200]>'
    assert expected == str(response_import_file)

@pytest.mark.xfail
def test_find_sim_by_name():
    session = requests.Session()
    response_sim_list = session.post("http://localhost:8000/simulation-list", json={"simulationType": "srw"})
    response_auth_guest_login = session.post("http://localhost:8000/auth-guest-login/srw")
    response_sim_list = session.post("http://localhost:8000/simulation-list", json={"simulationType": "srw"})
    given_name = 'NSLS-II SRX beamline'
    actual = None
    for sim in response_sim_list.json():
        if sim['name'] == given_name:
           actual = sim
    expected = str({'simulationId': 'h91dTvnZ', 'name': 'NSLS-II SRX beamline', 'folder': '/Light Source Facilities/NSLS-II/NSLS-II SRX beamline', 'last_modified': '2021-08-18 17:13', 'isExample': True, 'simulation': {'distanceFromSource': 20, 'documentationUrl': 'https://www.bnl.gov/ps/beamlines/beamline.php?r=5-ID', 'folder': '/Light Source Facilities/NSLS-II/NSLS-II SRX beamline', 'horizontalPointCount': 100, 'horizontalPosition': 0, 'horizontalRange': 2.5, 'isExample': True, 'name': 'NSLS-II SRX beamline', 'notes': '', 'outOfSessionSimulationId': 'qPYbqYWl', 'photonEnergy': 8000, 'sampleFactor': 0.1, 'samplingMethod': '1', 'simulationId': 'h91dTvnZ', 'simulationSerial': 1629306797967505, 'sourceType': 't', 'verticalPointCount': 100, 'verticalPosition': 0, 'verticalRange': 1.3}})
    assert expected == str(actual)


