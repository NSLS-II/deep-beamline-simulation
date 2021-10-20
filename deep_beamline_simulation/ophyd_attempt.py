import pprint
import requests
import time as ttime

from pathlib import Path

import deep_beamline_simulation


print("################################################")
print("ophyd attempt")
print("################################################")

# create session with requests
session = requests.Session()

# get cookies by viewing simulation-list
response_sim_list = session.post(
    "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
)
# output of simulation list
print(f"output from {response_sim_list.url}")
pprint.pprint(dict(response_sim_list.headers))
pprint.pprint(response_sim_list.json())
input("press enter to continue")

# login as a guest
print("\nguest login")
response_auth_guest_login = session.post("http://localhost:8000/auth-guest-login/srw")
# output from logging in
print(f"output from {response_auth_guest_login.url}")
pprint.pprint(response_auth_guest_login.json())
input("press enter to continue")

print("\nimport sim_example.zip")
# file dictionary for importing
sirepo_simulations_dir = (
    Path(deep_beamline_simulation.__path__[0]).parent / "sirepo_simulations"
)
simulation_zip_file_path = sirepo_simulations_dir / "sim_example.zip"
print(f"path to sim_example.zip: {simulation_zip_file_path}")
files = {"file": open(str(simulation_zip_file_path), "rb"), "folder": (None, "/foo")}

print(f"uploading {simulation_zip_file_path}")
# post the file to be imported with the dict
response_import_file = session.post(
    "http://localhost:8000/import-file/srw",
    files=files,
)

# output from importing a file
print(f"output from {response_import_file.url}")
# the json from the request is the data we need
uploaded_sim = response_import_file.json()
pprint.pprint(uploaded_sim)
#pprint.pprint(f"uploaded simulation:\n{uploaded_sim}")
input("press enter to continue")

print("\nlook for the uploaded simulation in the list of simulations")
# verify imported sim is in sim list
response_sim_list = session.post(
    "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
)
# view sim list (NOT FOR RUNNING SIM)
pprint.pprint(dict(response_sim_list.headers))
print(f"response from {response_sim_list.url}")
sim_list_results = response_sim_list.json()

simulation_folder = None
for sim_details in sorted(sim_list_results, key=lambda sim_details: sim_details["folder"]):
    if simulation_folder != sim_details["folder"]:
        print(f"simulation folder: {sim_details['folder']}")
        simulation_folder = sim_details["folder"]
    print(f"  {sim_details['simulationId']} : {sim_details['name']}")

input("press enter to continue")

#pprint.pprint(sim_list_results)
# get sim id from uploaded_sim
sim_id = uploaded_sim["models"]["simulation"]["simulationId"]
print("Simulation ID of uploaded sim: " + str(sim_id))

# set the sim_id to prevent 'simulationId not found' error
uploaded_sim["simulationId"] = sim_id
uploaded_sim["report"] = "intensityReport"

print(f"run simulation")
# run simulation here
response_run_simulation = session.post(
    "http://localhost:8000/run-simulation", json=uploaded_sim
)
print(f"run-simulation response:\n{pprint.pformat(response_run_simulation.json())}")

response_run_status = None
for i in range(1000):
    print("  waiting for simulation to end")
    response_run_status = session.post(
        "http://localhost:8000/run-status",
        json=(response_run_simulation.json())["nextRequest"],
    )
    state = (response_run_status.json())["state"]
    if state == "completed" or state == "error":
        print(f"simulation has ended in state '{state}'")
        break
    else:
        print(f"  run-status response:\n{pprint.pformat(response_run_status.json())}\n")
    print(f"waiting {response_run_status['nextRequestSeconds']}s")
    ttime.sleep(response_run_status["nextRequestSeconds"])

print(f"final run status:")
print(response_run_status.url)
# the 'points' value is a list, which is often very long
# I want to print some but not all of that list
# so I'll make it shorter here
response_run_status_details = response_run_status.json()
response_run_status_details["points"] = response_run_status_details["points"][:100]
pprint.pprint(response_run_status_details, compact=True)
