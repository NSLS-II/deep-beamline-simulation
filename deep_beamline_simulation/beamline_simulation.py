import json
import pprint
import requests
import time as ttime

from pathlib import Path

import deep_beamline_simulation


print("################################################")
print("example beamline simulation")
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
sim_id_to_name = {}
for sim_details in sorted(
    sim_list_results, key=lambda sim_details: sim_details["folder"]
):
    if simulation_folder != sim_details["folder"]:
        print(f"simulation folder: {sim_details['folder']}")
        simulation_folder = sim_details["folder"]
    print(f"  {sim_details['simulationId']} : {sim_details['name']}")
    sim_id_to_name[sim_details["name"]] = sim_details["simulationId"]

pprint.pprint(sim_id_to_name)
print(f"TES simulation id: {sim_id_to_name['NSLS-II TES beamline']}")
input("press enter to continue")

print(f"try simulation-data")
response_simulation_data = session.post(
    f"http://localhost:8000/simulation/srw/{sim_id_to_name['NSLS-II TES beamline']}/0"
)
print(f"simulation-data url: {response_simulation_data.url}")
pprint.pprint(response_simulation_data.json())

# print(f"run simulation")
# # run simulation here
# response_run_simulation = session.post(
#     "http://localhost:8000/run-simulation", json={"simulationId": sim_id_to_name['NSLS-II TES beamline']}
# )
# print(f"run-simulation response:\n{pprint.pformat(response_run_simulation.json())}")
