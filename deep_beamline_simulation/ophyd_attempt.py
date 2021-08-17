import pprint
import requests


# create session with requests
session = requests.Session()

# get cookies by viewing simulation-list
response_sim_list = session.post(
    "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
)

# output of simulation list
print(response_sim_list.url)
pprint.pprint(dict(response_sim_list.headers))

# login as a guest
response_auth_guest_login = session.post("http://localhost:8000/auth-guest-login/srw")

# output from logging in
print()
print(response_auth_guest_login.url)
pprint.pprint(response_auth_guest_login.json())

# file dictionary for importing
files = {"file": open("../example-2.zip", "rb"), "folder": (None, "/foo")}

# post the file to be imported with the dict
response_import_file = session.post(
    "http://localhost:8000/import-file/srw",
    files=files,
)

# output from importing a file
print()
print(response_import_file.url)
print(response_import_file)
# the json from the request is the data we need
uploaded_sim = response_import_file.json()
pprint.pprint(uploaded_sim)

# verify imported sim is in sim list
response_sim_list = session.post(
    "http://localhost:8000/simulation-list", json={"simulationType": "srw"}
)
# view sim list (NOT FOR RUNNING SIM)
print()
print(response_sim_list.url)
pprint.pprint(dict(response_sim_list.headers))
pprint.pprint(response_sim_list.json())

# get sim id from uploaded_sim
sim_id = uploaded_sim["models"]["simulation"]["simulationId"]
print("Simulation ID of uploaded sim: " + str(sim_id))
# copy the data from the uploaded_sim
data = uploaded_sim
# set the sim_id to prevent 'simulationId not found' error
data["simulationId"] = sim_id

# run simulation here
response_run_simulation = session.post(
    "http://localhost:8000/run-simulation", json=data
)

print(response_run_simulation.url)
print(response_run_simulation)
pprint.pprint(response_run_simulation.headers)

# #if __name__ == "__main__":
# #    main()
