import pprint

import requests


session = requests.Session()

response_sim_list = session.post(
    "http://localhost:8000/simulation-list",
    json={"simulationType": "srw"}
)
print(response_sim_list.url)
pprint.pprint(dict(response_sim_list.headers))

response_auth_guest_login = session.post(
    "http://localhost:8000/auth-guest-login/srw"
)

print(response_auth_guest_login.url)
pprint.pprint(response_auth_guest_login.json())

files = {
    'file': open('../aperture-only/sirepo-data.json', 'rb'),
    'folder': (None, "/foo")
}
response_import_file = session.post(
    "http://localhost:8000/import-file/srw",
    files=files,

)

print(response_import_file.url)
print(response_import_file)
pprint.pprint(response_import_file.json())

response_sim_list = session.post(
    "http://localhost:8000/simulation-list",
    json={"simulationType": "srw"}
)
print(response_sim_list.url)
pprint.pprint(dict(response_sim_list.headers))
pprint.pprint(response_sim_list.json()[-1])

response_run_simulation = session.get(
    "http://localhost:8000/run-simulation",
    json=response_import_file.json()
)

print(response_run_simulation.url)
print(response_run_simulation)
pprint.pprint(response_run_simulation.headers)
pprint.pprint(response_run_simulation.content)



#if __name__ == "__main__":
#    main()