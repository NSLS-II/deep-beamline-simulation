import requests
import random
import hashlib
import time
import ast
import numconv
import base64
import matplotlib.pyplot as plt


class Client:
    def __init__(self, sim_type="srw"):
        # create a session
        self.session = requests.Session()
        self.host = "http://10.10.10.10:8000"
        self.sim_name = None
        # sim type is typically 'srw'
        self.sim_type = sim_type
        self.sim_id = None
        self.sim_data = None
        self.cookies = None

    def put_sim_name(self, sim_name):
        """Set simulation name"""
        self.sim_name = sim_name

    def get_sim_name(self):
        """Get simulation name"""
        return self.sim_name

    def put_sim_data(self, data):
        """Set simulation data"""
        self.sim_data = data

    def get_sim_data(self):
        """Get simulation name"""
        return self.sim_data

    def put_sim_id(self, sim_id):
        """Set simulation id"""
        self.sim_id = sim_id

    def get_sim_id(self):
        """Get simulation name"""
        return self.sim_id

    def get_simulation_id(self):
        """Returns simulation id based on simulation name"""
        to_search = {"simulation.name": self.sim_name}
        response = self.post("/simulation-list", {"search": to_search})
        if len(response) > 1:
            raise ValueError(
                f"expecting only one simulation with name={self.sim_name} found {response}"
            )
        elif len(response) == 0:
            raise ValueError("No simulation id found")
        else:
            self.sim_id = response[0]["simulationId"]
        return response[0]["simulationId"]

    def get_simulation_description(self):
        """Returns simulation description based on name"""
        to_search = {"simulation.name": self.sim_name}
        post_dict = dict(simulationType="srw", search=to_search)
        description = self.post("/simulation-list", post_dict)
        return description

    def get_titles(self, data):
        """Returns list of beamline parts"""
        beam = data["models"]["beamline"]
        line = []
        for d in beam:
            for key in d:
                if key == "title":
                    line.append(d[key])
        return line

    def update_param(self, data, component, param, new_value):
        """Returns changed data given a component and value"""
        updated_data = data
        beamline = updated_data["models"]["beamline"]
        for c in beamline:
            if c["title"] == component:
                c[param] = new_value
        return updated_data

    def auth(self):
        """Connect to the server and returns the data for the simulation identified by sim_id"""
        req = dict(simulationType="srw", simulationId=self.sim_id)
        rand_num = random.SystemRandom()
        req["authNonce"] = (
            str(int(time.time()))
            + "-"
            + "".join(rand_num.choice(numconv.BASE62) for x in range(32))
        )
        hash_val = hashlib.sha256()
        hash_val.update(
            ":".join(
                [
                    req["authNonce"],
                    req["simulationType"],
                    req["simulationId"],
                    "bluesky",
                ]
            ).encode()
        )
        req["authHash"] = "v1:" + base64.urlsafe_b64encode(hash_val.digest()).decode()
        res = self._post_json("bluesky-auth", req)
        if not ("state" in res and res["state"] == "ok"):
            raise Exception("bluesky_auth failed: {}".format(res))
        schema = res["schema"]
        data = res["data"]
        return data, schema

    def login(self):
        """Instead of auth, login as a guest user each time"""
        r = self.post("simulation-list", {})
        assert r["srException"]["routeName"] == "missingCookies"
        self.post("/auth-guest-login/" + self.sim_type, {})

    @staticmethod
    def _assert_success(response, url):
        """Check that request is working"""
        assert (
            response.status_code == requests.codes.ok
        ), "{} request failed, status: {}".format(url, response.status_code)

    def _post_json(self, url, payload):
        """Returns the request response as a json"""
        response = requests.post(
            "{}/{}".format("http://localhost:8000", url),
            json=payload,
            cookies=self.cookies,
        )
        self._assert_success(response, url)
        if not self.cookies:
            self.cookies = response.cookies
        return response.json()

    def post(self, path, data, file=None):
        """Helper of _post_json"""
        data["simulationType"] = self.sim_type
        response = self.session.post(
            f"{self.host}/{path}",
            files=file,
            json=data,
        )
        assert response.ok, f"response={response} not ok"
        return response.json()

    def import_simulation(self, path):
        """imports zip file of simulation infomation"""
        return self.post(
            f"/import-file/{self.sim_type}",
            {},
            file={"file": open("./example-sim.zip", "rb"), "folder": (None, "/foo")},
        )

    def new_simulation(self, sim_name):
        """Creates new simulation"""
        data = {
            "folder": "/",
            "name": sim_name,
            "simulationType": self.sim_type,
            "sourceType": "u",
        }
        return self.post("/new-simulation", data)

    def run_simulation(self, data, max_calls=1000):
        """Returns simulation data after running it"""
        data["simulationId"] = self.sim_id
        response = self._post_json("run-simulation", data)
        for i in range(max_calls):
            state = response["state"]
            if state == "completed" or state == "error":
                break
            assert (
                "nextRequestSeconds" in response
            ), 'missing "nextRequestSeconds" in response:{}'.format(response)
            time.sleep(response["nextRequestSeconds"])
            response = self._post_json("run-status", response["nextRequest"])
        assert state == "completed", "simulation failed to complete: {}".format(state)
        return response

    def set_report_type(self, watchpoint):
        """Decide what watchpoint is returned when sim is run"""
        self.sim_data["report"] = watchpoint


client = Client()
client.login()
client.put_sim_name("NSLS-II ESM beamline")
print(f"simulation_id of existing simulation {client.get_simulation_id()}")

# Create a new empty simulation. This creates a new "empty" simulation that
# you can then modify using other API's (ex saveSimulationData and uploadFile).
# You can see these API's in use in a variety of tests. The clearest might be
# tests/lib_files_test.py
print(
    f'simulation id of new simulation {client.new_simulation("example")["models"]["simulation"]["simulationId"]}'
)

# Create new simulation by importing an archive of a simulation.  This
# allows you to hand craft a sirepo-data.json (how Sirepo internally represents
# simulations) with exactly the parameters you want. You can also upload the necessary
# lib files with the sim. After upload the sim is ready to be run or modified.
print(
    f'simulation id of imported simulation {client.import_simulation("example-sim.zip")["models"]["simulation"]["simulationId"]}'
)
