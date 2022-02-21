import base64
import hashlib
import numconv
import pprint
import random
import requests
import time as ttime

from pathlib import Path

import deep_beamline_simulation

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

print("bluesky login")
req = dict(simulationType="srw", simulationId="")
r = random.SystemRandom()
req["authNonce"] = (
    str(int(ttime.time())) + "-" + "".join(r.choice(numconv.BASE62) for x in range(32))
)
h = hashlib.sha256()
h.update(
    ":".join(
        [req["authNonce"], req["simulationType"], req["simulationId"], "bluesky"]
    ).encode()
)

req["authHash"] = "v1:" + base64.urlsafe_b64encode(h.digest()).decode()

# self.cookies = None
# res = self._post_json('bluesky-auth', req)
session.post("http://localhost:8000/bluesky-auth", data=req)
