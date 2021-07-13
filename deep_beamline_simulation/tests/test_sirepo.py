from sirepo_bluesky.sirepo_bluesky import SirepoBluesky
from deep_beamline_simulation.sirepo_data import sirepo_data
import deep_beamline_simulation.tests
import os
import pytest
import vcr

cassette_location = os.path.join(
    os.path.dirname(deep_beamline_simulation.tests.__file__), "vcr_cassettes"
)


def _test_auth(sim_id, server_name):
    sb = SirepoBluesky(server_name)
    data, schema = sb.auth("srw", sim_id)
    assert "beamline" in data["models"]


@vcr.use_cassette(f"{cassette_location}/sirepo_test.yml")
def test_auth_vcr():
    _test_auth(sim_id="avw4qnNw", server_name="http://10.10.10.10:8000")

