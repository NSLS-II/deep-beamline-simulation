import logging

import pytest

from deep_beamline_simulation import SirepoGuestSession


# urllib3 generates a lot of DEBUG logging output when
# all I want to see is the deep_beamline_simulation DEBUG output
# so explicitly turn off urllib3 logging
logging.getLogger("urllib3").setLevel(level=logging.WARNING)


@pytest.fixture
def sirepo_server_url():
    return "http://localhost:8000"


@pytest.fixture
def sirepo_guest_session(sirepo_server_url):
    """
    A factory-as-a-fixture.
    """

    def sirepo_guest_session_(simulation_type):
        return SirepoGuestSession(
            sirepo_server_url=sirepo_server_url, simulation_type=simulation_type
        )

    return sirepo_guest_session_
