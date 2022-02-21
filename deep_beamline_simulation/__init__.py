import logging
import time as ttime

from contextlib import ContextDecorator
from urllib.parse import urlparse

import requests

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


class SirepoGuestSession(ContextDecorator):
    def __init__(self, sirepo_server_url, simulation_type):
        """
        Parameters
        ----------
        sirepo_server_url: str
          URI specifying host and port, eg. "http://localhost:8000"
        simulation_type: str
          "srw" or "shadow"
        """
        log = logging.getLogger(self.__class__.__name__)

        parsed_url = urlparse(sirepo_server_url)
        self._server_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        log.debug(f"self._server_uri: '%s'", self._server_url)
        self.simulation_type = simulation_type.lower()

        self._session = None
        self._response_auth_guest_login = None

    def login(self):
        """Take the necessary steps to log in to sirepo as a guest.

        Client code should prefer the context manager protocol to calling this method directly, for example:

            with SirepoGuestSession(sirepo_server_url="http://localhost", simulation_type="srw") as sirepo_session:
                ...

        """
        log = logging.getLogger(self.__class__.__name__)

        self._session = requests.Session()

        # get cookies by calling simulation-list
        response_simulation_list = self._post_to_sirepo(
            f"{self._server_url}/simulation-list",
            json={"simulationType": self.simulation_type},
        )
        log.debug("response_simulation_list: %s", response_simulation_list)

        log.debug("logging in as guest to '%s'", self._server_url)
        # store the response for troubleshooting and automatic tests
        self._response_auth_guest_login = self._post_to_sirepo(
            # "http://localhost:8000/auth-guest-login/srw"
            f"{self._server_url}/auth-guest-login/{self.simulation_type}"
        )
        log.debug("response_auth_guest_login: '%s'", self._response_auth_guest_login)

    def logout(self):
        """Close the HTTP session.

        This is the counterpart to `login`. Client code should prefer the context manager protocol.

        """
        self._session.close()

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *exc):
        self.logout()
        return False

    def _post_to_sirepo(self, sirepo_request_url, **kwargs):
        log = logging.getLogger(self.__class__.__name__)

        log.debug("url: '%s', kwargs: '%s'", sirepo_request_url, dict(kwargs))
        sirepo_response = self._session.post(sirepo_request_url, **kwargs)
        log.debug(
            "response: '%s', elapsed time: '%s's, ",
            sirepo_response,
            sirepo_response.elapsed,
        )
        return sirepo_response

    def simulation_list(self):
        """Return results from Sirepo's `simulation-list` endpoint.

        Despite the name this method returns a dictionary, which
        is how the endpoint works so this seems reasonable.

        Returns
        -------
        dictionary of simulation "folders", "names", and ids:
          a dictionary keyed by Sirepo simulation folder path with values that are
            a dictionary keyed by Sirepo simulation name with values that are
              simulation id (str)

        for example:
        {
            ...
            '/Light Source Facilities/NSLS-II/NSLS-II 3PW QAS beamline': {
                'NSLS-II 3PW QAS beamline': 'pOJ3njmQ'
            },
            '/Light Source Facilities/NSLS-II/NSLS-II CHX beamline': {
                'NSLS-II CHX beamline': '1HMAcuUI',
                'NSLS-II CHX beamline (tabulated)': '1eekGYQV'
            },
            ...
        }
        """
        response_sim_list = self._post_to_sirepo(
            f"{self._server_url}/simulation-list",
            json={"simulationType": self.simulation_type},
        )
        sim_list_results = response_sim_list.json()

        # build a dictionary from the simulation list results
        # sim_folder_name_to_id means "folder" -> "name" -> simulation id
        sim_folder_name_to_id = {}
        for sim_details in sorted(
            sim_list_results, key=lambda sim_details_: sim_details_["folder"]
        ):
            simulation_folder = sim_details["folder"]
            if simulation_folder not in sim_folder_name_to_id:
                sim_folder_name_to_id[simulation_folder] = {}
            sim_folder_name_to_id[simulation_folder][sim_details["name"]] = sim_details[
                "simulationId"
            ]

        return sim_folder_name_to_id

    def simulation_data(self, simulation_id):
        """
        Request simulation data for the specified simulation id.
        """
        response_simulation_data = self._session.get(
            f"{self._server_url}/simulation/{self.simulation_type}/{simulation_id}/0"
        )
        return response_simulation_data.json()

    def run_simulation(self, simulation_id, simulation_data, simulation_report=None):
        """
        Start a simulation but do not wait for it to complete.
        """
        log = logging.getLogger(self.__class__.__name__)

        simulation_data_copy = simulation_data.copy()
        simulation_data_copy["simulationId"] = simulation_id
        if simulation_report:
            simulation_data_copy["report"] = simulation_report
        run_simulation_response = self._session.post(
            f"{self._server_url}/run-simulation", json=simulation_data_copy
        )
        log.debug(
            "run-simulation response: state '%s', nextRequest: '%s', nextRequestSeconds '%s'",
            run_simulation_response.json()["state"],
            run_simulation_response.json()["nextRequest"],
            run_simulation_response.json()["nextRequestSeconds"],
        )
        return run_simulation_response

    def wait_for_simulation(self, run_simulation_response, max_status_calls=100):
        """
        Wait for a running simulation to complete and return the final run-status response.
        """
        log = logging.getLogger(self.__class__.__name__)

        simulation_id = run_simulation_response.json()["nextRequest"]["simulationId"]

        run_status_response = run_simulation_response
        run_status = run_status_response.json()
        for status_call_i in range(max_status_calls):
            run_state = run_status["state"]
            if run_state == "completed":
                log.info("simulation '%s' completed", simulation_id)
                break
            elif run_state == "error":
                log.error("simulation failed with an error")
                raise Exception()
            else:
                log.debug("making run-status call %d", status_call_i)
                next_request_seconds = run_status["nextRequestSeconds"]
                log.debug("sleeping for '%s' second(s)", next_request_seconds)
                ttime.sleep(next_request_seconds)
                run_status_response = self._session.post(
                    f"{self._server_url}/run-status", json=run_status["nextRequest"]
                )
                run_status = run_status_response.json()
                log.debug("run_status.json: %s", run_status)

        # the simulation completed successfully
        log.debug(
            "after successful completion run_status_response: %s\n  json:\n %s",
            run_status_response,
            run_status_response.json(),
        )
        return run_status_response
