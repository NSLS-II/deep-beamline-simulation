from deep_beamline_simulation import SirepoGuestSession
from deep_beamline_simulation.ophyd import build_sirepo_simulation


def get_srx_sirepo_simulation(sirepo_server_url="http://localhost:8000"):
    with SirepoGuestSession(sirepo_server_url=sirepo_server_url, simulation_type="srw") as sirepo_session:
        simulation_table = sirepo_session.simulation_list()
        # pick a known simulation
        simulation_id = simulation_table[
            "/Light Source Facilities/NSLS-II/NSLS-II SRX beamline"
        ]["NSLS-II SRX beamline"]
        srx_simulation_data = sirepo_session.simulation_data(
            simulation_id=simulation_id
        )
        sirepo_simulation_class = build_sirepo_simulation(sirepo_simulation_data=srx_simulation_data)

    return sirepo_simulation_class