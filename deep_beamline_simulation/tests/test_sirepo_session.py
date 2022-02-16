from deep_beamline_simulation import SirepoGuestSession


def test_connect(sirepo_server_url):
    """
    This test is successful if no exception is raised.
    """
    with SirepoGuestSession(sirepo_server_url=sirepo_server_url) as sirepo_session:
        pass


def test_simulation_list(sirepo_server_url):
    with SirepoGuestSession(sirepo_server_url=sirepo_server_url) as sirepo_session:
        simulation_table = sirepo_session.simulation_list()

        # look for some known simulations
        assert "/Wavefront Propagation" in simulation_table
        assert (
            "Diffraction by an Aperture" in simulation_table["/Wavefront Propagation"]
        )
        assert (
            "/Light Source Facilities/NSLS-II/NSLS-II TES beamline" in simulation_table
        )
        assert (
            "NSLS-II TES beamline"
            in simulation_table["/Light Source Facilities/NSLS-II/NSLS-II TES beamline"]
        )


def test_simulation_data(sirepo_server_url):
    with SirepoGuestSession(sirepo_server_url=sirepo_server_url) as sirepo_session:
        simulation_table = sirepo_session.simulation_list()
        # pick a known simulation
        simulation_id = simulation_table[
            "/Light Source Facilities/NSLS-II/NSLS-II TES beamline"
        ]["NSLS-II TES beamline"]
        tes_simulation_data = sirepo_session.simulation_data(
            simulation_id=simulation_id
        )


def test_run_simulation(sirepo_server_url):
    with SirepoGuestSession(sirepo_server_url=sirepo_server_url) as sirepo_session:
        simulation_table = sirepo_session.simulation_list()
        # pick a known simulation
        simulation_id = simulation_table["/Wavefront Propagation"][
            "Diffraction by an Aperture"
        ]
        aperture_simulation_data = sirepo_session.simulation_data(
            simulation_id=simulation_id
        )

        run_simulation_response = sirepo_session.run_simulation(
            simulation_id=simulation_id,
            simulation_report="intensityReport",
            simulation_data=aperture_simulation_data
        )
        sirepo_session.wait_for_simulation(run_simulation_response)

        run_simulation_response = sirepo_session.run_simulation(
            simulation_id=simulation_id,
            simulation_report="watchpointReport6",
            simulation_data=aperture_simulation_data
        )
        sirepo_session.wait_for_simulation(run_simulation_response)
