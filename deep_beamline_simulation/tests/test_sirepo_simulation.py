import pytest
from bluesky import RunEngine
from bluesky.plans import count
import ophyd

from deep_beamline_simulation.ophyd import build_sirepo_simulation, build_sirepo_optical_element_class


def test_build_sirepo_optical_element_component():
    optical_element_data = {
        "autocomputeVectors": "horizontal",
        "grazingAngle": 2.5,
        "heightAmplification": 1,
        "heightProfileFile": "None",  # <--- a trap, this "None" gets converted to None maybe?
        "horizontalOffset": 0,
        "id": 3,
        "normalVectorX": 0.9999968750016276,
        "normalVectorY": 0,
        "normalVectorZ": -0.002499997395834147,
        "orientation": "x",
        "position": 34.2608, # <--- must change "position" to something else because ophyd.DEVICE_RESERVED_ATTRS
        "radius": 8871.45,
        "sagittalSize": 0.005,
        "tangentialSize": 0.95,
        "tangentialVectorX": 0.002499997395834147,
        "tangentialVectorY": 0,
        "title": "HFM",
        "type": "sphericalMirror",
        "verticalOffset": 0
    }
    optical_element_name, optical_element_class = build_sirepo_optical_element_class(sirepo_optical_element_data=optical_element_data)

    assert optical_element_name == "HFM"
    assert optical_element_class.__name__ == "sphericalMirror"

    # is there a ophyd.Component for each optical element parameter?
    for parameter_name in optical_element_data.keys():
        if parameter_name == "position":
            parameter_name = "position_"
        assert hasattr(optical_element_class, parameter_name)
        assert type(getattr(optical_element_class, parameter_name)) == ophyd.Component

    # instantiate the optical element class
    optical_element_instance = optical_element_class(name=optical_element_name)
    # is there an ophyd.Signal for each optical element parameter?
    # does ophyd.Signal.get() return the expected type?
    # does ophyd.Signal.get() return the expected value?
    for parameter_name, parameter_value in optical_element_data.items():
        if parameter_name == "position":
            parameter_name = "position_"
        assert hasattr(optical_element_instance, parameter_name)
        assert type(getattr(optical_element_instance, parameter_name)) == ophyd.Signal
        assert isinstance(
            getattr(optical_element_instance, parameter_name).get(),
            type(parameter_value)
        )
        assert getattr(optical_element_instance, parameter_name).get() == parameter_value

    optical_element_instance.read()


def test_build_sirepo_simulation(sirepo_guest_session):
    # https://jsonformatter.curiousconcept.com/# handy for large data
    with sirepo_guest_session(simulation_type="srw") as sirepo_session:
        simulation_table = sirepo_session.simulation_list()
        # pick a known simulation
        simulation_id = simulation_table[
            "/Light Source Facilities/NSLS-II/NSLS-II SRX beamline"
        ]["NSLS-II SRX beamline"]
        srx_simulation_data = sirepo_session.simulation_data(
            simulation_id=simulation_id
        )
        sirepo_simulation_class = build_sirepo_simulation(sirepo_simulation_data=srx_simulation_data)

    sirepo_simulation_instance = sirepo_simulation_class(name="srx")
    sirepo_simulation_instance.read()


@pytest.mark.skip
def test_count_sirepo_simulation(sirepo_guest_session):
    with sirepo_guest_session(simulation_type="srw") as sirepo_session:
        simulation_table = sirepo_session.simulation_list()
        # pick a known simulation
        simulation_id = simulation_table[
            "/Light Source Facilities/NSLS-II/NSLS-II SRX beamline"
        ]["NSLS-II SRX beamline"]
        srx_simulation_data = sirepo_session.simulation_data(
            simulation_id=simulation_id
        )
        sirepo_simulation_class = build_sirepo_simulation(sirepo_simulation_data=srx_simulation_data)

    # this function will store all documents
    # published by the RunEngine in a list
    published_bluesky_documents = list()

    def store_published_document(name, document):
        published_bluesky_documents.append((name, document))

    RE = RunEngine()
    RE.subscribe(store_published_document)

    sirepo_simulation = sirepo_simulation_class(name="srx")
    run_id = RE(count([sirepo_simulation]))

    assert len(published_bluesky_documents) > 0