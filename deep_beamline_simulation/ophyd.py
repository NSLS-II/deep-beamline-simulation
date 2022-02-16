import logging

import inflection
import ophyd


def build_sirepo_simulation(sirepo_simulation_data):
    """Build a SirepoSimulation(ophyd.Device) class with one ophyd.Device for each Sirepo optical element.

    Parameters
    ----------
    sirepo_simulation_data:

    In [6]: srx_beamline_simulation_data = sirepo_session.simulation_data(simulation_id="ep6O223w")

    In [7]: srx_beamline_simulation_data
    Out[7]:
    {
      'models': {
        'arbitraryMagField': {
          'interpolationOrder': 1,
          'longitudinalPosition': 0,
          'magneticFile': 'ivu21_srx_g6_2c.dat'
        },
        'beamline': [
          {
            'horizontalOffset': 0,
            'horizontalSize': 2,
            'id': 2,
            'position': 33.1798,
            'shape': 'r',
            'title': 'S0',
            'type': 'aperture',
            'verticalOffset': 0,
            'verticalSize': 1
          },
          {
            'autocomputeVectors': 'horizontal',
            'grazingAngle': 2.5,
            'heightAmplification': 1,
            'heightProfileFile': None,
            'horizontalOffset': 0,
            'id': 3,
            'normalVectorX': 0.9999968750016276,
            'normalVectorY': 0,
            'normalVectorZ': -0.002499997395834147,
            'orientation': 'x',
            'position': 34.2608,
            'radius': 8871.45,
            'sagittalSize': 0.005,
            'tangentialSize': 0.95,
            'tangentialVectorX': 0.002499997395834147,
            'tangentialVectorY': 0,
            'title': 'HFM',
            'type': 'sphericalMirror',
            'verticalOffset': 0
          },
          ...

    Returns
    -------

    """
    log = logging.getLogger("deep_beamline_simulation.ophyd")

    beamline_optical_element_components = {}

    # beamline_elements is a list in "beam order" of simulated optical elements
    beamline_elements = sirepo_simulation_data["models"]["beamline"]
    for i, beamline_element in enumerate(beamline_elements):
        # create an ophyd.Device class for each beamline element
        #   there will be some redundancy because a beamline may have several optical
        #   elements with the same "type", such as "aperture", and a separate class
        #   will be dynamically created for each one
        # the advantage is that the optical element parameters will be part of the class
        #   which helps when composing optical element Devices (see example)

        optical_element_name, optical_element_class = build_sirepo_optical_element_class(
            sirepo_optical_element_data=beamline_element
        )

        log.debug("create ophyd.Component with cls='%s' name='%s'", optical_element_class, optical_element_name)
        # need to handle problem names such as "DCM: C2"
        #  the transformation underscore(parameterize) converts "DCM: C2" to "dcm__c2"
        optical_element_python_identifier = inflection.underscore(inflection.parameterize(optical_element_name))
        beamline_optical_element_components[optical_element_python_identifier] = ophyd.Component(
            cls=optical_element_class,
            name=optical_element_name
        )

    sirepo_simulation_class = type(
        "SirepoSimulation",
        (ophyd.Device, ),
        beamline_optical_element_components
    )

    log.debug("sirepo simulation class: '%s'", sirepo_simulation_class)
    log.debug("component_names: '%s'", sirepo_simulation_class.component_names)

    return sirepo_simulation_class


def build_sirepo_optical_element_class(sirepo_optical_element_data):
    log = logging.getLogger("deep_beamline_simulation.ophyd")

    # build a dictionary of ophyd.Components creating Signals
    parameter_components = dict()
    log.debug(f"beamline element_data '%s'", sirepo_optical_element_data)
    for beamline_element_attr_name, beamline_element_attr_value in sirepo_optical_element_data.items():
        # for example, an aperture might have these parameters,
        #   each of which will become a Component(cls=ophyd.Signal, ...)
        # {
        #     "horizontalOffset": 0,
        #     "horizontalSize": 2,
        #     "id": 2,
        #     "position": 33.1798,     <---- must change "position" to something else because DEVICE_RESERVED_ATTRS
        #     "shape": "r",
        #     "title": "S0",
        #     "type": "aperture",
        #     "verticalOffset": 0,
        #     "verticalSize": 1
        # },
        if beamline_element_attr_name == "position":
            beamline_element_attr_name = "position_"

        log.debug(
            f"  '%s': '%s' (%s)",
            beamline_element_attr_name,
            beamline_element_attr_value,
            type(beamline_element_attr_value)
        )
        parameter_components[beamline_element_attr_name] = ophyd.Component(
            cls=ophyd.Signal,
            name=beamline_element_attr_name,
            value=beamline_element_attr_value
        )

    # beamline_element["type"] could be aperture, sphericalMirror, crystal, ellipsoidMirror, watch, ...
    beamline_element_type_name = sirepo_optical_element_data["type"]
    # beamline_device_instance_name must be a valid python identifier, but the optical element's "title"
    #   may contain spaces, so apply a transformation to make it a valid identifier
    beamline_element_instance_name = sirepo_optical_element_data["title"].replace(" ", "_")

    beamline_optical_element_class = type(
        beamline_element_type_name,
        (ophyd.Device, ),
        parameter_components
    )
    log.debug("optical element type: %s", dir(beamline_optical_element_class))

    return beamline_element_instance_name,  beamline_optical_element_class


# want to build classes like this:
# class Aperture(ophyd.Device):
#     position = ophyd.Component(cls=ophyd.Signal, name="position", value=33.1798)
#     shape = ophyd.Component(cls=ophyd.Signal, name="shape", value="r")
#
#
# class SirepoSimulation():
#     aperture = ophyd.Component(cls=Aperture, name="S0")
