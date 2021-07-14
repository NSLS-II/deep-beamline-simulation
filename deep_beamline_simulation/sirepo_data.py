import time
import numpy as np
import matplotlib.pyplot as plt
from sirepo_bluesky.sirepo_bluesky import SirepoBluesky


class sirepo_data:
    def __init__(self, simulation_id):
        # define sirepo bluesky object
        self.sb = SirepoBluesky("http://localhost:8000")
        # define sim _id
        self.sim_id = simulation_id
        # authenticate and define simulation
        self.data, self.schema = self.sb.auth("srw", self.sim_id)

    def get_simids(self):
        # stores all simulation id's and names
        id_dict = {}
        # get all simulation information
        sim_dict = self.sb.simulation_list()
        for d in sim_dict:
            for key in d:
                if key == "simulationId":
                    id_dict[d["name"]] = d[key]
        return id_dict

    def get_data(self):
        # returns data file
        return self.sb.get_datafile()

    def get_components(self):
        # get beamline dictionary
        beamline = self.data["models"]["beamline"]
        # create dictonary for organized data
        components = {}
        # get the title of each
        for d in beamline:
            parameter_list = {}
            for key in d:
                if key == "title":
                    components[d["title"]] = parameter_list
                else:
                    parameter_list[key] = d[key]
        return components

    # for specific beamline example
    def generate_data(self):
        # get beamline component, example: aperture
        aperture = self.sb.find_element(
            self.data["models"]["beamline"], "title", "Aperture"
        )
        # time this process
        t_start = time.time()
        # determine vertical and horizontal size range
        vertical = np.linspace(0.01, 1, 2)
        horizontal = np.linspace(0.01, 1, 2)
        # loop through each value
        for v in vertical:
            for h in horizontal:
                # define sizes
                aperture["horizontalSize"] = h
                aperture["verticalSize"] = v
                # pick watchpoint
                watch = self.sb.find_element(
                    self.data["models"]["beamline"], "title", "Watchpoint"
                )
                # define watchpoint report
                self.data["report"] = "watchpointReport{}".format(watch["id"])
                # run simulation
                # data is parsed into cols of number of horizontal points
                hor_ver_int = self.sb.run_simulation()
                print(
                    hor_ver_int["x_label"] + " has range " + str(hor_ver_int["x_range"])
                )
                print(
                    hor_ver_int["y_label"] + " has range " + str(hor_ver_int["y_range"])
                )
                print(
                    hor_ver_int["z_label"] + " has range " + str(hor_ver_int["z_range"])
                )
                # plot if needed
                # plt.imshow(hor_ver_int["z_matrix"])
                # plt.show()
                # plt.pause(0.01)
                # plt.close()
        t_end = time.time()
        # display time taken to collect data
        print((t_start - t_end) / 60.0)
        # possibly move to a file?
        intensity_matrix = hor_ver_int["z_matrix"]
        return intensity_matrix


def main():
    # use sim_id from the example beamline
    test_sim = sirepo_data("avw4qnNw")
    print(test_sim.get_simids())
    print(test_sim.get_components())
    test_sim.generate_data()


if __name__ == "__main__":
    main()
