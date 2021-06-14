# import statements
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np
from bluesky.run_engine import RunEngine
from bluesky.callbacks import best_effort
import bluesky.plans as bp
import sirepo_bluesky.sirepo_detector as sd
from sirepo_bluesky.srw_handler import SRWFileHandler
import databroker
from databroker import Broker
from ophyd.utils import make_dir_tree


def utils():
    # bluesky RunEngine
    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    RE.subscribe(bec)

    # Mongo Backend
    db = Broker.named('local')
    try:
        databroker.assets.utils.install_sentinels(db.reg.config, version=1)
    except Exception:
        pass
    # update database info
    RE.subscribe(db.insert)
    db.reg.register_handler('srw', SRWFileHandler, overwrite=True)

    # store data
    root_dir = '/tmp/sirepo_flyer_data'
    _= make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

    return RE, db


def simple_run(RE, db, sim_id, optic):
    sim_id = sim_id
    sirepo_det = sd.SirepoDetector(sim_id=sim_id)
    sirepo_det.select_optic(optic)

    param1 = sirepo_det.create_parameter('horizontalSize')
    param2 = sirepo_det.create_parameter('verticalSize')
    sirepo_det.read_attrs = ['image', 'mean', 'photon_energy']
    sirepo_det.configuration_attrs = ['horizontal_extent', 'vertical_extent', 'shape']

    RE(bp.grid_scan([sirepo_det],
                    param1, 0, 1, 10,
                    param2, 0, 1, 10,
                    True))
    plt.show()
    hdr = db[-1]
    data = np.array(list(hdr.data('sirepo_det_image')))
    plt.imshow(data[data.shape[0]-3])
    plt.show()


def main():
    RE, db = utils()
    # enter sim_id and optic
    sys.argv.pop(0)
    sim_id = sys.argv[0]
    optic  = sys.argv[1]
    simple_run(RE, db, sim_id, optic)


if __name__ == "__main__":
    main()
