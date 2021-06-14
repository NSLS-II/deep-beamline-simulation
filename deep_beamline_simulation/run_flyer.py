import datetime
from bluesky.run_engine import RunEngine
from bluesky.callbacks import best_effort
import bluesky.plans as bp
import databroker
from databroker import Broker
from ophyd.utils import make_dir_tree
from sirepo_bluesky.srw_handler import SRWFileHandler
import sirepo_bluesky.sirepo_flyer as sf
import sys

# handle databroker, bluesky, and mongo db
def bluesky_utils():
    # setup run engine
    RE = RunEngine({})
    bec = best_effort.BestEffortCallback()
    bec.disable_plots()
    RE.subscribe(bec)
    # setup databroker and mongo
    db = Broker.named('local')
    try:
        databroker.assets.utils.install_sentinels(db.reg.config, version=1)
    except Exception:
        pass
    # setup file handler
    RE.subscribe(db.insert)
    db.reg.register_handler('srw', SRWFileHandler, overwrite=True)
    db.reg.register_handler('SIREPO_FLYER', SRWFileHandler, overwrite=True)
    return RE, db

# run flyer simulation
def flyer(sim_id, aperture, lens, optic, watch_point):
    params_to_change = []
    for i in range(1,6):
        aperture_name = aperture
        parameters_update1 = {'horizontalSize': i * .1, 'verticalSize': (16 - i) * .1}
        lens_name = lens
        parameters_update2 = {'horizontalFocalLength': i + 7}
        optics_name = optic
        parameters_update3 = {'horizontalSize': 6 - i}
        params_to_change.append({aperture_name: parameters_update1,
                                 lens_name: parameters_update2,
                                 optics_name: parameters_update3})
    # setup data collection location
    root_dir = '/tmp/sirepo_flyer_data'
    _ = make_dir_tree(datetime.datetime.now().year, base_path=root_dir)
    sirepo_flyer = sf.SirepoFlyer(sim_id=sim_id, server_name='http://10.10.10.10:8000',
                              root_dir=root_dir, params_to_change=params_to_change,
                              watch_name=watch_point)
    return sirepo_flyer


def main():
    #Enter a sim_id, aperature name, lens name, optic name,
    # and watch point separated by spaces
    RE, db = bluesky_utils()
    sys.argv.pop(0)
    sim_id = sys.argv[0]
    aperture = sys.argv[1]
    lens = sys.argv[2]
    optic = sys.argv[3]
    watch_point = sys.argv[4]
    sirepo_flyer = flyer(sim_id, aperture, lens, optic, watch_point)
    RE(bp.fly([sirepo_flyer]))
    hdr = db[-1]
    data = hdr.data

if __name__ == "__main__":
    main()
