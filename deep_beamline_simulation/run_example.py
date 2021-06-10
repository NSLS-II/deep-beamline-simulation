# import statements
import datetime
import sirepo_bluesky.sirepo_detector as sd
from bluesky.run_engine import RunEngine
from bluesky.callbacks import best_effort
import bluesky.plans as bp
import databroker
from databroker import Broker
from ophyd.utils import make_dir_tree
from sirepo_bluesky.srw_handler import SRWFileHandler
import matplotlib.pyplot as plt

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

# plots
plt.ion()

# store data
root_dir = '/tmp/sirepo_flyer_data'
_= make_dir_tree(datetime.datetime.now().year, base_path=root_dir)

sim_id = 'RacK7rV1'
sirepo_det = sd.SirepoDetector(sim_id=sim_id)
sirepo_det.select_optic('Aperture')

param1 = sirepo_det.create_parameter('horizontalSize')
param2 = sirepo_det.create_parameter('verticalSize')
sirepo_det.read_attrs = ['image', 'mean', 'photon_energy']
sirepo_det.configuration_attrs = ['horizontal_extent', 'vertical_extent', 'shape']

RE(bp.grid_scan([sirepo_det],
                param1, 0, 1, 10,
                param2, 0, 1, 10,
                True))

#get the data
hdr = db[-1]
#hdr.table()
imgs = list(hdr.data('sirepo_det_image'))
print(imgs)
