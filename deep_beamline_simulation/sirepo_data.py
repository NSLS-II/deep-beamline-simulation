from sirepo_bluesky.sirepo_bluesky import SirepoBluesky

# define sirepo bluesky object
sb = SirepoBluesky('http://localhost:8000')

# define sim _id
sim_id = 'avw4qnNw'
# authenticate and define simulation
simulation, schema = sb.auth('srw', sim_id)
# get aperture
aperture = sb.find_element(simulation['models']['beamline'], 'title', 'Aperture')
# define sizes
aperture['horizontalSize'] = 0.1
aperture['verticalSize'] = 0.1
# pick watchpoint
watch = sb.find_element(simulation['models']['beamline'], 'title', 'Watchpoint')
# define watchpoint report
simulation['report'] = 'watchpointReport{}'.format(watch['id'])
# run simulation
# data is parsed into cols of number of horizontal points
hor_ver_int = sb.run_simulation()
print(hor_ver_int['x_label'] + ' has range ' + str(hor_ver_int['x_range']))
print(hor_ver_int['y_label'] + ' has range ' + str(hor_ver_int['y_range']))
print(hor_ver_int['z_label'] + ' has range ' + str(hor_ver_int['z_range']))

# get data
# all data meshed together
data_file = sb.get_datafile()
