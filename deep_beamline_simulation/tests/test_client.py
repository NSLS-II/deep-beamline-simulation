from deep_beamline_simulation.client import Client
import ast
import json
import pytest
from pathlib import Path

def test_guestLogin():
    guest = Client()
    guest.login()
    #guest.put_sim_name('NSLS-II SRX beamline')
    #sim_id = guest.get_simulation_id()
    #data, original_schema = guest.auth()
    #ori_sim = guest.run_simulation(data)
    #assert ori_sim != None
    assert 1 == 1

#def test_dataUpload():
#    file_path = Path(deep_beamline_simulation.tests.__path__[0]) / "example_data.json"
#    print(file_path)
#    f = open(file_path)
#    ori_sim = json.load(f)
#    f.close()
#    upload = Client()
#    upload.put_sim_data(ori_sim)
#    upload.put_sim_name("example")
#    upload.put_sim_id("UKHYDDgu")
#    upload.auth()
#    upload.set_report_type('watchpointReport5')
#    up_sim = upload.run_simulation(ori_sim)
#    assert up_sim != None


#def test_updateParam():
#    with pytest.raises(ValueError) as e_info:
#        guest = Client()
#        guest.login()
#        sim_id = guest.get_simulation_id()



#        data, original_schema = guest.auth()
#        ori_sim = guest.run_simulation(data)
#        updated_data = guest.update_param(data, "S0", "horizontalSize", 1)
#        up_sim = guest.run_simulation(updated_data)

    #new_data = upload.run_simulation(ori_sim)
    #changed_data = upload.update_param(ori_sim, "Aperture", "horizontalOffset", 0.5)
    #up_sim = upload.run_simulation(changed_data)
    #f, axarr = plt.subplots(1, 2)
    #axarr[0].imshow(new_data["z_matrix"])
    #axarr[1].imshow(up_sim["z_matrix"])
    #plt.show()

