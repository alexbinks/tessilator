# import the remaining modules required for the TESS analysis

from ..modules_to_import import *

import os, glob
from ..tess_functions2 import aper_run_cutouts

dirname = os.path.dirname(__file__)
ABDor_Table = ascii.read(dirname + "/ABDor_36_4_3_phot_out_fixed.tab")
files = sorted(glob.glob(dirname + "/ABDor*slice*.fits"))
list_ind = []

for file in files:
    list_ind.append(np.where(np.array(ABDor_Table["id"] == int(file[-9:-5]))))
ABDor_Table = ABDor_Table[list_ind]
print(ABDor_Table)
for i in range(len(ABDor_Table)):
    def test_cutout_ABDor_slices():    
        f = aper_run_cutouts(files[i], store_file="ABDor.txt")
        print(f)
        assert np.isclose(f[0][3], ABDor_Table["aperture_sum"][i][0][0], rtol=0.0, atol=1e-4)
        assert np.isclose(f[0][3], ABDor_Table["aperture_sum"][i][0][0], rtol=0.0, atol=1e-4)
    x = test_cutout_ABDor_slices()
