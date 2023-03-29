# import the remaining modules required for the TESS analysis

import os, glob
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from ..lc_analysis import aper_run

Targ_Table = Table()
Targ_Table["source_id"] = ["abc"]

dirname = os.path.dirname(__file__)
ABDor_Table = ascii.read(dirname +"/ABDor_tests/ABDor_36_4_3_phot_out_fixed.tab")
files = sorted(glob.glob(dirname +"/ABDor_tests/ABDor*slice*.fits"))
list_ind = []

for file in files:
    list_ind.append(np.where(np.array(ABDor_Table["id"] == int(file[-9:-5]))))
ABDor_Table = ABDor_Table[list_ind]
for i in range(len(ABDor_Table)):
    def test_cutout_ABDor_slices():
        f = aper_run(files[i], Targ_Table)
        assert np.isclose(f[0][3], ABDor_Table["aperture_sum"][i][0][0], rtol=0.0, atol=1e-4)
        assert np.isclose(f[0][3], ABDor_Table["aperture_sum"][i][0][0], rtol=0.0, atol=1e-4)
    x = test_cutout_ABDor_slices()
