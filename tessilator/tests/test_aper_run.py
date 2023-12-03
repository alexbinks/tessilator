# import the remaining modules required for the TESS analysis

import os
import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from ..lc_analysis import aper_run
from glob import glob



def test_aper_run():

    aper_tables = glob(f'./ABD*_tests/ap*csv') 
    for a in aper_tables:
# read in the aperture results
        phot_tab = ascii.read(a)
        phot_tab.rename_column('id', 'source_id')
        slice_images = sorted(glob(f"./{a.split('/')[1]}/*slice*"))

        for si in slice_images:
            print(si)
            print(phot_tab.columns)
            num_slice = int(si[-9:-5])
            g = np.where(phot_tab["run_no"] == num_slice)[0]
            if len(g) == 1:
                phot_comp = phot_tab[g]
                print(si)
                aper_slice = aper_run(si, phot_comp)
                print(aper_slice["run_no"][0], phot_comp["run_no"][0])
                assert np.isclose(aper_slice["flux"][0], phot_comp["flux"][0], rtol=0.01)

#            testo(si, phot_comp)

