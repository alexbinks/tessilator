# import the remaining modules required for the TESS analysis

import os
import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table
from ..aperture import aper_run
from glob import glob
from ..file_io import logger_tessilator

makelog=False
if makelog:
    logger = logger_tessilator('aper_run_tests')

def test_aper_run():

    target_roots = glob('./targets_tests/*')
    for target_root in target_roots:
        aper_tables = glob(f'./targets_tests/{target_root}*_tests/ap*csv')
        for aper_table in aper_tables:
# read in the aperture results
            phot_tab = ascii.read(aper_table)
            phot_tab.rename_column('id', 'source_id')
            slice_images = sorted(glob(f"./{aper_table.split('/')[1]}/*slice*"))
            print(slice_images)
            for si in slice_images:
                num_slice = int(si[-9:-5])
                g = np.where(phot_tab["run_no"] == num_slice)[0]
                if len(g) == 1:
                    phot_comp = phot_tab[g]
                    aper_slice = aper_run(si, phot_comp)
                    print(aper_slice["run_no"][0], phot_comp["run_no"][0])
                    print(aper_slice["flux"][0], phot_comp["flux"][0])
                    assert np.isclose(aper_slice["flux"][0], phot_comp["flux"][0], rtol=1.)
