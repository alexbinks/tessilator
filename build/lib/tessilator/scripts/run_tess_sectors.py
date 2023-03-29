from tessilator import tessilator
import numpy as np
import os
import logging
from astropy.table import join
from astropy.io import ascii

def main(args=None):
    logging.basicConfig(filename="output.log", level=logging.ERROR)

    fluxCon, scc, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
    conFile, periodFile = tessilator.setup_filenames(fileRef, scc=scc)
    t_large_sec_check = tessilator.test_table_large_sectors(tFile)

    if t_large_sec_check is not None:
        tTargets = t_large_sec_check
    else:
        gaia_data = tessilator.read_data(tFile, name_is_source_id=1)
        xy_pixel_data = tessilator.get_tess_pixel_xy(gaia_data)
        tTargets = join(gaia_data, xy_pixel_data, keys='source_id')
        ascii.write(tTargets, tFile, format='csv', overwrite=True)

    tTargets = tTargets[tTargets['Sector'] == scc[0]]
    Rad, SkyRad = 1.0, np.array([6.0,8.0])

    tTargets = tessilator.collect_contamination_data(tTargets, fluxCon, 0,
                                                   conFile, Rad=Rad)

    tessilator.all_sector(tTargets, scc, makePlots, periodFile)
