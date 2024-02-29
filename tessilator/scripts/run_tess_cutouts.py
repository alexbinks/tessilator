from tessilator import tessilator

import numpy as np
import logging

def main(args=None):

    fluxCon, lcCon, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
    periodFile = tessilator.setup_filenames(fileRef)

    Rad, SkyRad = 1.0, np.array([6.0,8.0])
    gaia_sys = True

    logging.basicConfig(filename="output.log", level=logging.INFO)

    print("Reading the table and formatting into astropy table structure.")
    print(tFile)
    tTargets = tessilator.read_data(tFile, gaia_sys=gaia_sys, type_coord='icrs')
    print("Done reading the table and formatting.")

    print("...now iterating over each source.")
    tessilator.all_sources_cutout(tTargets, periodFile, lcCon, fluxCon,
                                  makePlots, fileRef, Rad=Rad, choose_sec=None,
                                  save_phot=True, cbv_flag=False, store_lc=True,
                                  tot_attempts=10, cap_files=10, fix_noise=False)
