from tessilator import tessilator

import numpy as np
import logging

fluxCon, lcCon, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
conFile, periodFile = tessilator.setup_filenames(fileRef)

Rad, SkyRad = 1.0, np.array([6.0,8.0])
gaia_sys = True

logging.basicConfig(filename="output.log", level=logging.INFO)

print("Reading the table and formatting into astropy table structure.")
print(tFile)
tTargets = tessilator.read_data(tFile, gaia_sys=gaia_sys, type_coord='icrs')
print("Done reading the table and formatting.")

print("...now calculating the contamination.")
tTargets = tessilator.collect_contamination_data(tTargets, fluxCon, conFile, Rad=Rad)
print("Done calculating the contamination.")

print("...now iterating over each source.")
tessilator.all_sources_cutout(tTargets, periodFile, lcCon, fluxCon, conFile,
                              makePlots, choose_sec=None, store_lc=True, tot_attempts=10, cap_files=7, fix_noise=False)
