from tessilator import tessilator

import numpy as np
import logging

def main(args=None):
    fluxCon, lcCon, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
    conFile, periodFile = tessilator.setup_filenames(fileRef)

    Rad, SkyRad = 1.0, np.array([6.0,8.0])

    logging.basicConfig(filename="output.log", level=logging.WARNING)

    print("Reading the table and formatting into astropy table structure.")
    tTargets = tessilator.read_data(tFile)
    print("Done reading the table and formatting.")

    print("...now calculating the contamination.")
    tTargets = tessilator.collect_contamination_data(tTargets, fluxCon, lcCon,
                                                     conFile, Rad=Rad)
    print("Done calculating the contamination.")

    print("...now iterating over each source.")
    tessilator.all_sources_cutout(tTargets, periodFile, lcCon, fluxCon, conFile,
                                  makePlots, choose_sec=None)
