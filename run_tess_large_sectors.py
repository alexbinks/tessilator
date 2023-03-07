from TESSilator import tessilator
import numpy as np
import os
import logging
from astropy.table import join
from astropy.io import ascii

logging.basicConfig(filename="output.log", level=logging.ERROR)


fluxCon, scc, makePlots, fileRef, tFilename = tess_cutouts.setupInputParameters()
conFile, periodFile = tess_cutouts.setupFilenames(fileRef, scc=scc)
t_large_sec_check = tess_cutouts.test_table_large_sectors(tFilename)

if t_large_sec_check is not None:
    tTargets = t_large_sec_check
else:
    gaia_data = tess_cutouts.readData(tFilename, name_is_source_id=1)
    xy_pixel_data = tess_cutouts.getTESSPixelXY(gaia_data)
    tTargets = join(gaia_data, xy_pixel_data, keys='source_id')
    ascii.write(tTargets, tFilename, format='csv', overwrite=True)

tTargets = tTargets[tTargets['Sector'] == scc[0]]
Rad, SkyRad = 1.0, np.array([6.0,8.0])

tTargets = tess_cutouts.collectContaminationData(tTargets, fluxCon, 0, conFile, Rad=Rad)

tess_cutouts.iterate_all_cc(tTargets, scc, makePlots, periodFile)
