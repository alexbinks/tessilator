from TESSilator import tess_cutouts, tess_functions2
import numpy as np
import os
from astropy.table import join
from astropy.io import ascii

fluxCon, secNum, makePlots, fileRef, tFilename = tess_cutouts.setupInputParameters()
conFile, periodFile, storeFile = tess_cutouts.setupFilenames(fileRef, sector_num=secNum)

t_large_sec_check = tess_cutouts.test_table_large_sectors(tFilename)

if t_large_sec_check is not None:
    tTargets = t_large_sec_check
else:
    gaia_data = tess_cutouts.readData(tFilename, storeFile=storeFile, name_is_source_id=1)
    xy_pixel_data = tess_functions2.getTESSPixelXY(gaia_data)
    tTargets = join(gaia_data, xy_pixel_data, keys='source_id')
ascii.write(tTargets, tFilename, format='csv', overwrite=True)

tTargets = tTargets[tTargets['Sector'] == secNum]
Rad, SkyRad = 1.0, np.array([6.0,8.0])

tTargets = tess_cutouts.collectContaminationData(tTargets, fluxCon, 0, conFile, Rad=Rad)

tess_cutouts.iterate_all_cc(tTargets, storeFile, secNum, makePlots, periodFile)