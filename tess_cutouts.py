'''
Alexander Binks and Moritz Guenther, 2023

The TESSilator

This is a python3 program designed to provide an all-in-one module to measure lightcurves and stellar rotation periods from the Transiting Exoplanet Survey Satellite (TESS). Whilst there are many useful (and powerful) software tools available for working with TESS data, they are mostly provided as various steps in the data reduction process --- to our knowledge there are no programs that automate the full process from downloading the data (start) to obtaining rotation period measurements (finish). The software provided here fills this gap. The user provides a table of targets with basic Gaia DR3 information (source ID, sky positions, parallax and Gaia G magnitude) and simply allows the TESSilator to do the rest! The steps are:

(1) download photometric time-series data from TESS.

(2) scan the Gaia DR3 catalogue to quantify the level of background contamination from nearby sources.

(3) clean the lightcurves for poor quality data caused by systematic and instrumental effects.

(4) normalize and detrend lightcurves over the whole sector of observations.

(5) measure stellar rotation periods using two processes: the Lomb-Scargle periodogram and the Auto-Correlation Function.

(6) quantify various data quality metrics from photometric time-series data which can be used by the user to assess data reliability

The calling sequence is:

python tess_cutouts.py use_con make_plots file_cat t_filename

where use_con is the toggle for applying the contamination calculation (yes=1, no=0)
      make_plots gives the user the option to make plots (yes=1, no=0)
      file_cat is a string expression used to reference the files produced.
      t_filename is the name of the input file required for analysis.

In this module, the data is downloaded from TESSCut (Brasseur et al. 2019) -- a service which allows the user to acquire a stack of n 20x20 pixel "postage-stamp" image frames ordered in time sequence and centered using the celestial coordinates provided by the user, where n represents the number of TESS sectors available for download. It uses modules from the TesscutClass (https://astroquery.readthedocs.io/en/latest/api/astroquery.mast.TesscutClass.html - part of the astroquery.mast package) to download the data, then applies steps 2-6, sector-by-sector.

Since there are no requirements to download the full image frames (0.5-0.7TB and 1.8-2.0TB per sector for 30 and 10 min cadence observations, respectively) this software is recommended for users who require a relatively fast extraction for a manageable number of targets (i.e., < 5000). With the correct pre-requisite Python modules and an uninterrupted internet connection, for a target with 5 sectors of TESS data, the processing time is approximately 1-2 minutes (depending whether or not the user wants to measure contamination and/or generate plots). If the user is interested in formulating a much larger survey, we recommend they obtain the bulk downloads of the TESS calibrated full-frame images (https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html) and run the "tess_large_sectors.py" module, which has been used to measure rotation periods for catalogues of >1 million targets.

The module "tess_functions.py" contains the functions called to run both tess_cutouts.py and tess_large_sectors.py

Should there be any problems in using this software please contact Alex Binks (lead author) at abinks@mit.edu

If this package is useful for research leading to publication we would appreciate the following acknowledgement:

"The data from the Transiting Exoplanet Survey Satellite (TESS) was acquired using the TESSilator software package (Binks et al. 2023)."
'''



import sys
import os
import numpy as np


# Use the sys.argv arguments to save files with appropriate naming conventions 
use_con = int(sys.argv[1])
make_plots = int(sys.argv[2])
file_cat = sys.argv[3]
t_filename = sys.argv[4]

con_file = '_'.join(['contamination', file_cat, 'tesscut'])+'.ecsv'
period_file = '_'.join(['periods', file_cat, 'tesscut'])+'.ecsv'
store_file = '_'.join(["cur", file_cat, 'tesscut'])+".txt"


# Aperture size, background annulus and positional uncertainty (all in pixels)
# A 1-pixel aperture is considered an optimal extraction following experimentation.
# The details are described in Appendix B3 of Binks et al. 2022.
Rad, SkyRad, exprf = 1.0, np.array([6.0,8.0]), 0.65

#TESS pixel size in arcseconds
pixel_size = 21.0 

#Zero-point magnitude from Vanderspek et al. 2018
Zpt, eZpt = 20.44, 0.05



from astropy.io import ascii, fits
from astropy.table import QTable, Table, Column, join


# Produce an astropy table with the relevant column names and data types. These will be filled in list comprehensions later in the program.
final_table = Table(names=['source_id', 'ra', 'dec', 'parallax', 'Gmag', 'Sector', 'Camera', 'CCD', 'log_tot_bg_star', 'log_max_bg_star', 'n_contaminants', 'Period_Max', 'Period_Gauss', 'e_Period', 'Period_2', 'power1', 'power1_power2', 'FAP_001', 'amp', 'scatter', 'fdev', 'Ndata'],
                    dtype=(str, float, float, float, float, int, int, int, float, float, int, float, float, float, float, float, float, float, float, float, float, int))


# import the remaining modules required for the TESS analysis
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from photutils.centroids import centroid_com
from astroquery.gaia import Gaia
from astroquery.mast import Tesscut
import time

start_date = time.asctime(time.localtime(time.time()))
start = time.time()
print(start_date)

from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt

import traceback

from tess_functions2 import *



mpl.rcParams.update({'font.size': 14})



# Read the data from the input file as an astropy table (ascii format)
# Ensure the source_id has a string type (not long integer).
t_targets = Table(ascii.read(t_filename), names=('source_id', 'ra', 'dec', 'parallax', 'Gmag'))
print("File in = ", t_filename)
t_targets["source_id"].astype(str)

print(t_targets)


# Generate a new store_file if one exists previously. This is used as a debugging log. 
if os.path.exists(store_file):
    os.remove(store_file)
else:
    with open(store_file, 'a') as file1:
        file1.write('Starting Time: '+str(start_date)+'\n')


# If the user selects sys.argv[1] = 1, then the contamination method
# will be implemented. Warning --- if the contamination file already
# exists and a value of 1 is selected, the file will be overwritten.
# Alternatively, if this toggle is not selected, the contamination
# parameters are all initially assigned dummy values. Then if a pre-
# existing contamination file exists, these results will be used.
# The idea here is to save the user time if they have already used
# the contamination module in a previous run. If the pre-existing
# table is only partially completed, only the targets that match the
# input list will be changed (the missing data will still have the
# dummy values).
if use_con == 1:
    t_targets = contamination(t_targets)
    t_contam.write(con_file, overwrite=True)
else:
    t_targets["log_tot_bg"] = -999.
    t_targets["log_max_bg"] = -999.
    t_targets["num_tot_bg"] = -999.
    
    if os.path.exists(con_file):
        t_contam = Table.read(con_file)
        for i in range(len(t)):
            g = (t_contam["source_id"] == t_targets["source_id"].astype(str)[i])
            if len(g) >= 1:
                t_targets["log_tot_bg"][i] = t_contam["log_tot_bg"][g][0]
                t_targets["log_max_bg"][i] = t_contam["log_max_bg"][g][0]
                t_targets["num_tot_bg"][i] = t_contam["num_tot_bg"][g][0]


# As discussed between Alex and Moritz on 10 Jan 2023, we should break this loop into a set of functions
# That way we can run different tests and see which is the most efficient/readable technique.
# For now, I (Alex) will try and explain what's going on in the loops.

for i in range(len(t_targets)): # loop over each target in the input table
    cutout_coord = SkyCoord(t_targets["ra"][i], t_targets["dec"][i], unit="deg")
    # Download the fits files using Tesscut.download_cutouts
    manifest = Tesscut.download_cutouts(coordinates=cutout_coord, size=20, sector=None)
    for j in range(len(manifest)): # loop over each TESS sector returned for the current target.
        file_in = manifest["Local Path"][j]
        t_sp = file_in.split('-')
        scc = [int(t_sp[1][1:]), int(t_sp[2]), int(t_sp[3][0])] # simply extract the sector, ccd and camera numbers from the fits file.

        # create a numpy array which holds the photometric measurements for each timestep from the aper_run_cutouts routine.
        with open(store_file, 'a') as file1:
            file1.write(f'{t_targets["source_id"][i]}, {j+1}/{len(manifest)}\n')
        x = np.array(aper_run_cutouts(file_in))

        # make an astropy table out of the array, and remove magnitudes brigher than T=0 or fainter than T=25
        zz = Table(x, names=('num', 'xcenter', 'ycenter', 'flux', 'flux_err',
                             'bkg', 'total_bkg', 'flux_corr', 'mag',
                             'mag_err', 'time', 'qual'),
                      dtype=(int, float, float, float, float, float, float, float,
                             float, float, float, int))
        zg = zz[((zz['mag'] < 25.) & (zz['mag'] > 0.0))]
        clean_norm_lc, original_norm_lc = make_lc(zg)
        LS_dict = run_LS(clean_norm_lc)
        if make_plots == 1:
            make_LC_plots(clean_norm_lc, original_norm_lc, LS_dict, scc, t_targets[i])
        datarow = [
                   str(t_targets["source_id"][i]),
                   t_targets["ra"][i],
                   t_targets["dec"][i],
                   t_targets["parallax"][i],
                   t_targets["Gmag"][i],
                   scc[0],
                   scc[1],
                   scc[2],
                   t_targets["log_tot_bg"][i],
                   t_targets["log_max_bg"][i],
                   t_targets["num_tot_bg"][i],
                   LS_dict['period_best'],
                   LS_dict['Gauss_fit_peak_parameters'][1],
                   LS_dict['Gauss_fit_peak_parameters'][2],
                   LS_dict['period_second'],
                   LS_dict['power_best'],
                   LS_dict['power_best']/LS_dict['power_second'],
                   LS_dict['FAPs'][2],
                   LS_dict['pops_vals'][1],
                   LS_dict['phase_scatter'],
                   LS_dict['frac_phase_outliers'],
                   LS_dict['Ndata']
                  ]
        final_table.add_row(datarow)
final_table.write(period_file, overwrite=True)
