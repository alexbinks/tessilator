'''
Alexander Binks and Moritz Guenther, 2023

The TESSilator

Licence: MIT

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
# Or if these are not fully provided, prompt the user to provide input.
if len(sys.argv) != 6:
    use_con = int(input("Do you want to search for contaminants? 1=yes, 0=no"))
    calc_con = int(input("Would you like to calculate period data for the contaminants? 1=yes, 0=no"))
    make_plots = int(input("Would you like lightcurve/periodogram/phase plots?"))
    file_cat = input("Enter the unique name for referencing the output files") 
    t_filename = input("Enter the file name of your input table")
else:
    use_con = int(sys.argv[1])
    calc_con = int(sys.argv[2])
    make_plots = int(sys.argv[3])
    file_cat = sys.argv[4]
    t_filename = sys.argv[5]


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
final_table = Table(names=['source_id', 'ra', 'dec', 'parallax', 'Gmag', 'Sector', 'Camera', 'CCD', 'log_tot_bg_star', 'log_max_bg_star', 'n_contaminants', 'Period_Max', 'Period_Gauss', 'e_Period', 'Period_2', 'power1', 'power1_power2', 'FAP_001', 'amp', 'scatter', 'fdev', 'Ndata', 'cont_flags'],
                    dtype=(str, float, float, float, float, int, int, int, float, float, int, float, float, float, float, float, float, float, float, float, float, int, str))


# import the remaining modules required for the TESS analysis
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from photutils.centroids import centroid_com
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.mast import Tesscut
import time

start_date = time.asctime(time.localtime(time.time()))
start = time.time()
print(start_date)

from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

import traceback

from tess_functions2 import *



mpl.rcParams.update({'font.size': 14})


isHeader = 0
# Read the data from the input file as an astropy table (ascii format)
# Ensure the source_id has a string type (not long integer).
if isHeader == 0:
    t = ascii.read(t_filename, format='no_header')
elif isHeader == 1:
    t = ascii.read(t_filename)
t_targets = GetGAIAData(t)


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
    t_targets, con_table_full = contamination(t_targets)
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


def single_epoch(file_in, t_target, cutout_size):
    t_sp = file_in.split('-')
    scc = [int(t_sp[1][1:]), int(t_sp[2]), int(t_sp[3][0])] # simply extract the sector, ccd and camera numbers from the fits file.
    print(t_target)
# create a numpy array which holds the photometric measurements for each timestep from the aper_run_cutouts routine.
    
    XY_ctr = (cutout_size/2., cutout_size/2.)
    phot_targ = np.array(aper_run_cutouts(file_in, XY_ctr))
    phot_targ_table = make_phot_table(phot_targ)
    if len(phot_targ_table) < 50 or phot_targ_table is None:
        return
    clean_norm_lc, original_norm_lc = make_lc(phot_targ_table)
    if len(clean_norm_lc) > 0:
        d_target = run_LS(clean_norm_lc)
        print(d_target["period_best"], d_target["pops_vals"])
        labels_cont = 'z'    


    if calc_con == 1:
        if use_con != 1:
            print("Contaminants not identified! Please toggle calc_con=1\nContinuing program using only the target.")
        else:
            phot_cont = {}
            labels_cont = ''
            con_table = con_table_full[con_table_full["source_id_target"] == t_target["source_id"]]
            X_con, Y_con = find_XY_cont(file_in, con_table)
            X_con += XY_ctr[0]
            Y_con += XY_ctr[1]
            XY_arr = np.array([X_con, Y_con]).T
            for z in range(len(XY_arr)):
                XY_con = tuple((XY_arr[z][0], XY_arr[z][1]))
                phot_cont[z] = np.array(aper_run_cutouts(file_in, XY_con))
                phot_cont_table = make_phot_table(phot_cont[z])
                if len(phot_cont_table) > 50 or phot_cont_table is not None:
                    clean_norm_lc_cont, original_norm_lc_cont = make_lc(phot_cont_table)
                    if len(clean_norm_lc) > 0:
                        d_target = run_LS(clean_norm_lc)
                        d_cont = run_LS(clean_norm_lc_cont)
                        labels_cont += isPeriodCont(d_target, d_cont, con_table[z])
                    else:
                        labels_cont += 'd'
                else:
                    labels_cont += 'd'

# make an astropy table out of the array
    if make_plots == 1:
        make_LC_plots(file_in, clean_norm_lc, original_norm_lc, d_target, scc, t_target)


    datarow = [
               str(t_target["source_id"]),
               t_target["ra"],
               t_target["dec"],
               t_target["parallax"],
               t_target["Gmag"],
               scc[0],
               scc[1],
               scc[2],
               t_target["log_tot_bg"],
               t_target["log_max_bg"],
               t_target["num_tot_bg"],
               d_target['period_best'],
               d_target['Gauss_fit_peak_parameters'][1],
               d_target['Gauss_fit_peak_parameters'][2],
               d_target['period_second'],
               d_target['power_best'],
               d_target['power_best']/d_target['power_second'],
               d_target['FAPs'][2],
               d_target['pops_vals'][1],
               d_target['phase_scatter'],
               d_target['frac_phase_outliers'],
               d_target['Ndata'],
               labels_cont
              ]
    final_table.add_row(datarow)
    final_table.write(period_file, overwrite=True)


def iterate_source(coord, target, cutout_size=20):
    manifest = Tesscut.download_cutouts(coordinates=coord, size=cutout_size, sector=None)
    for j in range(len(manifest)):
         print(f"{j+1} of {len(manifest)} sectors")
         file_in = manifest["Local Path"][j]
         with open(store_file, 'a') as file1:
             file1.write(f'{target["source_id"]}, {j+1}/{len(manifest)}\n')
         t = single_epoch(file_in, target, cutout_size)

def iterate_sources(t_targets):
    for i, target in enumerate(t_targets):
        print(f"{target['source_id']}, star # {i+1} of {len(t_targets)}")
        coo = SkyCoord(target["ra"], target["dec"], unit="deg")
        table_one_source = iterate_source(coo, target)
        

final_result = iterate_sources(t_targets)
print(f"Finished {len(t_targets)} in {(time.time() - start)/60.:.2f} minutes")
