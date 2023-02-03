from TESSilator import tess_cutouts

# Standard library imports
import sys
import os
import math
import time
import traceback
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning)



# Third party imports
import numpy as np

import pyinputplus as pyip

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy.io import ascii, fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.mast import Tesscut

from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



# Local application imports
from TESSilator.tess_stars2px import tess_stars2px_function_entry






# Use the sys.argv arguments to save files with appropriate naming conventions
# Or if these are not fully provided, prompt the user to provide input.
if len(sys.argv) != 6:
    flux_con = pyip.inputInt("Do you want to search for contaminants? 1=yes, 0=no", min=0, max=1)
    LC_con = pyip.inputInt("Would you like to calculate period data for the contaminants? 1=yes, 0=no", min=0, max=1)
    make_plots = pyip.inputInt("Would you like lightcurve/periodogram/phase plots? 1=yes, 0=no", min=0, max=1)
    file_ref = pyip.inputStr("Enter the unique name for referencing the output files")
    while True:
        t_filename = pyip.inputStr("Enter the file name of your input table or object. If this is an object please use double quotations around the target identifier.")
        if t_filename.startswith('"') & t_filename.endswith('"'):
            t_name = t_filename[1:-1]
            t_name_joined = t_name.replace(' ', '_')+'.dat'
            with open(t_name_joined, 'a') as single_target:
                single_target.write(t_name)
            t_filename = t_name_joined
            break
        if os.path.exists(t_filename):
            break
else:
    flux_con = int(sys.argv[1])
    LC_con = int(sys.argv[2])
    make_plots = int(sys.argv[3])
    file_ref = sys.argv[4]
    t_filename = sys.argv[5]


con_file, period_file, store_file = tess_cutouts.setupFilenames(file_ref)
t_targets = tess_cutouts.readData(t_filename)
print("Done reading the table and formatting, now calculating the contamination.")
t_targets = tess_cutouts.collectContaminationData(t_targets, con_file, flux_con=True, LC_con=0)
print("Done calculating the contamination, now iterating over each source.")
tess_cutouts.iterate_sources(t_targets, store_file)
        
print(f"Finished {len(t_targets)} in {(time.time() - start)/60.:.2f} minutes")
