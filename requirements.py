import sys
import os
import pyinputplus as pyip
import math

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.io import ascii, fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.mast import Tesscut

from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

from tess_stars2px import tess_stars2px_function_entry

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import traceback
import time

import numpy as np

import warnings
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning)

