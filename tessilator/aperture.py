'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

This module contains functions to perform aperture photmetry.

The aperture photometry is defined by a single function 'aper_run', which reads
the TESS image files, performs aperture phtoometry and returns an astropy table
containing the timestamps, magnitudes and fluxes derived from aperture
photometry. Additionally, one can use the function 'calc_radius', which
evaluates the relative brightness of neighbouring pixels to automatically
determine the size of the aperture radius. If the full-frame images are used, a
function named 'get_xy_pos' provides a WCS transformation to convert celestial
coordinates to image (X-Y) pixels. 
'''

###############################################################################
####################################IMPORTS####################################
###############################################################################
# Internal
import warnings
import inspect
import sys

# Third party
import numpy as np
import json

from scipy import stats

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry, ApertureStats



# Local application
from .fixedconstants import *
from .logger import logger_tessilator
###############################################################################
###############################################################################
###############################################################################


# initialize the logger object
logger = logger_tessilator(__name__)



def get_xy_pos(targets, head):
    '''Locate the X-Y position for targets in a given Sector/CCD/Camera mode
    
    The function reads in the RA and DEC positions of each target, and the
    metadata (header) of the input fits frame containing the WCS information.
    A WCS transformation is attempted first, which uses the
    `world_to_array_index` module to assign pixel values that match the
    indexing order of numpy arrays. If the WCS transformation fails, then the
    `Xpos` and `Ypos` columns from the input table are used. If `Xpos` and
    `Ypos` are not available, the function returns an error.

    parameters
    ----------
    targets : `astropy.table.Table`
        The table of input data with celestial coordinates.
    head : `astropy.io.fits`
        The fits header containing the WCS coordinate details.

    returns
    -------
    positions : `tuple`
        A tuple of X-Y pixel positions for each target.
    '''


    try:
        w = WCS(head)
        c = SkyCoord(targets['ra'], targets['dec'], unit=u.deg, frame='icrs')
        y_obj, x_obj = w.world_to_array_index(c)
        if len(y_obj) > 1:
            positions = tuple(zip(x_obj, y_obj))
        else:
            positions = (x_obj[0], y_obj[0])
        logger.info("The WCS coordinates were successfully applied.")
        return positions

    except:
        if ("Xpos" in targets.colnames) and ("Ypos" in targets.colnames):
            positions = tuple(zip(targets["Xpos"], targets["Ypos"]))
            logger.warning("XY positions used directly - the aperture will be offset by a few sub-pixels!")
        else:
            logger.error("Couldn't get the WCS coordinates to work...")
            return
    


def calc_rad(flux_vals, positions, f_lim=0.2, max_rad=3, default_rad=1):
    '''Calculate the appropriate pixel radius for the aperture
    
    This function uses a basic algorithm to calculate the most appropriate
    radius size to use for the circular aperture photometry of the TESS image
    frames. If the ratio of the median value of neighbouring (8, in a square
    surrounding the central pixel) pixels compared to the central pixel is
    greater than 'f_lim', then expand the radius by one pixel, and test the
    next set of surrounding pixels. If, after 'n_pix' pixels the condition is
    still satisfied, set the pixel radius equal to 1. The latter constraint is
    intended to avoid contamination from neighbouring sources.
    
    parameters
    ----------
    flux_vals : `np.array`
        The raw flux values from each pixel in the image.
    positions : `tuple`
        The X,Y position of the central pixel
    f_lim : `float`, optional, default=0.2
        The limiting threshold flux for the criterion.
    max_rad : `int`, optional, default=3
        The maximum number of pixels for the aperture radius.
    default_rad : `int`, optional, default=1
        The default aperture radius to be used in case of an error.

    returns
    -------
    aper_rad : `float`
        The pixel radius.
    '''
    try:
        x0, y0 = int(positions[0]), int(positions[1])
        f_max = flux_vals[x0,y0]
        mask_ori = np.zeros([flux_vals.shape[0], flux_vals.shape[1]])
        mask = mask_ori
        i = 1
        while i <= max_rad:
            mask[x0-i:x0+i+1,y0-i] = 1
            mask[x0-i:x0+i+1,y0+i] = 1
            mask[x0-i,y0-i:y0+i] = 1
            mask[x0+i,y0-i:y0+i] = 1
            f_sum = mask*flux_vals
            f_new = np.median(f_sum[np.where(f_sum > 0)])
            if f_new/f_max < f_lim:
                break
            else:
                mask = mask_ori
                i += 1
        if i > 3:
            i = 1
        aper_rad = i-0.5
    except:
        logger.warning("The aperture radius could not be calculated with this algorithm. The default value will be used.")
        aper_rad = default_rad-0.5
    return aper_rad


def aper_run(file_in, targets, FixRad=None, SkyRad=(6.,8.), XY_pos=(10.,10.)):
    '''Perform aperture photometry for the image data.

    This function reads in each fits file, determines the pixel radius for an
    image aperture and performs aperture photometry. A table of aperture
    photometry results is returned, which forms the raw lightcurve to be
    processed in subsequent functions.

    parameters
    ----------
    file_in : `str`
        Name of the fits file containing image data
    targets : `astropy.table.Table`
        The table of input data
    FixRad : `float`, optional, default=1.
        The pixel radius to define the circular area for the aperture
    SkyRad : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux
    XY_pos : `tuple`, optional, default=(10.,10.)
        The X-Y centroid (in pixels) of the aperture.

    returns
    -------
    full_phot_table : `astropy.table.Table`
        The formatted table containing results from the aperture photometry.
    '''
    if isinstance(file_in, np.ndarray):
        fits_files = file_in
    else:
        fits_files = [file_in]

    full_phot_table = Table(names=('run_no','id', 'aperture_rad', 'xcenter', 'ycenter', 'flux',
                                   'flux_err', 'bkg', 'total_bkg',
                                   'reg_oflux', 'mag', 'mag_err', 'time'),
                            dtype=(int, str, float, float, float, float, float, float,
                                   float, float, float, float, float))
    print(fits_files)
    for f_num, f_file in enumerate(fits_files):
        print(f'{f_num}, {f_file}, running aperture photometry')
        try:
            with fits.open(f_file) as hdul:
                data = hdul[1].data
                if data.ndim == 1:
#                    head = hdul[1].header
                    if "FLUX_ERR" in data.names:
                        n_steps = data.shape[0]-1
                        flux_vals = data["FLUX"]
                        qual_val = data["QUALITY"]
                        time_val = data["TIME"]
                        erro_vals = data["FLUX_ERR"]
                    else:
                        n_steps = 1
                        flux_vals = data["FLUX"]
                        qual_val = [data["QUALITY"][0]]
                        time_val = [data["TIME"][0]]
                        erro_vals = 0.001*flux_vals
                    positions = XY_pos
                elif data.ndim == 2:
                    n_steps = 1
                    head_meta = hdul[0].header
                    head_data = hdul[1].header
                    qual_val = [head_data["DQUALITY"]]
                    time_val = [(head_meta['TSTART']) + (head_meta['TSTOP'])/2.]
                    flux_vals = [data]
                    erro_vals = [hdul[2].data]
                    positions = get_xy_pos(targets, head_data)

                if not FixRad:
                    rad_val = []
                    for n_step in range(n_steps):
                    #define a circular aperture around all objects
                        rad_x = calc_rad(flux_vals[n_step], positions)
                        rad_val.append(rad_x)
                    if len(rad_val) > 1:
                        Rad = stats.mode(np.array(rad_val), keepdims=False)[0]
                    else:
                        Rad = rad_val[0]
                else:
                    rad_vel = np.repeat(FixRad, n_steps)
                    Rad = FixRad
                 
                for n_step in range(n_steps):
                    if qual_val[n_step] == 0:
                        aperture = CircularAperture(positions, Rad)
                        #select a background annulus
                        annulus_aperture = CircularAnnulus(positions,
                                                           SkyRad[0],
                                                           SkyRad[1])
                        if flux_vals[:][:][n_step].ndim == 1:
                            flux_ap = flux_vals
                            erro_ap = erro_vals
                        else:
                            flux_ap = flux_vals[:][:][n_step]
                            erro_ap = erro_vals[:][:][n_step]
                        #get the image statistics for the background annulus
                        aperstats = ApertureStats(flux_ap, annulus_aperture)
                        #obtain the raw (source+background) flux
                        with np.errstate(invalid='ignore'):
                            t = aperture_photometry(flux_ap, aperture,
                                                    error=erro_ap)

                        #calculate the background contribution to the aperture
                        aperture_area = aperture.area_overlap(flux_ap)
                        #print out the data to "t"
                        t['run_no'] = n_step
                        t['aperture_rad'] = rad_val[n_step]
                        t['id'] = targets['source_id']
                        t['id'] = t['id'].astype(str)
                        t['bkg'] = aperstats.median
                        t['tot_bkg'] = \
                            t['bkg'] * aperture_area
                        t['ap_sum_sub'] = \
                            t['aperture_sum'] - t['tot_bkg']
                        t['mag'] = -999.
                        t['mag_err'] = -999.
                        g = np.where(t['ap_sum_sub'] > 0.)[0]
                        t['mag'][g] = -2.5*np.log10(t['ap_sum_sub'][g].data)+Zpt
                        t['mag_err'][g] = np.abs((-2.5/np.log(10))*\
                            t['aperture_sum_err'][g].data/\
                            t['aperture_sum'][g].data)
                        t['time'] = time_val[n_step]
                        fix_cols = ['run_no', 'id', 'aperture_rad', 'xcenter', 'ycenter',
                                    'aperture_sum', 'aperture_sum_err', 'bkg',
                                    'tot_bkg', 'ap_sum_sub', 'mag',
                                    'mag_err', 'time']
                        t = t[fix_cols]
                        for r in range(len(t)):
                            full_phot_table.add_row(t[r])
        except:
            print(f"There is a problem opening the file {f_file}")
            logger.error(f"There is a problem opening the file {f_file}")
            continue
    return full_phot_table, Rad
    
    

__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__], predicate = lambda f: inspect.isfunction(f) and f.__module__ == __name__)]
