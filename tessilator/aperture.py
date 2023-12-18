'''

Alexander Binks & Moritz Guenther, December 2023

Licence: MIT 2023

This module contains functions to perform aperture photmetry.

The aperture photometry is defined by a single function 'aper_run', which reads the TESS image files, performs aperture phtoometry and returns an astropy table containing the times, magnitudes and fluxes from aperture photometry.
'''

# imports
import logging
__all__ = ['aper_run', 'get_xy_pos', 'logger']


import warnings

# Third party imports
import numpy as np
import json


from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u


from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry, ApertureStats


# Local application imports
from .fixedconstants import *


# initialize the logger object
logger = logging.getLogger(__name__)



def get_xy_pos(targets, head):
    '''Locate the X-Y position for targets in a given Sector/CCD/Camera mode

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
    w = WCS(head)
    c = SkyCoord(targets['ra'], targets['dec'], unit=u.deg, frame='icrs')
    try:
        y_obj, x_obj = w.world_to_array_index(c)
    except:
        logger.warning("Couldn't get the WCS coordinates to work...")
        positions = tuple(zip(targets["Xpos"], targets["Ypos"]))
        return positions
    positions = tuple(zip(x_obj, y_obj))
    return positions
    




def aper_run(file_in, targets, Rad=1., SkyRad=(6.,8.), XY_pos=(10.,10.)):
    '''Perform aperture photometry for the image data.

    This function reads in each fits file, and performs aperture photometry.
    A table of aperture photometry results is returned, which forms the raw
    lightcurve to be processed in subsequent functions.

    parameters
    ----------
    file_in : `str`
        Name of the fits file containing image data
    targets : `astropy.table.Table`
        The table of input data
    Rad : `float`, optional, default=1.
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

    full_phot_table = Table(names=('run_no','id', 'xcenter', 'ycenter', 'flux',
                                   'flux_err', 'bkg', 'total_bkg',
                                   'reg_oflux', 'mag', 'mag_err', 'time'),
                            dtype=(int, str, float, float, float, float, float,
                                   float, float, float, float, float))
    print(fits_files)
    for f_num, f_file in enumerate(fits_files):
        print(f'{f_num}, {f_file}, running aperture photometry')
        try:
            with fits.open(f_file) as hdul:
                data = hdul[1].data
                if data.ndim == 1:
                    head = hdul[0].header
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
                    head = hdul[1].header
                    qual_val = [head["DQUALITY"]]
                    time_val = [(head['TSTART'] + head['TSTOP'])/2.]
                    flux_vals = [data]
                    erro_vals = [hdul[2].data]
                    positions = get_xy_pos(targets, head)
                print('okay, n_steps=', n_steps)

                for n_step in range(n_steps):
                    if qual_val[n_step] == 0:
                        #define a circular aperture around all objects
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
                        fix_cols = ['run_no', 'id', 'xcenter', 'ycenter',
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
    return full_phot_table
    
    
    

