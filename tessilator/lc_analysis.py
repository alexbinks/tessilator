'''

Alexander Binks & Moritz Guenther, July 2023

Licence: MIT 2023

This module contains functions to perform aperture photmetry and clean lightcurves. These are:

1)  aper_run - returns a numpy array containing the times, magnitudes and
    fluxes from aperture photometry.
    
2)  clean_lc/detrend_lc/make_lc - these 3 functions are made to correct for systematics
    using the co-trending basis vectors, remove spurious data points, ensure that only
    "strings" of contiguous data are being processed, and each string is detrended using
    a series of decision processes, including normalisation, selection of low-order
    polynomial fits and whether to detrend the lightcurve as a whole, or in "strings".
    Finally, the lightcurve is pieced together and make_lc returns a table containing
    the data ready for periodogram analysis.
    
3)  run_LS - function to conduct Lomb-Scargle periodogram analysis. Returns a table
    with period measurements, plus several data quality flags. If required a plot of
    the lightcurve, periodogram and phase-folded lightcurve is provided.

'''

# imports
import logging
__all__ = ['aic_selector', 'aper_run', 'clean_edges_outlier', 'clean_edges_scatter', 'detrend_lc',
           'get_time_segments', 'get_xy_pos', 'logger', 'make_lc',
           'normalisation_choice', 'remove_sparse_data', 'sin_fit', 'test_cbv_fit']


import warnings

# Third party imports
import numpy as np
import os
import json


from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.stats import akaike_info_criterion_lsq


from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry, ApertureStats
from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit
from scipy.stats import iqr

import itertools as it
from operator import itemgetter

# Local application imports
from .fixedconstants import *


# initialize the logger object
logger = logging.getLogger(__name__)
#logger_aq = logging.getLogger("astroquery")
#logger_aq.setLevel(logging.ERROR)    


#class NumpyEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, np.ndarray):
#            return obj.tolist()
#        return json.JSONEncoder.default(self, obj)



def aic_selector(x, y, poly_max=3, cov_min=1e-10):
    '''Chooses the most appropriate polynomial fit, using the Aikaike Information Criterion
    
    This function uses the Aikaike Information Criterion to find the most appropriate polynomial order to a set of X, Y data points.
    
    parameters
    ----------
    x : `Iterable`
        The independent variable
    y : `Iterable`
        The dependent variable
    err : `Iterable`
        The error bar on 'y'
    poly_max : `int`, optional, default=10
        The maximum polynomial order to test

    returns
    -------
    poly_ord : `int`
        The best polynomial order
    coeffs : `list`
        The polynomial coefficients.
    
    '''
    
    q = 0
    N = float(len(x))
    try:
        while q < poly_max:
            k1, k2 = q+1, q+2
            p1, r1, _,_,_ = np.polyfit(x, y, q, full=True)
            p2, r2, _,_,_ = np.polyfit(x, y, q+1, full=True)
            with np.errstate(invalid='ignore'):
                SSR1 = np.sum((np.polyval(p1, x) - y)**2)
                SSR2 = np.sum((np.polyval(p2, x) - y)**2)
            AIC1 = akaike_info_criterion_lsq(SSR1, k1, N)
            AIC2 = akaike_info_criterion_lsq(SSR2, k2, N)
            if (AIC1 < (AIC2 + 2)) | (r1 < cov_min):
                poly_ord, coeffs = q, p1
                return poly_ord, list(coeffs)
            else:
                q += 1
                if q >= poly_max:
                    poly_ord, coeffs = q, p2
                    return poly_ord, list(coeffs)
    except:
        return 0, [1.0]




def aper_run(file_in, targets, Rad=1., SkyRad=[6.,8.], XY_pos=(10.,10.)):
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
    
    
def clean_flux_algorithm(g):
    '''A basic algorithm that trims both sides of a contiguous data string
    if a condition is not satisfied, until the condition is met for the
    first time.
    
    parameters
    ----------
    g : `Iter`
        The array of boolean results indicating if a condition is met (=1) or not (=0)
    
    returns
    -------
    start : `int`
        The trimmed start point of the array
    fin : `int`
        The trimmed final part of the array
    '''
    i, j = 0, len(g)-1
    while i < j:
        if g[i] != 1:
            i+=1
        else:
            start=i
            break
    while j > 0:
        if g[j] != 1:
            j-=1
        else:
            fin=j
            break
    if j <= i:
        start, fin = 0, len(g)-1
        return start, fin
    else:
        return start, fin
    
    

    
def clean_edges_outlier(f, MAD_fac=5.):
    '''Remove spurious data points at the start and end parts of the lightcurve.

    The start and end point of each data section must have a flux value within
    a given number of MAD from the median flux in the sector. This is done
    because often after large time gaps the temperature of the sensors changes,
    and including these components is likely to just result in a signal from
    instrumental noise.

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.

    returns
    -------
    start : `int`
       The start index for the data string.
    fin : `int`
       The end index for the data string.
    '''
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    f_diff = np.zeros(len(f))
    f_diff[1:] = np.diff(f)
    f_diff_med = np.median(np.absolute(f_diff))
    try: 
        g = (np.abs(f-f_med) < MAD_fac*f_MAD).astype(int)
        start, fin = clean_flux_algorithm(g)
        
    except:
        logger.error(f'Something went wrong with the arrays when doing the lightcurve edge clipping')
        start, fin = 0, len(g)-1
    return start, fin


def clean_edges_scatter(f, MAD_fac=5., len_IQR=[11,10]):
    '''Remove data points from the lightcurve that are likely to be spurious.

    Occasionally there are lightcurves that appear to have very scattered data at the start and end
    points of the lightcurves. These can degrade the quality of the periodogram analysis, or even return
    an incorrect period.
    
    The idea is to group the first "n_avg" datapoints, and calculate the median absolute deviation (MAD).
    If this local MAD value is greater (less) than "MAD_fac" times the MAD of the full lightcurve, then
    we flag this point with 0 (1). The first and last "(n_avg-1)/2" in the lightcurve are given a constant
    value. If the first/last MAD comparison yield a "1" value, then we do no extra cleaning to the lightcurve. 

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.
    n_avg : `int`, optional, default=21
        The number of data points to be used in the local MAD value.

    returns
    -------
    s : `int`
       The start index for the data string.
    f : `int`
       The end index for the data string.
    '''
    n_avg =min(len_IQR[0], int(len(f)/len_IQR[1]))
    if n_avg // 2 == 0:
        n_avg += 1
    p_e = int((n_avg-1)/2)
    # get the median time and flux, the median absolute deviation in flux
    # and the time difference for each neighbouring point.
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    f_diff = np.zeros(len(f))
    f_diff[1:] = np.diff(f)
    f_diff_med = np.median(np.absolute(f_diff))
    
#    f_mean = np.array(np.convolve(f_diff, np.ones(n_avg)/n_avg, mode='valid'))
    f_x = np.array([MAD(f[i:i+n_avg], scale='normal') for i in range(len(f)-n_avg+1)])
#    f_x = np.array([iqr(f[i:i+n_avg]) for i in range(len(f)-n_avg+1)])
    f_diff_run = np.pad(f_x, (p_e, p_e), 'constant', constant_values=(MAD_fac*f_MAD, MAD_fac*f_MAD))
    try: 
        g = (np.abs(f_diff_run) < MAD_fac*f_diff_med).astype(int)
        start, fin = clean_flux_algorithm(g)
        if start <= p_e:
            start = 0
        if fin >= len(g)-1-(2*p_e+1):
            fin = len(g)-1
    except:
        logger.error(f'Something went wrong with the arrays when doing the lightcurve edge clipping')
        start, fin = 0, len(g)-1
    return start, fin



def detrend_lc(t,f,lc, MAD_fac=2., poly_max=3):
    '''Detrend and normalise the lightcurves.

    | This function runs 3 major operations to detrend the lightcurve, as follows:
    | 1. Choose the best detrending polynomial using the Aikaike Information Criterion, and detrend the full lightcurve.
    | 2. Decide whether to use the detrended lightcurve from part 1, or to separate the lightcurve into individual components and detrend each one separately.
    | 3. Return the detrended flux.

    parameters
    ----------
    t : `Iterable'
        the time component of the lightcurve
    f : `Iterable`
        the flux component of the lightcurve.
    err : `Iterable`
        the flux error component of the lightcurve.
    lc : `Iterable`
        The index representing the lightcurve component. Note this
        must be indexed starting from 1.
    MAD_fac : `float`, optional, default = 2.
        The factor to multiply the median absolute deviation by.
    poly_max : `int`, optional, default=8
        The maximum order of the polynomial fit.

    returns
    -------
    f_norm : `Iterable'
        The corrected lightcurve after the detrending procedures. 
    '''

    # 1. Choose the best detrending polynomial using the Aikaike Information Criterion, and
    #    detrend the lightcurve as a whole.
    s_fit_0, coeffs_0 = aic_selector(t, f, poly_max=poly_max)
    f_norm = f/np.polyval(coeffs_0, t)

    # 2. Decide whether to use the detrended lightcurve from part 1, or to separate the
    #    lightcurve into individual components and detrend each one separately
    norm_comp, f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD = \
                    normalisation_choice(t, f, lc, MAD_fac=MAD_fac, poly_max=poly_max)

    # 3. Detrend the lightcurve following steps 1 and 2.
    s_fit, coeffs = [], []
    if norm_comp:
        # normalise each component separately.
        f_detrend = np.array([])
        for l in np.unique(lc):
            g = np.array(lc == l)
            s_fit_n, coeffs_n = aic_selector(t[g], f[g], poly_max=poly_max)
            s_fit.append(s_fit_n)
            coeffs.append(coeffs_n)
            f_n = f[g]/np.polyval(coeffs_n, t[g])
            f_detrend = np.append(f_detrend, f_n)
        f_norm = f_detrend
    else:
        # normalise the entire lightcurve as a whole
        f_norm = f_norm
    detr_dict = {'norm_comp' : norm_comp,
                 'f1_at_f2_0' : f1_at_f2_0,
                 'f2_at_f2_0' : f2_at_f2_0,
                 'f1_MAD' : f1_MAD,
                 'f2_MAD' : f2_MAD,
                 's_fit' : s_fit,
                 'coeffs' : coeffs}
    return f_norm, detr_dict


def get_time_segments(t, t_fac=10.):
    td     = np.zeros(len(t))
    td[1:] = np.diff(t)
    t_arr = (td <= t_fac*np.median(td)).astype(int)
    groups = (list(group) for key, group in it.groupby(enumerate(t_arr), key=itemgetter(1))
                      if key)
    ss = [[group[0][0], group[-1][0]] for group in groups if group[-1][0] > group[0][0]]
    ss = np.array(ss).T
    ds, df = ss[0,:], ss[1,:]
    ds[1:] = [ds[i]-1 for i in range(1,len(ds))]
    df = np.array([(i+1) for i in df])
    return ds, df


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
    
    
def make_lc(phot_table, name_lc='target', store_lc=False, lc_dir='lc'):
    '''Construct the normalised TESS lightcurve.

    | The function runs the following tasks:
    | (1) Read the table produced from the aperture photometry
    | (2) Normalise the lightcurve using the median flux value.
    | (4) Detrend the lightcurve using "detrend_lc"
    | (5) Return the original normalised lightcurve and the cleaned lightcurve

    parameters
    ----------
    phot_table : `astropy.table.Table` or `dict`
        | The data table containing aperture photometry returned by aper_run.py. Columns must include:
        | "time" -> The time coordinate for each image
        | "mag" -> The target magnitude
        | "reg_oflux" or "cbv_oflux" -> The total flux subtracted by the background flux
        | "flux_err" -> The error on flux_corr
    name_lc : `str`, optional, default='target'
        The name of the file which the lightcurve data will be saved to.
        The target name
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True

    returns
    -------
    final_tabs : `list`
        A list of tables containing the lightcurve data
        These are for the original lightcurve, and the cbv-corrected lightcurve
        if required and it satisfies the criteria from test_cbv_fit.
    '''
    
    f_labels = ['reg_oflux']
    cbv_ret = False
    if "cbv_oflux" in phot_table.colnames:
        f_labels.append('cbv_oflux')
        use_cbv = test_cbv_fit(phot_table["time"], phot_table["reg_oflux"], phot_table["cbv_oflux"])
        if use_cbv:
            cbv_ret = True

    final_tabs = []
    for f_label in f_labels:
        final_lc = {}
        final_lc["time"] = phot_table["time"].data
        final_lc["mag"] = phot_table["mag"].data
        final_lc[f'{f_label}'] = phot_table[f'{f_label}'].data
        final_lc["eflux"] = phot_table["flux_err"].data
        flux_dict, detr_dict = run_make_lc_steps(final_lc, f_label)
        if len(flux_dict["time"]) > 50: 
            flux_tab = Table(flux_dict)
            if f_label == "reg_oflux":
                final_tabs.append(flux_tab)
            if (f_label == "cbv_oflux") and (cbv_ret):
                final_tabs.append(flux_tab)
            if store_lc:
                path_exist = os.path.exists(f'./{lc_dir}')
                if not path_exist:
                    os.makedirs(f'./{lc_dir}')
                flux_tab.write(f'./{lc_dir}/{name_lc}_{f_label}.csv', format='csv', overwrite=True)
                with open(f'./{lc_dir}/{name_lc}_{f_label}.json', 'w') as convert_file:
                    convert_file.write(json.dumps(detr_dict))
    return final_tabs



def normalisation_choice(t_orig, f_orig, lc_part, MAD_fac=2., poly_max=4):
    '''Choose how to detrend the lightcurve, either as one component or in parts.
    
    There are always at least two data components in a TESS sector because of the finite time needed for data retrieval. This can sometimes lead to discontinuities between components because of TESS systematics and the temperature gradients across the photometer. These discontinuities can cause the phase of the sinusoidal fit to change, leading to low power output in the periodograms. Alternatively, if the data are all detrended individually, but the data is relatively continuous, this can lead to shorter period measurements.
    
    The idea with this function is that a polynomial fit is made to each component (chosen using the Aikaike Information Criterion). The extrapolated flux value from component 1 is calculated at the point where component 2 starts. If the difference between this value and the actual normalised flux at this point is greater than a given threshold, the data should be detrended separately. Otherwise the full lightcurve can be detrended as a whole.
    
    parameters
    ----------
    t_orig : `Iterable`
        The time coordinate
    f_orig : `Iterable`
        The original, normalised flux values
    e_orig : `Iterable`
        The error on "f_orig"
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    MAD_fac : `float`, optional, default = 2.
        The factor to multiply the median absolute deviation by.    

    returns
    -------
    norm_comp : `bool`
        Determines whether the data should be detrended as one whole component (False) or in parts (True)
    '''
    norm_comp = False
    Ncomp = len(np.unique(lc_part))
    f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD = [], [], [], []
    if Ncomp > 1:
        i = 1
        while i < Ncomp:
            g1 = np.array(lc_part == i)
            g2 = np.array(lc_part == i+1)
            try:
                s_fit1, coeff1 = aic_selector(t_orig[g1], f_orig[g1], poly_max=poly_max)
                s_fit2, coeff2 = aic_selector(t_orig[g2], f_orig[g2], poly_max=poly_max)
                f1_at_f2_0.append(np.polyval(coeff1, t_orig[g2][0]))
                f2_at_f2_0.append(np.polyval(coeff2, t_orig[g2][0])) # YES THIS IS SUPPOSED TO BE AT INDEX "g2"
                f1_n = f_orig[g1]/np.polyval(coeff1, t_orig[g1])
                f2_n = f_orig[g2]/np.polyval(coeff2, t_orig[g2])
                f1_MAD.append(MAD(f1_n, scale='normal'))
                f2_MAD.append(MAD(f2_n, scale='normal'))
                if abs(f1_at_f2_0[i-1] - f2_at_f2_0[i-i]) > MAD_fac*((f1_MAD[i-1]+f2_MAD[i-1])/2.):
                    norm_comp = True
                    break
                else:
                    i += 1
            except:
                logger.error('Could not run the AIC selector, probably because of a zero-division.')
                f1_at_f2_0.append(np.polyval([1], t_orig[g2][0]))
                f2_at_f2_0.append(np.polyval([1], t_orig[g2][0]))
                f1_n = f_orig[g1]
                f2_n = f_orig[g2]
                f1_MAD.append(MAD(f1_n, scale='normal'))
                f2_MAD.append(MAD(f2_n, scale='normal'))
                norm_comp = False
                break
    return norm_comp, f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD


def remove_sparse_data(x_start, x_end, std_crit=100):
    '''Removes very sparse bits of data from the lightcurve when there are 3 or more components.

    Calculate the mean (mc) and standard deviation (sc) for the number of data
    points in each component (N).
    If sc > "std_crit", then only keep components with N > std_crit.
    
    parameters
    ----------
    x_start : `Iterable`
        The starting indices for each data component
    x_end : `Iterable`
        The end indices for each data component
    std_crit: `int`
        The minimum number of data points in a component

    returns
    -------
    y_start : `np.array`
        The starting indices of the new array
    y_end : `np.array`
        The end indices of the new array
    '''

    y_start, y_end = np.array(x_start), np.array(x_end)
    # n_points is an array containing the length of the components
    n_points = np.array([x_end[i] - x_start[i] for i in range(len(x_start))])
    if len(x_start) > 2:
        mean_point, std_point = np.mean(n_points), np.std(n_points)
        if std_point > 1.*std_crit:
            g = n_points > 1.*std_crit
            y_start, y_end = y_start[g], y_end[g]
            return y_start, y_end
    return y_start, y_end


def run_make_lc_steps(f_lc, f_orig, min_comp_frac=0.1, outl_mad_fac=3.):
    '''Produce the lightcurves using the cleaning, normalisation and detrending functions
    
    During each procedure, the function keeps a record of datapoints that are either kept
    or rejected, allowing users to assess the amount of data loss.
    
    The function makes the following steps...
    (1) select data points that are within "orig_mad_fac" (MAD) values of the median.
    (2) clean the lightcurve from (1) using clean_lc algorithm.
    (3) normalise the flux by dividing by the median value of the lightcurve produced in (2).
    (4) Include only data from (3) that are within "norm_mad_fac" values of the median.
    (5) detrend the lightcurve produced in (4) following the normalisation choice and the
        appropriate AIC-selected polynomials.
        
    parameters
    ----------
    f_lc : `dict'
        The initial lightcurve with the minimum following keys required:
        (1) 'time' -> the time coordinate
        (2) 'eflux' -> the error in the flux
        (3) 'f_orig' -> see the f_orig parameter
    f_orig : `str'
        This string determines which of the original flux values to choose.
        It forms the final part of the f_lc keys.
        It could be either 'reg_oflux' (the regular, original flux) or
        'cbv_oflux' (the original flux corrected using co-trending basis
        vectors)
    min_comp_frac : `float', optional, default=0.1
        The minimum relative size of a flux component when correcting for
        sparse data in the cleaning functions.
    orig_mad_fac : `float', optional, default=20.
        The factor of MAD values for the initial flux values (1). For inclusion
        in the lightcurve the initial flux values must lie within "orig_mad_fac"
        times the MAD value from the median flux.
    norm_mad_fac : `float', optional, default=2.
        The factor of MAD for the cleaned lightcurve flux values (3).
        
    returns
    -------
    f_lc : `dict'
        A dictionary storing the full set of results from the lightcurve analysis.
        As well as the keys from the inputs, the final keys returned are:
        1: "pass_mad_1" -> datapoints that satisfy (1), boolean
        2: "pass_clean" -> datapoints that qualify from (2), boolean
        3: "nflux" -> the normalised flux, which is calculated by dividing the
           median flux of datapoints that have True "pass_clean" values
        4: "enflux" -> the normalised errors for nflux
        5: "lc_part" -> the indexed lightcurve value calculated during the cleaning.
        6: "pass_mad_2" -> datapoints that qualify from (4), boolean
        7: "nflux_detrend" -> the normalised, detrended flux values.
    '''
    
    # (1) normalise the original flux points
    f_lc['nflux_ori'] = f_lc[f'{f_orig}']/np.median(f_lc[f'{f_orig}'])
    f_lc['nflux_err'] = f_lc['eflux']/f_lc[f'{f_orig}']

    # (2) split the lightcurve into 'time segments'
    ds1, df1 = get_time_segments(f_lc["time"])
    
    # (3) remove very sparse elements from the lightcurve
    comp_lengths = np.array([f-s for s, f in zip(ds1, df1)])
    std_crit_val = int(np.sum(comp_lengths)*min_comp_frac)
    ds2, df2 = remove_sparse_data(ds1, df1, std_crit=std_crit_val)
    f_lc["pass_sparse"] = np.array(np.zeros(len(f_lc["time"])), dtype='bool')
    for s, f in zip(ds2, df2):
        f_lc["pass_sparse"][s:f] = True

    # (4) run the first detrending process to pass to the cleaning function.
    f_lc["lc_part"] = np.zeros(len(f_lc["time"]), dtype=int)
    for i, (s, f) in enumerate(zip(ds2, df2)):
        f_lc["lc_part"][s:f] = int(i+1)
    g_cln = f_lc["pass_sparse"]
    f_lc["nflux_dtr"] = np.full(len(f_lc["time"]), -999.)
    f_lc["nflux_dtr"][g_cln], detr_dict = detrend_lc(f_lc["time"][g_cln], f_lc["nflux_ori"][g_cln], f_lc["lc_part"][g_cln], poly_max=1)

    # (5) clean the lightcurve using clean_lc algorithm
    ds3, df3 = [], []
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        s_1, f_1 = clean_edges_outlier(f_lc["nflux_dtr"][g])
        g_time = g[s_1:f_1]
        s_2, f_2 = clean_edges_scatter(f_lc["nflux_dtr"][g_time])
        ds3.append(g[s_2])
        df3.append(g[f_2])
    f_lc["pass_clean"] = np.array(np.zeros(len(f_lc["time"])), dtype='bool')
    for s, f in zip(ds3, df3):
        f_lc["pass_clean"][s:f] = True
    
    # (6) detrend the original lightcurve, but only using the data that passed the
    # the previous criteria
#    g_cln = f_lc["pass_clean"]
#    f_lc["nflux_dtr2"] = np.full(len(f_lc["time"]), -999.)
#    f_lc["nflux_dtr2"][g_cln], detr_dict = detrend_lc(f_lc["time"][g_cln], f_lc["nflux_ori"][g_cln], f_lc["lc_part"][g_cln], poly_max=1)

    # (7) finally cut out data that are extreme outliers.
    med_lc = np.median(f_lc["nflux_dtr"][g_cln])
    MAD_lc = MAD(f_lc["nflux_dtr"][g_cln], scale='normal')
    f_lc["pass_outlier"] = np.array(np.zeros(len(f_lc["time"])), dtype='bool')
    for f in range(len(f_lc["time"])):
        if abs(f_lc["nflux_dtr"][f] - med_lc) < outl_mad_fac*MAD_lc:
            f_lc["pass_outlier"][f] = True

    # (8) return the dictionary
    return f_lc, detr_dict


def sin_fit(x, y0, A, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase) to a regular
    sinusoidal function.

    parameters
    ----------
    x : `Iterable`
        list of input values
    y0 : `float`
        The midpoint of the sine curve
    A : `float`
        The amplitude of the sine curve
    phi : `float`
        The phase angle of the sine curve

    returns
    -------
    sin_fit : `list`
        A list of sin curve values.
    '''
    sin_fit = y0 + A*np.sin(2.*np.pi*x + phi)
    return sin_fit


def test_cbv_fit(t, of, cf):
    '''Run a score test to determine whether the cbv-corrected lightcurve should be considered.
    
    Whilst the cbv-corrected flux are designed to eliminate systematic artefacts by identifying
    features common to many stars (using PCA), the routine can overfit the data, and often the
    cbv corrections inject too much unwanted noise (particularly for targets with low signal
    to noise).
    
    Therefore the plan here is to assess lightcurves produced by the cbv corrections
    by comparing basic attributes with the non-corrected lightcurves. These scores come down
    to the number of outliers (test 1), size of the median absolute deviation (test 2), and 
    which lightcurve provides the lowest chi-squared value to a sinusoidal fit (test 3).
    
    If the "original lightcurve" scores higher, then the cbv-corrected lightcurve is not
    considered for further analysis.
    
    parameters
    ----------
    t : `Iterable'
        The time component of the lightcurve
    of : `Iterable'
        The original flux
    cf : `Iterable'
        The cbv-corrected flux
    
    returns
    -------
    use_cbv : `bool'
        True if cf score > of score, else False.
    '''
    
    of_score, cf_score = 0, 0

#1) number of outliers test
    of_nflux, cf_nflux = np.array(of)/np.median(of), np.array(cf)/np.median(cf)
    of_nMADf, cf_nMADf = MAD(of_nflux, scale='normal'), MAD(cf_nflux, scale='normal')
    num_of, num_cf = np.sum(abs(of_nflux-1.) > of_nMADf), np.sum(abs(cf_nflux-1.) > cf_nMADf)
    if num_of > num_cf:
        cf_score += 1
    else:
        of_score += 1
#2) which has the largest MAD value
    if of_nMADf > cf_nMADf:
        cf_score += 1
    else:
        of_score += 1
#3) which makes the best sine fit?
    try:
        pops_of, popsc_of = curve_fit(sin_fit, t, of_nflux,
                                bounds=(0, [2., 2., 1000.]))
        pops_cf, popsc_cf = curve_fit(sin_fit, t, cf_nflux,
                                bounds=(0, [2., 2., 1000.]))
        yp_of = sin_fit(of_nflux, *pops_of)
        yp_cf = sin_fit(cf_nflux, *pops_cf)
        chi_of = np.sum((yp_of-of_nflux)**2)/(len(of_nflux)-len(pops_of)-1)
        chi_cf = np.sum((yp_cf-cf_nflux)**2)/(len(cf_nflux)-len(pops_cf)-1)
        if chi_of > chi_cf:
            cf_score += 1
        else:
            of_score += 1
    except:
        logger.error('Could not do the sine-fit comparison for ori vs cbv lightcurves')
# get the final score - if cbv wins, then a True statement is returned.    
    if of_score >= cf_score:
        use_cbv = False
    else:
        use_cbv = True
    return use_cbv


