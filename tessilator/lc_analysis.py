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
__all__ = ['aic_selector', 'aper_run', 'clean_flux_edges', 'detrend_lc',
           'get_time_segments', 'get_xy_pos', 'logger', 'make_lc',
           'normalisation_choice', 'remove_sparse_data', 'sin_fit', 'test_cbv_fit']


import warnings

# Third party imports
import numpy as np
import os

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
import itertools as it
from operator import itemgetter

# Local application imports
from .fixedconstants import *


# initialize the logger object
logger = logging.getLogger(__name__)
#logger_aq = logging.getLogger("astroquery")
#logger_aq.setLevel(logging.ERROR)    

def aic_selector(x, y, poly_max=3):
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
    while q < poly_max:
        k1, k2 = q+1, q+2
        p1, r1, _,_,_ = np.polyfit(x, y, q, full=True)
        p2, r2, _,_,_ = np.polyfit(x, y, q+1, full=True)
        with np.errstate(invalid='ignore'):
            SSR1 = np.sum((np.polyval(p1, x) - y)**2)
            SSR2 = np.sum((np.polyval(p2, x) - y)**2)
        AIC1 = akaike_info_criterion_lsq(SSR1, k1, N)
        AIC2 = akaike_info_criterion_lsq(SSR2, k2, N)
        
        if AIC1 < (AIC2 + 2):
            poly_ord, coeffs = q, p1
            return poly_ord, coeffs
        else:
            q += 1
            if q == poly_max-1:
                poly_ord, coeffs = q+1, p2
                return poly_ord, coeffs



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

    full_phot_table = Table(names=('id', 'xcenter', 'ycenter', 'flux',
                                   'flux_err', 'bkg', 'total_bkg',
                                   'reg_oflux', 'mag', 'mag_err', 'time'),
                            dtype=(str, float, float, float, float, float,
                                   float, float, float, float, float))
    for f_num, f_file in enumerate(fits_files):
        print(f_num, f_file)
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
                        fix_cols = ['id', 'xcenter', 'ycenter',
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
    
    
def clean_flux_edges(f, MAD_fac=.1):
    '''Remove data points from the lightcurve that are likely to be spurious.

    Many lightcurves have a 1 or 2 day gap. To avoid systematic offsets and
    ensure the data is efficiently normalized, the lightcurve is split into
    "strings" of contiguous data. Neighbouring data points must have been
    observed within "time_fac" times the median absolute deviation of the
    time difference between each observation.
    
    The start and end point of each data section must have a flux value within
    a given number of MAD from the median flux in the sector. This is done
    because often after large time gaps the temperature of the sensors changes,
    and including these components is likely to just result in a signal from
    instrumental noise.
    
    The function returns the start and end points for each data section in the
    sector, which must contain a chosen number of data points. This is to
    ensure there are enough datapoints to construct a periodogram analysis.

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.

    returns
    -------
    s : `int`
       The start index for the data string.
    f : `int`
       The end index for the data string.
    '''
    
    # get the median time and flux, the median absolute deviation in flux
    # and the time difference for each neighbouring point.
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    g = (np.abs(f-f_med) <= MAD_fac*f_MAD).astype(int)
    i=0
    while i < len(f)-1:
        if g[i] != 1:
            i+=1
        else:
            s=i
            break
    i=len(f)-1
    while i > 0:
        if g[i] != 1:
            i-=1
        else:
            f=i
            break
    return s, f


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
    s_fit, coeffs = aic_selector(t, f, poly_max=poly_max)
#    print(f'step 1: {s_fit}, {coeffs}')
    f_norm = f/np.polyval(coeffs, t)
    # 2. Decide whether to use the detrended lightcurve from part 1, or to separate the
    #    lightcurve into individual components and detrend each one separately
    norm_comp = normalisation_choice(t, f, lc, MAD_fac=MAD_fac, poly_max=poly_max)
#    print(f'do we normalise separately? {norm_comp}')
    s_fit, coeffs = None, None
    # 3. Detrend the lightcurve following steps 1 and 2.
    if norm_comp:
        # normalise each component separately.
        f_detrend = np.array([])
        for l in np.unique(lc):
            g = np.array(lc == l)
            s_fit, coeffs = aic_selector(t[g], f[g], poly_max=poly_max)
#            print(f'final_fits: {s_fit}, {coeffs}') 
            f_n = f[g]/np.polyval(coeffs, t[g])
            f_detrend = np.append(f_detrend, f_n)
        f_norm = f_detrend
    else:
        # normalise the entire lightcurve as a whole
        f_norm = f_norm
    return f_norm


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
    df = [(i+1) for i in df]
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
    
    
def make_lc(phot_table, name_lc, store_lc=False, lc_dir='lc'):
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
    name_lc : `str`
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
        f_out = run_make_lc_steps(final_lc, f_label)
        if len(f_out["time"]) > 50: 
            tab_out = Table(f_out)
            if f_label == "reg_oflux":
                final_tabs.append(tab_out)
            if (f_label == "cbv_oflux") and (cbv_ret):
                final_tabs.append(tab_out)
            if store_lc:
                path_exist = os.path.exists(f'./{lc_dir}')
                if not path_exist:
                    os.makedirs(f'./{lc_dir}')
                tab_out.write(f'./{lc_dir}/{name_lc}_{f_label}.csv', format='csv', overwrite=True)
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
#    print(np.unique(lc_part))
    norm_comp = False
    Ncomp = len(np.unique(lc_part))
    if Ncomp > 1:
        i = 1
        while i < Ncomp:
            g1 = np.array(lc_part == i)
            g2 = np.array(lc_part == i+1)
            
            s_fit1, coeff1 = aic_selector(t_orig[g1], f_orig[g1], poly_max=poly_max)
            s_fit2, coeff2 = aic_selector(t_orig[g2], f_orig[g2], poly_max=poly_max)

#            print(f'lc {i}: {s_fit1}, {coeff1}')
#            print(f'lc {i+1}: {s_fit2}, {coeff2}')
            
            f1_at_f2_0 = np.polyval(coeff1, t_orig[g2][0]) # yes, the index IS supposed to be [g2]
            f2_at_f2_0 = np.polyval(coeff2, t_orig[g2][0])
#            print(f'matching points: {f1_at_f2_0}, {f2_at_f2_0}')
#            print(f'time at f2_0: {t_orig[g2][0]}')
            f1_n = f_orig[g1]/np.polyval(coeff1, t_orig[g1])
            f2_n = f_orig[g2]/np.polyval(coeff2, t_orig[g2])
            f1_MAD = MAD(f1_n, scale='normal')
            f2_MAD = MAD(f2_n, scale='normal')
#            if abs(f1_at_f2_0 - f2_at_f2_0) > MAD_fac*((f1_MAD+f2_MAD)/2.) and \
#               max(np.sum(g1), np.sum(g2))/min(np.sum(g1), np.sum(g2)) < 3.0:
            if abs(f1_at_f2_0 - f2_at_f2_0) > MAD_fac*((f1_MAD+f2_MAD)/2.):
                norm_comp = True
                return norm_comp
            else: i += 1
    return norm_comp


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


def run_make_lc_steps(f_lc, f_orig, min_comp_frac=0.1, orig_mad_fac=20., norm_mad_fac=2.):
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
    f_lc["nflux_dt1"] = np.full(len(f_lc["time"]), -999.)
    f_lc["nflux_dt1"][g_cln] = detrend_lc(f_lc["time"][g_cln], f_lc["nflux_ori"][g_cln], f_lc["lc_part"][g_cln], poly_max=5)
    # (5) clean the lightcurve using clean_lc algorithm
    ds3, df3 = [], []
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        s, f = clean_flux_edges(f_lc["nflux_dt1"][g])
        ds3.append(g[s])
        df3.append(g[f])
    f_lc["pass_clean"] = np.array(np.zeros(len(f_lc["time"])), dtype='bool')
    for s, f in zip(ds3, df3):
        f_lc["pass_clean"][s:f] = True
    
    # (6) detrend the original lightcurve, but only using the data that passed the
    # the previous criteria
    g_cln = f_lc["pass_clean"]
    f_lc["nflux_dt2"] = np.full(len(f_lc["time"]), -999.)
    f_lc["nflux_dt2"][g_cln] = detrend_lc(f_lc["time"][g_cln], f_lc["nflux_ori"][g_cln], f_lc["lc_part"][g_cln], poly_max=5)

    # (7) return the dictionary
    return f_lc


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
# get the final score - if cbv wins, then a True statement is returned.    
    if of_score >= cf_score:
        use_cbv = False
    else:
        use_cbv = True
    return use_cbv
