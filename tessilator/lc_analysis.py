'''

Alexander Binks & Moritz Guenther, January 2023

Licence: MIT 2023

This module contains functions to perform aperture photmetry and clean lightcurves. These are:

1)  aper_run - returns a numpy array containing the times, magnitudes and
    fluxes from aperture photometry.
    
2)  clean_lc/detrend_lc/make_lc - these 3 functions are made to remove spurious data
    points, ensure that only "strings" of contiguous data are being processed, and
    each string is detrended using a series of decision processes.
    Finally, the lightcurve is pieced together and make_lc returns a table containing
    the data ready for periodogram analysis.
    
3)  run_LS - function to conduct Lomb-Scargle periodogram analysis. Returns a table
    with period measurements, plus several data quality flags. If required a plot of
    the lightcurve, periodogram and phase-folded lightcurve is provided.

'''

# imports
import logging
__all__ = ['logger', 'get_xy_pos', 'aper_run', 'AIC_selector', 'clean_lc',
           'remove_sparse_data', 'mean_of_arrays', 'moving_average', 'check_for_jumps',
           'normalisation_choice', 'detrend_lc', 'make_lc', 'get_second_peak', 
           'gauss_fit', 'sin_fit', 'gauss_fit_peak', 'run_ls', 'is_period_cont']


import warnings

# Third party imports
import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry, ApertureStats
from astroquery.gaia import Gaia
from astroquery.mast import Tesscut
from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit
import itertools as it


from collections.abc import Iterable

# Local application imports
from .tess_stars2px import tess_stars2px_function_entry
from .fixedconstants import *


# initialize the logger object
logger = logging.getLogger(__name__)
logger_aq = logging.getLogger("astroquery")
logger_aq.setLevel(logging.ERROR)    
    



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
                                   'flux_corr', 'mag', 'mag_err', 'time'),
                            dtype=(str, float, float, float, float, float,
                                   float, float, float, float, float))
    for f_num, f_file in enumerate(fits_files):
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


def AIC_selector(x, y, err, poly_max=10):
    '''Chooses the most appropriate polynomial fit, using the Aikaike Information Criterion
    
    This function uses the Aikaike Information Criterion to find the most appropriate polynomial order to a set of X, Y data points.
    
    parameters
    ----------
    x : `Iterable`
        The x-component of the data
    y : `Iterable`
        The y-component of the data
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
    while q < poly_max:
        p1, r1, _,_,_ = np.polyfit(x, y, q, full=True)
        p2, r2, _,_,_ = np.polyfit(x, y, q+1, full=True)
        chi1 = np.sum(((np.polyval(p1, x) - y)**2)/(err)**2)
        chi2 = np.sum(((np.polyval(p2, x) - y)**2)/(err)**2)

        # AIC = 2(k - ln(L)) + term for AICc
        # -ln(L) = chi-squared/2
        # AIC = 2(k + chi-squared/2)
        # model 1 has q+1 free parameters (constant and q terms)
        AIC1 = 2.*(q+1) + chi1 + (2.*(q+1)*(q+2)/(len(x)-(q+1)-1))
        AIC2 = 2.*(q+2) + chi2 + (2.*(q+2)*(q+3)/(len(x)-(q+2)-1))
        if AIC1 < AIC2:
            poly_ord, coeffs = q, p1
            return poly_ord, coeffs
        else:
            q += 1
            if q == poly_max:
                poly_ord, coeffs = q+1, p2
                return q+1, p2


def clean_lc(t, f, err, MAD_fac=2.0, time_fac=10., min_num_per_group=50):
    '''Remove data points from the lightcurve that are likely to be spurious.

    Many lightcurves have a 1 or 2 day gap. To avoid systematic offsets and
    ensure the data is efficiently normalized, the lightcurve is split into
    "strings" of contiguous data. Neighbouring data points must have been
    observed within 10 times the median absolute deviation of the time
    difference between each observation.
    
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
    t : `Iterable`
        The set of time coordinates (in days)
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow. 
    time_fac : `float`, optional, default=10.
        The maximum time gap allowed between neighbouring datapoints.
    min_num_per_group : `int`, optional, default=50
        The minimum number of datapoints allowed in a contiguous set.

    returns
    -------
    start_index : `list`
       The start indices for each saved data string.
    end_index : `list`
       The end indices for each saved data string.
    '''

    s_fit, coeffs = AIC_selector(t, f, err, poly_max=8)
    f = f/np.polyval(coeffs, t)
    
    tm, fm = np.median(t), np.median(f)
    f_MAD  = MAD(f, scale='normal')
    td     = np.zeros(len(t))
    td[1:] = np.diff(t)

    A0 = (np.abs(f-fm) <= MAD_fac*f_MAD).astype(int)
    A1 = (td <= time_fac*np.median(td)).astype(int)
    B  = (A0+A1).astype(int)
    gs, gf = [], []
    i = 0    
    l = 0
    while i < len(f)-1:
        if B[i] == 2:
            gs.append(i)
            j = i+1
            while (A1[j]) == 1 and (j < len(f)-1):
                j = j+1
            if j == len(f)-1:
                l = l+1
                break
            else:
                k = j
                j = j-1
                while A0[j] == 0:
                    j = j-1
                gf.append(j)
                i = k + 1
        else:
            i = i+1
    if l == 1:
        k = len(f)-1
        while B[k] != 2:
            k = k-1
        gf.append(k)
    
    gs, gf = np.array(gs), np.array(gf)
    start_index = gs[(gf-gs)>min_num_per_group]
    end_index = gf[(gf-gs)>min_num_per_group]
    return start_index, end_index




def remove_sparse_data(x_start, x_end, std_crit=100):
    '''Removes very sparse bits of data from the lightcurve when there are 3 or more components.
    
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
    n_points = np.array([x_end[i]+1 - x_start[i] for i in range(len(x_start))])
    if len(x_start) > 2:
        mean_point, std_point = np.mean(n_points), np.std(n_points)
        if std_point > std_crit:
            g = n_points > mean_point - std_point
            y_start, y_end = x_start[g], x_end[g]
            return y_start, y_end
    return y_start, y_end


def mean_of_arrays(arr, num):
    '''Calculate the mean and standard deviation of an array which is split into N components.
    
    parameters
    ----------
    arr : `Iterable`
        The input array
    num : `int`
        The number of arrays to split the data (equally) into

    returns
    -------
    mean_out : `float`
        The mean of the list of arrays.
    std_out : `float`
        The standard deviation of the list of arrays.
    '''
    x = np.array_split(arr, num)
    ar = np.array(list(it.zip_longest(*x, fillvalue=np.nan)))
    return np.nanmean(ar, axis=0), np.nanstd(ar, axis=0)
    

def moving_average(x, w):
    '''Calculate the moving average of an array.
    
    parameters
    ----------
    x : `Iterable`
        The input data to be analysed.
    w : `int`
        The number of data points that the moving average will convolve.

    returns
    -------
    z : `np.array`
        An array of the moving averages
    '''
    
    z = np.convolve(x, np.ones(w), 'valid') / w
    return z
    


def check_for_jumps(time, flux, eflux, lc_part, n_avg=10, thresh_diff=10.0):
    '''Identify if the lightcurve has jumps.
    
    A jumpy lightcurve is one that has small contiguous data points that change in flux significantly compared to the amplitude of the lightcurve. These could be due to some instrumental noise or response to a non-astrophysical effect. They may also be indicative of a stellar flare or active event.
    
    This function takes a running average of the differences in flux, and flags lightcurves if the absolute value exceeds a threshold. These will be flagged as "jumpy" lightcurves.

    parameters
    ----------
    time : `Iterable`
        The time coordinate
    flux : `Iterable`
        The original, normalised flux values
    eflux : `Iterable`
        The error on "flux"
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    n_avg : `int`, optional, default=10
        The number of data points to calculate the running average
    thresh_diff : `float`, optional, default=10.0
        The threshold value, which, if exceeded, will yield a "jumpy" lightcurve

    returns
    -------
    jump_flag : `Boolean`
        This will be True if a jumpy lightcurve is identified, otherwise False.
    '''
    
    jump_flag = False
    
    for i in range(len(np.unique(lc_part))):
        g = np.array(lc_part == i)
        f_mean = moving_average(flux[g], n_avg)
        t_mean = moving_average(time[g], n_avg)
        
        f_shifts = np.abs(np.diff(f_mean))

        median_f_shifts = np.median(f_shifts)
        max_f_shifts = np.max(f_shifts)
        if max_f_shifts/median_f_shifts > thresh_diff:
            jump_flag = True
            return jump_flag

    return jump_flag


def normalisation_choice(t_orig, f_orig, e_orig, lc_part, MAD_fac=2.0):
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
    MAD_fac : `float`, optional, default = 2.0
        The factor to multiply the median absolute deviation by.    

    returns
    -------
    norm_comp : `Boolean`
        Determines whether the data should be detrended as one whole component (False) or in parts (True)
    '''

    norm_comp = False
    Ncomp = len(np.unique(lc_part))
    if Ncomp > 1:
        i = 0
        while i < Ncomp-1:
            g1 = np.array(lc_part == i)
            g2 = np.array(lc_part == i+1)
            s_fit1, coeff1 = AIC_selector(t_orig[g1], f_orig[g1], e_orig[g1], poly_max=1)
            s_fit2, coeff2 = AIC_selector(t_orig[g2], f_orig[g2], e_orig[g2], poly_max=1)

            f1_at_f2_0 = np.polyval(coeff1, t_orig[g2][0])
            f2_at_f2_0 = np.polyval(coeff2, t_orig[g2][0])
            f1_n = f_orig[g1]/np.polyval(coeff1, t_orig[g1])
            f2_n = f_orig[g2]/np.polyval(coeff2, t_orig[g2])
            f1_MAD = MAD(f1_n, scale='normal')
            f2_MAD = MAD(f2_n, scale='normal')
            if abs(f1_at_f2_0 - f2_at_f2_0) > MAD_fac*((f1_MAD+f2_MAD)/2.) and \
               max(len(g1), len(g2))/min(len(g1), len(g2)) < 3.0:
                norm_comp = True
                return norm_comp
            else: i += 1
    return norm_comp


def detrend_lc(ds,df,t,m,f,err, MAD_fac=2.0, poly_max=8):
    '''Detrend and normalise the lightcurves.

    | This function runs several operations to detrend the lightcurve, as follows:
    | 1. Remove any sparse components of the lightcurve that will not benefit the lightcurve analysis.
    | 2. Remove data points that are clearly outliers.
    | 3. Choose the best detrending polynomial using the Aikaike Information Criterion, and detrend the full lightcurve.
    | 4. Decide whether to use the detrended lightcurve from part 3, or to separate the lightcurve into individual components and detrend each one separately.
    | 5. Place the results into a dictionary.

    parameters
    ----------
    ds : `Iterable`
        the start indices for the lightcurve.
    df : `Iterable`
        the end indices for the lightcurve.
    t : `Iterable`
        The list of time coordinates.
    m : `Iterable`
        The list of magnitude coordinates.
    f : `Iterable`
        The list of flux coordinates.
    err : `Iterable`
        The list of flux error coordinates.
    MAD_fac : `float`, optional, default = 2.0
        The factor to multiply the median absolute deviation by.
    poly_max : `int`, optional, default=8
        The maximum order of the polynomial fit.

    returns
    -------
    dict_lc : `dict`
        | A dictionary containing the following keys:
        | "time" -> The time coordinate
        | "mag" -> The magnitude coordinate
        | "oflux" -> The original, normalised flux values
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The error on "nflux"
        | "polyord" -> The polynomial order used for each detrend        
    '''

    dict_lc = {"time":[], "mag":[], "oflux":[], "nflux":[], "enflux":[], "lc_part":[]}
    t_orig, f_orig, e_orig = np.array([]), np.array([]), np.array([])
    m_orig, lc_part = np.array([]), np.array([])


    # 1. Remove sparse data
    std_crit_val = int(np.sum(np.array([df[i]+1 - ds[i] for i in range(len(ds))]))/10.)
    ds, df = remove_sparse_data(ds, df, std_crit=std_crit_val)
        
    # 2. Include only data that are within a given number of median absolute deviation values.
    for i in range(len(ds)):
        # determine the start and end points for each data string
        rs, rf = ds[i], df[i]+1

        fm = np.median(f[rs:rf])
        f_MAD  = MAD(f[rs:rf], scale='normal')
        g = np.abs(f[rs:rf]-fm) <= MAD_fac*f_MAD
        
        t_orig = np.append(t_orig, t[rs:rf][g])
        m_orig = np.append(m_orig, m[rs:rf][g])
        f_orig = np.append(f_orig, f[rs:rf][g])
        e_orig = np.append(e_orig, err[rs:rf][g])
        lc_part = np.append(lc_part, np.full(np.sum(g), i))

    # 3. Choose the best detrending polynomial using the Aikaike Information Criterion, and
    #    detrend the lightcurve
    s_fit, coeffs = AIC_selector(t_orig, f_orig, e_orig, poly_max=8)
    f_norm = f_orig/np.polyval(coeffs, t_orig)

    # 4. Decide whether to use the detrended lightcurve from part 3, or to separate the
    #    lightcurve into individual components and detrend each one separately
    norm_comp = normalisation_choice(t_orig, f_orig, e_orig, lc_part, MAD_fac=2.0)

    if norm_comp:
        f_detrend = np.array([])
        for i in range(len(ds)):
            g = np.array(lc_part == i)
            s_fit, coeffs = AIC_selector(t_orig[g], f_orig[g], e_orig[g], poly_max=8)
            f_n = f_orig[g]/np.polyval(coeffs, t_orig[g])
            f_detrend = np.append(f_detrend, f_n)
        f_norm = f_detrend
    else:
        f_norm = f_norm

    # 5. Place the results into a dictionary.
    for i in range(len(t_orig)):
        dict_lc["time"].append(t_orig[i])
        dict_lc["mag"].append(m_orig[i])
        dict_lc["oflux"].append(f_orig[i])
        dict_lc["nflux"].append(f_norm[i])
        dict_lc["enflux"].append(e_orig[i])
        dict_lc["lc_part"].append(lc_part[i])

    return dict_lc
    
    

def make_lc(phot_table):
    '''Construct the normalised TESS lightcurve.

    | The function runs the following tasks:
    | (1) Read the table produced from the aperture photometry
    | (2) Normalise the lightcurve using the median flux value.
    | (3) Clean the lightcurve from spurious data using "clean_lc"
    | (4) Detrend the lightcurve using "detrend_lc"
    | (5) Return the original normalised lightcurve and the cleaned lightcurve

    parameters
    ----------
    phot_table : `astropy.table.Table` or `dict`
        | The data table containing aperture photometry returned by aper_run.py. Columns must include:
        | "time" -> The time coordinate for each image
        | "mag" -> The target magnitude
        | "flux_corr" -> The total flux subtracted by the background flux
        | "flux_err" -> The error on flux_corr
        

    returns
    -------
    cln : `dict`
        | The cleaned, detrended, normalised lightcurve, with the keys:
        | "time" -> The time coordinate
        | "time0" -> The time coordinate relative to the first data point
        | "oflux" -> The original, normalised flux values
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The error on "nflux"
        | "lc_part" -> The running index for each contiguous data section in the lightcurve

    orig : `dict`
        | The original, normalised lightcurve, with the keys:
        | "time" -> The time coordinate
        | "nflux" -> The original, normalised flux values
        | "mag" -> The TESS magnitude values
    '''
    
    m_diff = phot_table["flux_corr"][:] - np.median(phot_table["flux_corr"][:])
    m_thr = 20.0*MAD(phot_table["flux_corr"][:], scale='normal')
    g = np.abs(m_diff < m_thr)
    time = np.array(phot_table["time"][:][g])
    mag = np.array(phot_table["mag"][:][g])
    flux = np.array(phot_table["flux_corr"][:][g])
    eflux = np.array(phot_table["flux_err"][:][g])

    ds, df = clean_lc(time, flux, eflux)

    if (len(ds) == 0) or (len(df) == 0):
        return [], []
    # 1st: normalise the flux by dividing by the median value
    nflux  = flux/np.median(flux)
    neflux = eflux/flux
    # 2nd: detrend each lightcurve sections by either a straight-line fit or a
    # parabola. The choice is selected using AIC.
    cln = detrend_lc(ds, df, time, mag, nflux, neflux)
    cln["time0"] = [cln["time"][i]-time[0] for i in range(len(cln["time"]))]
    orig = dict()
    orig["time"] = np.array(time)
    orig["nflux"] = np.array(nflux)
    orig["mag"] = np.array(mag)
    if len(cln["time"]) > 50:
        return cln, orig
    else:
        return [], []



def get_second_peak(power):
    '''An algorithm to identify the second-highest peak in the periodogram

    parameters
    ----------
    power : `Iterable`
        A set of power values calculated from the periodogram analysis.

    returns
    -------
    a_g : `list`
        A list of indices corresponding to the Gaussian around the peak power.
    a_o : `list`
        A list of indices corresponding to all other parts of the periodogram.
    '''
    # Get the left side of the peak
    a = np.arange(len(power))

    p_m = np.argmax(power)
    x = p_m
    while (power[x-1] < power[x]) and (x > 0):
        x = x-1
    p_l = x
    p_lx = 0
    while (power[p_l] > 0.85*power[p_m]) and (p_l > 1):
        p_lx = 1
        p_l = p_l - 1
    if p_lx == 1:
        while (power[p_l] > power[p_l-1]) and (p_l > 0):
            p_l = p_l - 1
    if p_l < 0:
        p_l = 0

    # Get the right side of the peak
    x = p_m
    if x < len(power)-1:
        while (power[x+1] < power[x]) and (x < len(power)-2):
            x = x+1
        p_r = x
        p_rx = 0
        while (power[p_r] > 0.85*power[p_m]) and (p_r < len(power)-2):
            p_rx = 1
            p_r = p_r + 1
        if p_rx == 1:
           while (power[p_r] > power[p_r+1]) and (p_r < len(power)-2):
                p_r = p_r + 1
        if p_r > len(power)-1:
            p_r = len(power)-1
        a_g = a[p_l:p_r+1]
        a_o = a[np.setdiff1d(np.arange(a.shape[0]), a_g)] 
    elif x == len(power)-1:
        a_g = a[x]
        a_o = a[0:x]
    return a_g, a_o


def gauss_fit(x, a0, x_mean, sigma):
    '''Construct a simple Gaussian.

    Return Gaussian values from a given amplitude (a0), mean (x_mean) and
    uncertainty (sigma) for a distribution of values

    parameters
    ----------
    x : `Iterable`
        list of input values
    a0 : `float`
        Amplitude of a Gaussian
    x_mean : `float`
        The mean value of a Gaussian
    sigma : `float`
        The Gaussian uncertainty

    returns
    -------
    gaussian : `list`
        A list of Gaussian values.
    '''

    gaussian = a0*np.exp(-(x-x_mean)**2/(2*sigma**2))
    return gaussian

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


def gauss_fit_peak(period, power):
    '''
    Applies the Gaussian fit to the periodogram. If there are more than 3 data
    points (i.e., more data points than fixed parameters), the "gauss_fit"
    module is used to return the fit parameters. If there are 3 or less points,
    the maximum peak is located and 9 data points are interpolated between the
    2 neighbouring data points of the maximum peak, and the "gauss_fit" module
    is applied.
    
    parameters
    ----------
    period : `Iterable`
        The period values around the peak.
    power : `Iterable`
        The power values around the peak.
        
    returns
    -------
    popt : `list`
        The best-fit Gaussian parameters: A, B and C where A is the amplitude,
        B is the mean and C is the uncertainty.
    '''
    if len(period) > 3:
        popt, _ = curve_fit(gauss_fit, period, power,
                            bounds=([0, period[0], 0],
                                    [1., period[-1], period[-1]-period[0]]))
        ym = gauss_fit(period, *popt)
    else:
        p_m = np.argmax(power)
        peak_vals = [p_m-1, p_m, p_m+1]
        x = period[peak_vals]
        y = power[peak_vals]
        xvals = np.linspace(x[0], x[-1], 9)
        yvals = np.interp(xvals, x, y)
        popt, _ = curve_fit(gauss_fit, xvals, yvals,
                            bounds=(0, [1., np.inf, np.inf]))
        ym = gauss_fit(xvals, *popt)        
    return popt, ym

 
 

def run_ls(cln, n_sca=10, p_min_thresh=0.05, p_max_thresh=100., samples_per_peak=10):
    '''Run Lomb-Scargle periodogram and return a dictionary of results.

    parameters
    ----------
    cln : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time0" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous sections
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : `int`, optional, default=10
        The number of samples to measure in each periodogram peak.

    returns
    -------
    LS_dict : `dict`
        A dictionary of parameters calculated from the periodogram analysis. These are:
        | "median_MAD_nLC" : The median and median absolute deviation of the normalised lightcurve.
        | "jump_flag" : A flag determining if the lightcurve has sharp jumps in flux.
        | "period" : A list of period values from the periodogram analysis.
        | "power" :  A list of power values from the periodogram analysis.
        | "period_best" : The period corrseponding to the highest power output.
        | "power_best" : The highest power output.
        | "time" : The time coordinate corresponding to the normalised lightcurve.
        | "y_fit_LS" : The best fit sinusoidal function.
        | "AIC_sine" : The Aikaike Information Criterion value of the best-fit sinusoid
        | "AIC_line" : The Aikaike Information Criterion value of the best-fit linear function.
        | "FAPs" : The power output for the false alarm probability values of 0.1, 1 and 10%
        | "Gauss_fit_peak_parameters" : Parameters for the Gaussian fit to the highest power peak
        | "Gauss_fit_peak_y_values" : The corresponding y-values for the Gaussian fit
        | "period_around_peak" : The period values covered by the Gaussian fit
        | "power_around_peak" : The power values across the period range covered by the Gaussian fit
        | "period_not_peak" : The period values not covered by the Gaussian fit
        | "power_not_peak" : The power values across the period range not covered by the Gaussian fit
        | "period_second" : The period of the second highest peak.
        | "power_second" : The power of the second highest peak.
        | "phase_fit_x" : The time co-ordinates from the best-fit sinusoid to the phase-folded lightcurve.
        | "phase_fit_y" : The normalised flux co-ordinates from the best-fit sinusoid to the phase-folded lightcurve.
        | "phase_x" : The time co-ordinates from the phase-folded lightcurve.
        | "phase_y" : The normalised flux co-ordinates from the phase-folded lightcurve.
        | "phase_chisq" : The chi-square fit between the phase-folded lightcurve and the sinusoidal fit.
        | "phase_col" : The cycle number for each data point.
        | "pops_vals" : The best-fit parameters from the sinusoidal fit to the phase-folded lightcurve.
        | "pops_cov" : The corresponding co-variance matrix from the "pops_val" parameters.
        | "phase_scatter" : The typical scatter in flux around the best-fit.
        | "frac_phase_outliers" : The fraction of data points that are more than 3 median absolute deviation values from the best-fit.
        | "Ndata" : The number of data points used in the periodogram analysis.
    '''
    LS_dict = dict()
    
    time = np.array(cln["time0"])
    nflux = np.array(cln["nflux"])
    enflux = np.array(cln["enflux"])
    lc_part = np.array(cln["lc_part"])
    
    jump_flag = check_for_jumps(time, nflux, enflux, lc_part)
#    jump_flag = 0
    med_f, MAD_f = np.median(nflux), MAD(nflux, scale='normal')
    ls = LombScargle(time, nflux, dy=enflux)
    frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                    maximum_frequency=1./p_min_thresh,
                                    samples_per_peak=samples_per_peak)
    FAP = ls.false_alarm_probability(power.max())
    probabilities = [0.1, 0.05, 0.01]
    FAP_test = ls.false_alarm_level(probabilities)
    p_m = np.argmax(power)
    
    y_fit_sine = ls.model(time, frequency[p_m])
    y_fit_sine_param = ls.model_parameters(frequency[p_m])
    chisq_model_sine = np.sum((y_fit_sine-nflux)**2/enflux**2)/(len(nflux)-3-1)
    line_fit, _,_,_,_ = np.polyfit(time, nflux, 1, full=True)
    y_fit_line = np.polyval(line_fit, time)
    chisq_model_line = np.sum((y_fit_line-nflux)**2/enflux**2)/(len(nflux)-len(line_fit)-1)

    AIC_sine, AIC_line = 2.*3. + chisq_model_sine, 2.*2. + chisq_model_line

    print(f'(chisq, AIC) sine = {chisq_model_sine}, {AIC_sine}')
    print(f'(chisq, AIC) line = {chisq_model_line}, {AIC_line}')
    period_best = 1.0/frequency[p_m]
    power_best = power[p_m]
    period = 1./frequency[::-1]
    power = power[::-1]
    # a_g: array of datapoints that form the Gaussian around the highest power
    # a_o: the array for all other datapoints
    a_g, a_o = get_second_peak(power)

    if isinstance(a_g, Iterable):
        pow_r = max(power[a_g])-min(power[a_g])
        a_g_fit = a_g[power[a_g] > min(power[a_g]) + .05*pow_r]
        popt, ym = gauss_fit_peak(period[a_g_fit], power[a_g_fit])
    else:
        if period[a_g] == p_max_thresh:
            popt = [1.0, p_max_thresh, 50.]
            a_g_fit = np.arange(a_g-10, a_g)
            ym = power[a_g_fit]
        elif period[a_g] == p_min_thresh:
            popt = [1.0, p_min_thresh, 50.]
            a_g_fit = np.arange(a_g, a_g+10)
            ym = power[a_g_fit]
        else:
            popt = [-999, -999, -999]
            a_g_fit = np.arange(a_g-2, a_g+3)
            ym = power[a_g_fit]
    
    per_a_o, power_a_o = period[a_o], power[a_o]
    per_2 = per_a_o[np.argmax(power[a_o])]
    pow_2 = power_a_o[np.argmax(power[a_o])]
    pow_pow2 = 1.0*power_best/pow_2
    tdiff = np.array(time-min(time))
    nflux = np.array(nflux)
    pha, cyc = np.modf(tdiff/period_best)
    pha, cyc = np.array(pha), np.array(cyc)
    f = np.argsort(pha)
    p = np.argsort(tdiff/period_best)
    pha_fit, nf_fit, ef_fit, cyc_fit = pha[f], nflux[f], enflux[f], cyc[f].astype(int)
    pha_plt, nf_plt, ef_plt, cyc_plt = pha[p], nflux[p], enflux[p], cyc[p].astype(int)
    try:
        pops, popsc = curve_fit(sin_fit, pha_fit, nf_fit,
                                bounds=(0, [2., 2., 1000.]))#, 2.*np.pi]))
    except Exception:
        logger.warning(Exception)
        pops, popsc = np.array([1., 0.001, 0.5]), 0
        pass

    # order the phase folded lightcurve by phase and split into N even parts.
    # find the standard deviation in the measurements for each bin and use
    # the median of the standard deviation values to represent the final scatter
    # in the phase curve.
     
    sca_mean, sca_stdev = mean_of_arrays(nf_fit, n_sca)
    sca_median = np.median(sca_stdev)

    Ndata = len(nflux)
    yp = sin_fit(pha_fit, *pops)
    chi_sq = np.sum(((yp-pha_fit)/ef_fit)**2)/(len(pha_fit)-len(pops)-1)
    chi_sq = np.sum((yp-pha_fit)**2)/(len(pha_fit)-len(pops)-1)
    
    pha_sct = MAD(yp - nflux, scale='normal')
    fdev = 1.*np.sum(np.abs(nflux - yp) > 3.0*pha_sct)/Ndata
    LS_dict['median_MAD_nLC'] = [med_f, MAD_f]
    LS_dict['jump_flag'] = jump_flag
    LS_dict['period'] = period
    LS_dict['power'] = power
    LS_dict['period_best'] = period_best
    LS_dict['power_best'] = power_best
    LS_dict['time'] = time 
    LS_dict['y_fit_LS'] = y_fit_sine
    LS_dict['AIC_sine'] = AIC_sine
    LS_dict['AIC_line'] = AIC_line
    LS_dict['FAPs'] = FAP_test
    LS_dict['Gauss_fit_peak_parameters'] = popt
    LS_dict['Gauss_fit_peak_y_values'] = ym
    LS_dict['period_around_peak'] = period[a_g_fit]
    LS_dict['power_around_peak'] = power[a_g_fit]
    LS_dict['period_not_peak'] = period[a_o] 
    LS_dict['power_not_peak'] = power[a_o] 
    LS_dict['period_second'] = per_2
    LS_dict['power_second'] = pow_2
    LS_dict['phase_fit_x'] = pha_fit
    LS_dict['phase_fit_y'] = yp
    LS_dict['phase_x'] = pha_plt
    LS_dict['phase_y'] = nf_plt
    LS_dict['phase_chisq'] = chi_sq
    LS_dict['phase_col'] = cyc_plt
    LS_dict['pops_vals'] = pops    
    LS_dict['pops_cov'] = popsc
    LS_dict['phase_scatter'] = sca_median
    LS_dict['frac_phase_outliers'] = fdev
    LS_dict['Ndata'] = Ndata
    return LS_dict
    
    
def is_period_cont(d_target, d_cont, t_cont, frac_amp_cont=0.5):
    '''Identify neighbouring contaminants that may cause the periodicity.

    If the user selects to measure periods for the neighbouring contaminants
    this function returns a flag to assess if a contaminant may actually be
    the source causing the observed periodicity.

    parameters
    ----------
    d_target : `dict`
        A dictionary containing periodogram data of the target star.
    d_cont : `dict`
        A dictionary containing periodogram data of the contaminant star.
    t_cont : `astropy.table.Table`
        A table containing Gaia data for the contaminant star
    frac_amp_cont : `float`, optional, default=0.5
        The threshold factor to account for the difference in amplitude
        of the two stars. If this is high, then the contaminants will be
        less likely to be flagged as the potential source
    
    returns
    -------
    output : `str`
        | Either ``a``, ``b`` or ``c``.
        | (a) The contaminant is probably the source causing the periodicity
        | (b) The contaminant might be the source causing the periodicity
        | (c) The contaminant is not the source causing the periodicity
        
    '''
    per_targ = d_target["period_best"]
    per_cont = d_cont["period_best"]
    err_targ = d_target["Gauss_fit_peak_parameters"][2]
    err_cont = d_cont["Gauss_fit_peak_parameters"][2]
    amp_targ = d_target["pops_vals"][1]
    amp_cont = d_cont["pops_vals"][1]
    flux_frac = 10**(t_cont["log_flux_frac"])

    if abs(per_targ - per_cont) < (err_targ + err_cont):
        if amp_targ/amp_cont > (frac_amp_cont*flux_frac):
            output = 'a'
        else:
            output = 'b'
    else:
        output = 'c'
    return output
