'''

Alexander Binks & Moritz Guenther, January 2023

Licence: MIT 2023

This module contains functions which are called upon by either the tess_cutouts.py
or tess_large_sectors.py modules. These are:

1)  aper_run - returns a numpy array containing the times, magnitudes and
    fluxes from aperture photometry.
    
2)  clean_lc/detrend_lc/make_lc - these 3 functions are made to remove spurious data
    points, ensure that only "strings" of contiguous data are being processed, and
    each string is detrended using either a linear (1st) or quadratic (2nd) polynomial
    (chosen by Aikake Inference Criterion) using only data contained in each string.
    Finally, the lightcurve is pieced together and make_lc returns a table containing
    the data ready for periodogram analysis.
    
3)  run_LS - function to conduct Lomb-Scargle periodogram analysis. Returns a table
    with period measurements, plus several data quality flags. If required a plot of
    the lightcurve, periodogram and phase-folded lightcurve is provided.

'''

# imports
import logging
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

# Local application imports
from .tess_stars2px import tess_stars2px_function_entry
from .fixedconstants import *



# initialize the logger object
logger = logging.getLogger(__name__)
logger_aq = logging.getLogger("astroquery")
logger_aq.setLevel(logging.ERROR)    
    



def get_xy_pos(targets, head):
    '''Locate the X-Y position for targets in a given Sector/CCD/Camera mode
    
    .. _astropy.table.Table: https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.timeseries.Table
    .. |astropy.table.Table| replace:: **astropy.table.Table** 
    .. _astropy.io.fits: https://docs.astropy.org/en/stable/io/fits/index.html
    .. |astropy.io.fits| replace:: **astropy.io.fits**
    .. _tuple: https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences
    .. |tuple| replace:: **tuple**

    parameters
    ----------
    targets : |astropy.table.Table|_
        The table of input data with celestial coordinates.
    head : |astropy.io.fits|_
        The fits header containing the WCS coordinate details.

    returns
    -------
    positions : |tuple|_
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

    .. _str: https://docs.python.org/3/library/stdtypes.html#str
    .. |str| replace:: **str**
    .. _float: https://docs.python.org/3/library/functions.html#float
    .. |float| replace:: **float**
    .. _tuple: https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences
    .. |tuple| replace:: **tuple**
    .. _astropy.table.Table: https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.timeseries.Table
    .. |astropy.table.Table| replace:: **astropy.table.Table** 

    This function reads in each fits file, and performs aperture photometry.
    A table of aperture photometry results is returned, which forms the raw
    lightcurve to be processed in subsequent functions.

    parameters
    ----------
    file_in : |str|_
        Name of the fits file containing image data
    targets : |astropy.table.Table|_
        The table of input data
    Rad : |float|_, optional, default=1.
        The pixel radius to define the circular area for the aperture
    SkyRad : |tuple|_, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux
    XY_pos : |tuple|_, optional, default=(10.,10.)
        The X-Y centroid (in pixels) of the aperture.

    returns
    -------
    full_phot_table : |astropy.table.Table|_
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
                    n_steps = data.shape[0]-1
                    head = hdul[0].header
                    qual_val = data["QUALITY"]
                    time_val = data["TIME"]
                    flux_vals = data["FLUX"]
                    erro_vals = data["FLUX_ERR"]
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
                        #get the image statistics for the background annulus
                        aperstats = ApertureStats(flux_vals[:][:][n_step],
                                                  annulus_aperture)
                        #obtain the raw (source+background) flux
                        phot_table = aperture_photometry(
                                     flux_vals[:][:][n_step], aperture,
                                     error=erro_vals[:][:][n_step])
                        #calculate the background contribution to the aperture
                        aperture_area = aperture.area_overlap(
                                        flux_vals[:][:][n_step])
                        #print out the data to "phot_table"
                        phot_table['id'] = targets['source_id']
                        phot_table['id'] = phot_table['id'].astype(str)
                        phot_table['bkg'] = aperstats.median
                        phot_table['total_bkg'] = \
                            phot_table['bkg'] * aperture_area
                        phot_table['aperture_sum_bkgsub'] = \
                            phot_table['aperture_sum'] - phot_table['total_bkg']
                        with warnings.catch_warnings():
                            warnings.simplefilter(action='ignore',
                                                  category=RuntimeWarning)
                            phot_table['mag'] = -2.5*np.log10(
                                phot_table['aperture_sum_bkgsub'])+Zpt
                            phot_table['mag_err'] = np.abs((-2.5/np.log(10))*\
                                phot_table['aperture_sum_err']/\
                                phot_table['aperture_sum'])
                        phot_table['time'] = time_val[n_step]
                        fix_cols = ['id', 'xcenter', 'ycenter',
                                    'aperture_sum', 'aperture_sum_err', 'bkg',
                                    'total_bkg', 'aperture_sum_bkgsub', 'mag',
                                    'mag_err', 'time']
                        phot_table = phot_table[fix_cols]
                        for r in range(len(phot_table)):
                            full_phot_table.add_row(phot_table[r])
        except:
            print(f"There is a problem opening the file {f_file}")
            logger.error(f"There is a problem opening the file {f_file}")
            continue
    return full_phot_table




def clean_lc(t, f, MAD_fac=2., time_fac=10., min_num_per_group=50):
    '''Remove data points from the lightcurve that are likely to be spurious.

    .. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    .. |Iterable| replace:: **Iterable** 
    .. _float: https://docs.python.org/3/library/functions.html#float
    .. |float| replace:: **float**
    .. _int: https://docs.python.org/3/library/functions.html#int
    .. |int| replace:: **int** 
    .. _list: https://docs.python.org/3/tutorial/datastructures.html
    .. |list| replace:: **list**

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
    t : |Iterable|_
        The set of time coordinates (in days)
    f : |Iterable|_
        The set of normalised flux coordinates
    MAD_fac : |float|_, optional, default=2.
        The threshold number of MAD values to allow. 
    time_fac : |float|_, optional, default=10.
        The maximum time gap allowed between neighbouring datapoints.
    min_num_per_group : |int|_, optional, default=50
        The minimum number of datapoints allowed in a contiguous set.

    returns
    -------
    start_index : |list|_
       The start indices for each saved data string.
    end_index : |list|_
       The end indices for each saved data string.
    '''
    
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



def detrend_lc(ds,df,t,f,err):
    '''Detrend and normalise the lightcurves.

    .. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    .. |Iterable| replace:: **Iterable** 
    .. _dict: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    .. |dict| replace:: **dict**


    This function operates on each section of data returned from the "clean_lc"
    function and performs a detrending routine so that data from separate
    sections can be connected. Five lists are outputted which form the
    normalized, detrended lightcurve.
    
    The choice of polynomial fit for the detrending function is linear or
    quadratic, and depends on which component best satisfies the Aikake
    Information Criterion.

    parameters
    ----------
    ds : |Iterable|_
        the start indices for the lightcurve.
    df : |Iterable|_
        the end indices for the lightcurve.
    t : |Iterable|_
        The list of time coordinates.
    f : |Iterable|_
        The list of flux coordinates.
    err : |Iterable|_
        The list of flux error coordinates.

    returns
    -------
    dict_lc : |dict|_
        | A dictionary containing the following keys:
        | "time" -> The time coordinate
        | "oflux" -> The original, normalised flux values
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The error on "nflux"
        | "polyord" -> The polynomial order used for each detrend        
    '''
    dict_lc = {"time":[], "oflux":[], "nflux":[], "enflux":[], "polyord":[]}
    for i in range(len(ds)):
        rs, rf = ds[i], df[i]+1
        t_orig, f_orig, e_orig = t[rs:rf], f[rs:rf], err[rs:rf]
        p1, r1, _,_,_ = np.polyfit(t_orig, f_orig, 1, full=True)
        p2, r2, _,_,_ = np.polyfit(t_orig, f_orig, 2, full=True)
        chi1 = np.sum((np.polyval(p1, t_orig-f_orig)**2)/(e_orig)**2)
        chi2 = np.sum((np.polyval(p2, t_orig-f_orig)**2)/(e_orig)**2)
        AIC1, AIC2 = 2.*(2. - np.log(chi1/2.)), 2.*(3. - np.log(chi2/2.))
        if AIC1 < AIC2:
            f_n = f_orig/np.polyval(p2, t_orig)
            s_fit = 1
        else:
            f_n = f_orig/np.polyval(p2, t_orig)
            s_fit = 2
        for i in range(len(t_orig)):
            dict_lc["time"].append(t_orig[i])
            dict_lc["oflux"].append(f_orig[i])
            dict_lc["nflux"].append(f_n[i])
            dict_lc["enflux"].append(e_orig[i])
            dict_lc["polyord"].append(s_fit)
    return dict_lc


def make_lc(phot_table):
    '''Construct the normalised TESS lightcurve.
    
    .. _astropy.table.Table: https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.timeseries.Table
    .. |astropy.table.Table| replace:: **astropy.table.Table** 
    .. _dict: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    .. |dict| replace:: **dict**

    | The function runs the following tasks:
    | (1) Read the table produced from the aperture photometry
    | (2) Normalise the lightcurve using the median flux value.
    | (3) Clean the lightcurve from spurious data using "clean_lc"
    | (4) Detrend the lightcurve using "detrend_lc"
    | (5) Return the original normalised lightcurve and the cleaned lightcurve

    parameters
    ----------
    phot_table : |astropy.table.Table|_ or |dict|_
        | The data table containing aperture photometry returned by aper_run.py. Columns must include:
        | "time" -> The time coordinate for each image
        | "mag" -> The target magnitude
        | "flux_corr" -> The total flux subtracted by the background flux
        | "flux_err" -> The error on flux_corr
        

    returns
    -------
    cln : |dict|_
        | The cleaned, detrended, normalised lightcurve, with the keys:
        | "time" -> The time coordinate
        | "time0" -> The time coordinate relative to the first data point
        | "oflux" -> The original, normalised flux values
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The error on "nflux"
        | "polyord" -> The polynomial order used for each detrend        

    orig : |dict|_
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
    ds, df = clean_lc(time, flux)

    if (len(ds) == 0) or (len(df) == 0):
        return# [], []
    # 1st: normalise the flux by dividing by the median value
    nflux  = flux/np.median(flux)
    neflux = eflux/flux
    # 2nd: detrend each lightcurve sections by either a straight-line fit or a
    # parabola. The choice is selected using AIC.
    cln = detrend_lc(ds, df, time, nflux, neflux)
    cln["time0"] = [cln["time"][i]-time[0] for i in range(len(cln["time"]))]
    orig = dict()
    orig["time"] = np.array(time)
    orig["nflux"] = np.array(nflux)
    orig["mag"] = np.array(mag)
    if len(cln["time"]) > 50:
        return cln, orig
    else:
        return# [], []



def get_second_peak(power):
    '''An algorithm to identify the second-highest peak in the periodogram
    
    .. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    .. |Iterable| replace:: **Iterable** 
    .. _list: https://docs.python.org/3/tutorial/datastructures.html
    .. |list| replace:: **list**

    parameters
    ----------
    power : |Iterable|_
        A set of power values calculated from the periodogram analysis.

    returns
    -------
    a_g : |list|_
        A list of indices corresponding to the Gaussian around the peak power.
    a_o : |list|_
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


def filter_out_none(unfiltered):
    '''
    Simple function to remove list items if they are empty.

    .. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    .. |Iterable| replace:: **Iterable** 
    .. _list: https://docs.python.org/3/tutorial/datastructures.html
    .. |list| replace:: **list**

    parameters
    ----------
    unfiltered : |Iterable|_
        The input set of values

    returns
    -------
    filtered : |list|_
        The filtered list.

    >>> filter_out_none([1, 4, None, 6, 9])
    [1, 4, 6, 9]
    '''

    filtered = list(filter(lambda item: item is not None, unfiltered))
    return filtered


def gauss_fit(x, a0, x_mean, sigma):
    '''Construct a simple Gaussian.
    
    .. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    .. |Iterable| replace:: **Iterable** 
    .. _float: https://docs.python.org/3/library/functions.html#float
    .. |float| replace:: **float**
    .. _list: https://docs.python.org/3/tutorial/datastructures.html
    .. |list| replace:: **list**

    Return Gaussian values from a given amplitude (a0), mean (x_mean) and
    uncertainty (sigma) for a distribution of values

    parameters
    ----------
    x : |Iterable|_
        list of input values
    a0 : |float|_
        Amplitude of a Gaussian
    x_mean : |float|_
        The mean value of a Gaussian
    sigma : |float|_
        The Gaussian uncertainty

    returns
    -------
    gaussian : |list|_
        A list of Gaussian values.
    '''

    gaussian = a0*np.exp(-(x-x_mean)**2/(2*sigma**2))
    return gaussian

def sine_fit(x, y0, A, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase) to a regular
    sinusoidal function.

    .. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
    .. |Iterable| replace:: **Iterable** 
    .. _float: https://docs.python.org/3/library/functions.html#float
    .. |float| replace:: **float**
    .. _list: https://docs.python.org/3/tutorial/datastructures.html
    .. |list| replace:: **list**

    parameters
    ----------
    x : |Iterable|_
        list of input values
    y0 : |float|_
        The midpoint of the sine curve
    A : |float|_
        The amplitude of the sine curve
    phi : |float|_
        The phase angle of the sine curve

    returns
    -------
    sine : |list|_
        A list of sine curve values.
    '''
    sine = y0 + A*np.sin(2.*np.pi*x + phi)
    return sine


def run_ls(cln, p_min_thresh=0.05, p_max_thresh=100., samples_per_peak=10):
    '''Run Lomb-Scargle periodogram and return a dictionary of results.

    .. _dict: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    .. |dict| replace:: **dict**
    .. _float: https://docs.python.org/3/library/functions.html#float
    .. |float| replace:: **float**
    .. _int: https://docs.python.org/3/library/functions.html#int
    .. |int| replace:: **int** 

    parameters
    ----------
    cln : |dict|_
        A dictionary containing the lightcurve data. The keys must include
        | "time0" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
    p_min_thresh : |float|_, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : |float|_, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : |int|_, optional, default=10
        The number of samples to measure in each periodogram peak.

    returns
    -------
    LS_dict : |dict|_
        A dictionary of parameters calculated from the periodogram analysis
    '''
    LS_dict = dict()
    med_f, MAD_f = np.median(cln["nflux"]), MAD(cln["nflux"], scale='normal')
    ls = LombScargle(cln["time0"], cln["nflux"])
    frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                    maximum_frequency=1./p_min_thresh,
                                    samples_per_peak=samples_per_peak)
    FAP = ls.false_alarm_probability(power.max())
    probabilities = [0.1, 0.05, 0.01]
    FAP_test = ls.false_alarm_level(probabilities)
    p_m = np.argmax(power)
    y_fit = ls.model(cln["time0"], frequency[p_m])
    period_best = 1.0/frequency[p_m]
    power_best = power[p_m]

    period = 1./frequency[::-1]
    power = power[::-1]
    # a_g: array of datapoints that form the Gaussian around the highest power
    # a_o: the array for all other datapoints
    a_g, a_o = get_second_peak(power)
    try:
        popt, _ = curve_fit(gauss_fit, period[a_g], power[a_g],
                            bounds=(0, [1., p_max_thresh, p_max_thresh]))
    except Exception:
        logger.warning(Exception)
        popt = np.array([1, p_max_thresh/2, p_max_thresh/2])
        pass

    ym = gauss_fit(period[a_g], *popt)
    per_a_o, power_a_o = period[a_o], power[a_o]
    per_2 = per_a_o[np.argmax(power[a_o])]
    pow_2 = power_a_o[np.argmax(power[a_o])]
    pow_pow2 = 1.0*power_best/pow_2
    tdiff = np.array(cln["time0"]-min(cln["time0"]))
    nflux = np.array(cln["nflux"])
    pha, cyc = np.modf(tdiff/period_best)
    pha, cyc = np.array(pha), np.array(cyc)
    f = np.argsort(pha)
    p = np.argsort(tdiff/period_best)
    pha_fit, nf_fit, cyc_fit = pha[f], nflux[f], cyc[f].astype(int)
    pha_plt, nf_plt, cyc_plt = pha[p], nflux[p], cyc[p].astype(int)
    try:
        pops, popsc = curve_fit(sine_fit, pha_fit, nf_fit,
                                bounds=(0, [2., 2., 2.*np.pi]))
    except Exception:
        logger.warning(Exception)
        pops = np.array([1., 0.001, 0.5])
        pass
            
    Ndata = len(cln)
    yp = sine_fit(pha_fit, *pops)
    pha_sct = MAD(yp - nflux, scale='normal')
    fdev = 1.*np.sum(np.abs(nflux - yp) > 3.0*pha_sct)/Ndata
    LS_dict['median_MAD_nLC'] = [med_f, MAD_f]
    LS_dict['period'] = period
    LS_dict['power'] = power
    LS_dict['period_best'] = period_best
    LS_dict['power_best'] = power_best 
    LS_dict['y_fit_LS'] = y_fit 
    LS_dict['FAPs'] = FAP_test
    LS_dict['Gauss_fit_peak_parameters'] = popt
    LS_dict['Gauss_fit_peak_y_values'] = ym
    LS_dict['period_around_peak'] = period[a_g]
    LS_dict['power_around_peak'] = power[a_g]
    LS_dict['period_not_peak'] = period[a_o] 
    LS_dict['power_not_peak'] = power[a_o] 
    LS_dict['period_second'] = per_2
    LS_dict['power_second'] = pow_2
    LS_dict['phase_fit_x'] = pha_fit
    LS_dict['phase_fit_y'] = yp
    LS_dict['phase_x'] = pha_plt
    LS_dict['phase_y'] = nf_plt
    LS_dict['phase_col'] = cyc_plt
    LS_dict['pops_vals'] = pops    
    LS_dict['pops_cov'] = popsc
    LS_dict['phase_scatter'] = pha_sct
    LS_dict['frac_phase_outliers'] = fdev
    LS_dict['Ndata'] = Ndata
    return LS_dict
    
    
def is_period_cont(d_target, d_cont, t_cont, frac_amp_cont=0.5):
    '''Identify neighbouring contaminants that may cause the periodicity.

    .. _dict: https://docs.python.org/3/tutorial/datastructures.html#dictionaries
    .. |dict| replace:: **dict**
    .. _astropy.table.Table: https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.timeseries.Table
    .. |astropy.table.Table| replace:: **astropy.table.Table** 
    .. _float: https://docs.python.org/3/library/functions.html#float
    .. |float| replace:: **float**
    .. _str: https://docs.python.org/3/library/stdtypes.html#str
    .. |str| replace:: **str**

    If the user selects to measure periods for the neighbouring contaminants
    this function returns a flag to assess if a contaminant may actually be
    the source causing the observed periodicity.

    parameters
    ----------
    d_target : |dict|_
        A dictionary containing periodogram data of the target star.
    d_cont : |dict|_
        A dictionary containing periodogram data of the contaminant star.
    t_cont : |astropy.table.Table|_
        A table containing Gaia data for the contaminant star
    frac_amp_cont : |float|_, optional, default=0.5
        The threshold factor to account for the difference in amplitude
        of the two stars. If this is high, then the contaminants will be
        less likely to be flagged as the potential source
    
    returns
    -------
    output : |str|_
        | Either `a', `b' or `c'
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


