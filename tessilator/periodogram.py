'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

A set of functions to perform a Lomb-Scargle periodogram analysis to the TESS
lightcurves.

'''
###############################################################################
####################################IMPORTS####################################
###############################################################################
#Internal
import warnings

# Third party imports
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import inspect

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.stats import akaike_info_criterion_lsq

from scipy.stats import median_abs_deviation as MAD
from scipy.stats import mode
from scipy.stats import chi2

from scipy.optimize import curve_fit
import itertools as it

from collections.abc import Iterable


# Local application imports
from .file_io import logger_tessilator
from .lc_analysis import sin_fit
###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__) 




def check_for_jumps(time, flux, lc_part, n_avg=10, thresh_diff=10.):
    '''Identify if the lightcurve has jumps.
    
    A jumpy lightcurve is one that has small contiguous data points that change
    in flux significantly compared to the amplitude of the lightcurve. These
    could be due to some instrumental noise or response to a non-astrophysical
    effect. They may also be indicative of a stellar flare or active event.
    
    This function takes a running average of the differences in flux, and flags
    lightcurves if the absolute value exceeds a threshold. These will be
    flagged as "jumpy" lightcurves.

    parameters
    ----------
    time : `Iterable`
        The time coordinate
    flux : `Iterable`
        The original, normalised flux values
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    n_avg : `int`, optional, default=10
        The number of data points to calculate the running average
    thresh_diff : `float`, optional, default=10.
        The threshold value, which, if exceeded, will yield a "jumpy"
        lightcurve

    returns
    -------
    jump_flag : `bool`
        This will be True if a jumpy lightcurve is identified, otherwise False.
    '''
    
    jump_flag = False
    try:
        for lc in np.unique(lc_part):
            g = np.array(lc_part == lc)
        
            f_mean = np.convolve(flux[g], np.ones(n_avg), 'valid') / n_avg
            t_mean = np.convolve(flux[g], np.ones(n_avg), 'valid') / n_avg
        
            f_shifts = np.abs(np.diff(f_mean))

            median_f_shifts = np.median(f_shifts)
            max_f_shifts = np.max(f_shifts)
            if max_f_shifts/median_f_shifts > thresh_diff:
                jump_flag = True
                return jump_flag
    except:
        logger.error('Could not run the jump flag criteria for this target.')
        return jump_flag
    return jump_flag


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


def gauss_fit_peak(period, power, max_power=1):
    '''Applies the Gaussian fit to the periodogram. If there are more than 3
    datapoints (i.e., more datapoints than fixed parameters), the "gauss_fit"
    module is used to return the fit parameters. If there are 3 or less points,
    the maximum peak is located and 9 datapoints are interpolated between the
    2 neighbouring data points of the maximum peak, and the "gauss_fit" module
    is applied.
    
    parameters
    ----------
    period : `Iterable`
        The period values around the peak.
    power : `Iterable`
        The power values around the peak.
    max_power : `float`, optional, default=1
        The maximum value for the power output.
        
    returns
    -------
    popt : `list`
        The best-fit Gaussian parameters: A, B and C where A is the amplitude,
        B is the mean and C is the uncertainty.
    ym : `list`
        The y values calculated from the Gaussian fit.
    '''
    if len(period) > 3:
        try:
            period_diff = period[-1]-period[0]
            popt, _ = curve_fit(gauss_fit, period, power,
                                bounds=([0, period[0], 0],
                                        [max_power, period[-1], period_diff]))
            ym = gauss_fit(period, *popt)
        except:
            logger.error(f"Couldn't find the optimal parameters for the "
                         f"Gaussian fit!")
            p_m = np.argmax(power)
            peak_vals = [p_m-1, p_m, p_m+1]
            x = period[peak_vals]
            y = power[peak_vals]
            xvals = np.linspace(x[0], x[-1], 9)
            yvals = np.interp(xvals, x, y)
            popt, _ = curve_fit(gauss_fit, xvals, yvals,
                                bounds=(0, [1., np.inf, np.inf]))
            ym = gauss_fit(xvals, *popt)     

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
    
    
def get_next_peak(power, frac_peak=0.85, option=None):
    '''An algorithm to identify the "next"-highest peak in the periodogram

    parameters
    ----------
    power : `Iterable`
        A set of power values calculated from the periodogram analysis.
    frac_peak : `float`, optional, default=0.85
        The relative height of the maximum peak, below which the data will be
        included.
    option : `float`, optional, default=None
        The value that the power must also be below. If kept as None, this is
        equal to frac_peak*the highest power in the array.

    returns
    -------
    a_o : `list`
        A list of indices corresponding to all other parts of the periodogram.
    '''
    # Get the left side of the peak
    a = np.arange(len(power))

    p_m = np.argmax(power)

    if not option:
        cond_2 = frac_peak*power[p_m]
    else:
        cond_2 = option

    x = p_m
    while (power[x-1] < power[x]) and (x > 0):
        x = x-1
    p_l = x
    p_lx = 0
    while (power[p_l] > cond_2) and (p_l > 1):
        p_lx = 1
        p_l = p_l - 1
    if p_lx == 1:
        while (power[p_l-1] < power[p_l]) and (p_l > 0):
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
        while (power[p_r] > cond_2) and (p_r < len(power)-3):
            p_rx = 1
            p_r = p_r + 1
        if p_rx == 1:
           while (power[p_r+1] < power[p_r]) and (p_r < len(power)-2):
                p_r = p_r + 1
        if p_r > len(power)-1:
            p_r = len(power)-1
    elif x == len(power)-1:
        p_r = x

    #return the indices that do not constitute part of the specific periodogram
    #peak.
    a_g = a[p_l:p_r+2]
    a_o = a[np.setdiff1d(np.arange(a.shape[0]), a_g)] 
    return a_o


def mean_of_arrays(arr, num):
    '''Calculate the mean and standard deviation of an array which is split
    into N components.
    
    parameters
    ----------
    arr : `Iterable`
        The input array
    num : `int`
        The number of arrays in which to (equally) split the data

    returns
    -------
    mean_out : `float`
        The mean of the list of arrays.
    std_out : `float`
        The standard deviation of the list of arrays.
    '''
    x = np.array_split(arr, num)
    ar = np.array(list(it.zip_longest(*x, fillvalue=np.nan)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) 
        mean_out, std_out = np.nanmean(ar, axis=0), np.nanstd(ar, axis=0) 
    return mean_out, std_out
    

def get_Gauss_params_pg(period, power, indices=None, max_power=1,
                        gauss_min_frac=0.05, p_min_thresh=0.05,
                        p_max_thresh=100.):
    '''Calculate the best Gaussian-fit parameters to the periodogram output.
    
    parameters
    ----------
    period : `Iterable`
        The set of period outputs from the periodogram analysis.
    power : `Iterable`
        The set of power outputs from the periodogram analysis.
    indices : `Iterable`, optional, default=None
        The specific indices of the period/power array to be used.
    max_power : `float`, optional, default=1
        The maximum value for the power output.
    gauss_min_frac : `float`, optional, default=0.05
        The minimum ratio between the the Gaussian fit and max_power.
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.

    results
    -------
    returns
    -------
    popt : `list`
        The best-fit Gaussian parameters: A, B and C where A is the amplitude,
        B is the mean and C is the uncertainty.
    ym : `list`
        The y values calculated from the Gaussian fit.
    ''' 
# first -- check there are more than 3 values for the Gaussian fit.
    if isinstance(indices, Iterable):
        ind=indices
    else:
        ind=np.arange(len(period))

    if len(ind) > 3:
        pow_r = max(power[ind])-min(power[ind])
        ind_fit = ind[power[ind] >= min(power[ind]) + gauss_min_frac*pow_r]
        popt, ym = gauss_fit_peak(period[ind], power[ind], max_power=max_power)
    else:
        if np.isclose(period[ind], p_max_thresh, atol=0.001).any():
            popt = [1.0, p_max_thresh, 50.]
            ind_fit = np.arange(ind-10, ind)
            ym = power[ind]
        elif np.isclose(period[ind], p_min_thresh, atol=0.001).any():
            popt = [1.0, p_min_thresh, 50.]
            ind_fit = np.arange(ind, ind+10)
            ym = power[ind_fit]
        else:
            popt = [-999, -999, -999]
            n, i_n = 0, ind[-1]
            while i_n+1 < len(power):
                n+=1
                i_n+=1
                if n==3:
                    break
            m, i_m = 0, ind[0]
            while i_m-1 > 0:
                m-=1
                i_m-=1
                if m==-2:
                    break
            ind_fit = np.arange(ind[0]-m, ind[-1]+n)
            ym = power[ind_fit-1]
            
    return popt, ym


def initialise_LS_dict(lc_data, check_jump=False, p_min_thresh=0.05,
                       p_max_thresh=100., samples_per_peak=50):
    '''Run the periodogram analysis and store initial results to a dictionary.
    
    parameters
    ----------
    lc_data : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous
                       sections
    check_jump : `bool`, optional, default=False
        Choose to check the lightcurve for jumpy data, using the
        "check_for_jumps" function.
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : `int`, optional, default=10
        The number of samples to measure in each periodogram peak.

    results
    -------
    LS_dict : `dict`
        The dictionary of initial parameters from the periodogram analysis
    cln_lc : `dict`
        The lightcurve data, with any lines containing a False flag removed.
    ls : `astropy.timeseries.LombScargle`
        The Lomb-Scargle periodogram object for the given lightcurve.
    
    '''
    # a_g: array of datapoints that form the Gaussian around the highest power
    # a_o: the array for all other datapoints
    LS_dict = {}

    cln_cond = (np.logical_and.reduce([
                   lc_data["pass_clean_scatter"],
                   lc_data["pass_clean_outlier"],
                   lc_data["pass_full_outlier"]
                   ])) & (lc_data["nflux_dtr"] > -999.)
    cln_lc = lc_data[cln_cond]

    
   
    LS_dict["time"] = np.array(cln_lc["time"])
    LS_dict["nflux"] = np.array(cln_lc["nflux_dtr"])
    LS_dict["enflux"] = np.array(cln_lc["nflux_err"])
    LS_dict["lc_part"] = np.array(cln_lc["lc_part"])

    # calculate the median and MAD flux
    LS_dict['median_MAD_nLC'] = [np.median(LS_dict["nflux"]),
                                 MAD(LS_dict["nflux"], scale='normal')]

    # calculate the TESS magnitude (median and MAD value) 
    # note that this is from the "ORIGINAL LIGHTCURVE"
    try:
        mag = lc_data["mag"]
        LS_dict['Tmag_MED'] = np.median(mag[mag > -999])
        LS_dict['Tmag_MAD'] = MAD(mag[mag > -999])
    except:
        LS_dict['Tmag_MED'], LS_dict['Tmag_MAD'] = -999, -999    


    # assess the jump flag
    LS_dict['jump_flag'] = -999
    if check_jump:
        LS_dict['jump_flag'] = int(check_for_jumps(LS_dict["time"],
                                                   LS_dict["nflux"],
                                                   LS_dict["lc_part"]))


    # run the LS periodogram
    ls = LombScargle(LS_dict["time"], LS_dict["nflux"], dy=LS_dict["enflux"])
    frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                    maximum_frequency=1./p_min_thresh,
                                    samples_per_peak=samples_per_peak)

    p_m = np.argmax(power)

    LS_dict['a_1'] = np.arange(len(power))
    LS_dict['period_a_1'] = 1./frequency[::-1]
    LS_dict['power_a_1'] = power[::-1]
    LS_dict['period_1'] = 1.0/frequency[p_m]
    LS_dict['power_1'] = power[p_m]



   # calculate the false alarm probability values, using the quick method
    try:
        FAP = ls.false_alarm_probability(power.max())
        probabilities = [0.1, 0.05, 0.01]
        LS_dict['FAPs'] = ls.false_alarm_level(probabilities)
    except:
        logger.error('Something went wrong with the FAP test, maybe division '
                     'by 0.')
        LS_dict['FAPs'] = np.array([0.3, 0.2, 0.1])


    LS_dict['shuffle_flag'] = 0
    LS_dict['period_shuffle'] = -999
    LS_dict['period_shuffle_err'] = -999

    return LS_dict, cln_lc, ls


def write_periodogram(LS_dict, name_pg='', pg_dir='', lc_type=''):
    '''Save the period and power output from the periodogram analysis to file.
    
    parameters
    ----------
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
    name_pg : `str`, optional, default=''
        The name of the file to save the periodogram results
    pg_dir : `str`, optional, default=''
        The directory of the file to save the periodogram results
    lc_type : `str`, optional, default=''
        An additional string for reference in the file name.
    
    returns
    -------
    Nothing returned. The result is saved to file.
    '''
    if name_pg:
        res_table = Table(names=('period', 'power'), dtype=(float,float))
        for pe, po in zip(LS_dict['period_a_1'], LS_dict['power_a_1']):
            res_table.add_row([pe, po])
        res_table['period'].info.format = '.6f'
        res_table['power'].info.format = '.4e'
        res_table.write(f'{pg_dir}/{name_pg}_{lc_type}.csv', overwrite=True)


def get_periodogram_peaks(LS_dict, n_peaks=4):
    '''Calculate parameters for a given number of periodogram peaks
    
    This function locates the `n_peaks` highest peaks in the periodogram and
    calculates the associated period, power and Gaussian-fit parameters.
    
    parameters
    ----------
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
    n_peaks : `int`, optional, default=4
        The number of periodogram peaks to analyse.
        
    results
    -------
    Nothing returned, the LS_dict dictionary is updated with new parameters.
    '''
    for i in 1+np.arange(n_peaks):
        try:
            # get the indices of all the peaks that were not part of the last
            # peak
            LS_dict[f'a_{i+1}'] = get_next_peak(LS_dict[f'power_a_{i}'])
            # all the indices that 'are' part of the peak
            LS_dict[f'a_g_{i}'] = np.delete(np.array(LS_dict[f'a_{i}']),
                                            np.array(LS_dict[f'a_{i+1}']))
            x1, x2 = get_Gauss_params_pg(LS_dict["period_a_1"],
                                         LS_dict["power_a_1"],
                                         indices=LS_dict[f'a_g_{i}'])
            LS_dict[f'Gauss_{i}'], LS_dict[f'Gauss_y_{i}'] = x1, x2 
           # find all the new period values in the new array
            x3 = LS_dict[f'period_a_{i}'][LS_dict[f'a_{i+1}']]
            LS_dict[f'period_a_{i+1}'] = x3
            # find all the new power values in the new array
            x4 = LS_dict[f'power_a_{i}'][LS_dict[f'a_{i+1}']]
            LS_dict[f'power_a_{i+1}'] = x4
            # calculate the period of the maximum power peak
            x5 = LS_dict[f'period_a_{i+1}'][np.argmax(LS_dict[f'power_a_{i+1}'])]
            LS_dict[f'period_{i+1}'] = x5
            # return the maximum power peak value
            x6 = LS_dict[f'power_a_{i+1}'][np.argmax(LS_dict[f'power_a_{i+1}'])]
            LS_dict[f'power_{i+1}'] = x6
        except:
            logger.error('Something went wrong with the periods/powers of '
                         'subsequent peaks. Probably an empty array of '
                         'values.')
            LS_dict[f'Gauss_{i}'] = [-999, -999, -999]
            LS_dict[f'period_a_{i+1}'] = -999
            LS_dict[f'power_a_{i+1}'] = -999
            LS_dict[f'period_{i+1}'] = -999
            LS_dict[f'power_{i+1}'] = -999
        LS_dict.pop(f'a_{n_peaks+1}', None)
        LS_dict.pop(f'period_a_{n_peaks+1}', None)
        LS_dict.pop(f'power_a_{n_peaks+1}', None)
        LS_dict.pop(f'period_{n_peaks+1}', None)
        LS_dict.pop(f'power_{n_peaks+1}', None)

    try:
        LS_dict['period_around_1'] = LS_dict["period_a_1"][LS_dict['a_g_1']]
        LS_dict['power_around_1'] = LS_dict["power_a_1"][LS_dict['a_g_1']]
    except:
        LS_dict['period_around_1'] = -999
        LS_dict['power_around_1'] = -999


def shuffle_periodogram(lc_data, n_shuf_runs=100, p_min=0.1, p_max=100.,
                        n_min=0.1, n_max=1.0):
    '''Generate period measurements by sampling subsets of the lightcurve.
    
    parameters
    ----------
    lc_data : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous
                       sections
    n_shuf_runs : `int`, optional, default=100
        The number of measurements to be made
    p_min : `float`, optional, default=0.1
        The minimum period for the shuffling method
    p_max : `float`, optional, default=100.
        The maximum period for the shuffling method
    n_min : `float`, optional, default=0.1
        The minimum fraction of a group to be used in the periodogram analysis
    n_max : `float`, optional, default=1.0
        The maximum fraction of a group to be used in the periodogram analysis
    
    results
    -------
    periods_out : `np.array`
        The array of calculated periods.
    '''
    period_arr = []
    sections = np.unique(lc_data['lc_part'])
    for n_run in range(n_shuf_runs):
        try:
            s = np.random.choice(sections)
            lc_s = lc_data[lc_data["lc_part"] == s]
            n = np.random.uniform(low=n_min, high=n_max)
            l_n = int(n*len(lc_s))
            n_start = int(np.random.uniform(low=0.0, high=len(lc_s)-l_n))
            n_fin = n_start + l_n 
            lc_use = lc_s[n_start:n_fin]
            p_max = 4.*(lc_use['time'][-1] - lc_use['time'][0])
            time = np.array(lc_use["time"])
            nflux = np.array(lc_use["nflux_dtr"])
            p1, r1, _,_,_ = np.polyfit(time, nflux, 1, full=True)
            nflux_n = nflux/np.polyval(p1, time)

            enflux = np.array(lc_use["nflux_err"])

            ls = LombScargle(time, nflux_n, dy=enflux)

            frequency, power = ls.autopower(minimum_frequency=1./p_max,
                                            maximum_frequency=1./p_min,
                                            samples_per_peak=50)
            p_m = np.argmax(power)
            period_max = 1.0/frequency[p_m]
            power_max = power[p_m]
            period_arr.append(period_max)
        except:
            continue
    periods_out = np.array(period_arr)
    return periods_out
    

def plotticks_shuffle(crit, xpos, ypos, ax):
    '''Simple function for plotting tick marks on the shuffle plots

    parameters
    ----------
    crit : `bool`
        True or False according to a given criteria
    xpos : `float`
        x-position for the text (in normalised coordinates)
    ypos : `float`
        y-position for the text (in normalised coordinates)
    ax : `matplotlib.pyplot.axes`
        the axes object to apply the tickmarks to.
        
    returns
    -------
    ax : `matplotlib.pyplot.axes`
        the axes object to apply the tickmarks to (after the tickmarks)  
    '''
    if crit:
        ax.text(xpos, ypos, "\u2714", fontsize=30, color='green',
                transform=ax.transAxes, horizontalalignment='right')
    else:
        ax.text(xpos, ypos, "\u2718", fontsize=30, color='red',
                transform=ax.transAxes, horizontalalignment='right')
    return ax


def shuffle_check(cln_lc, LS_dict, shuf_per=False, n_shuf_runs=5000,
                  p_min=0.05, p_max=100., n_min=.1, n_max=1., bin1=50,
                  bin2_fac=10, n_peaks=4, make_shuf_plot=False,
                  shuf_dir='plot_shuf',
                  name_shuf_plot='example_shuf_plot.png'):
    '''Choose the period from original periodogram, or from the shuffled method

    In the case of low signal to noise, the original periodogram analysis
    will often predict an incorrect period because the noise is too dominant.
    Therefore, an alternative period can be incorporated, which is capable of
    detecting periods in noisy data.
    
    The idea is that a large number of periods are calculated from smaller
    portions of the whole lightcurve, and then if the resulting distribution
    of periods is small enough, then the shuffled period measurement is used as
    the main period.
    
    The algorithm works as follows:
    | 1. run the shuffle_periodogram function.
    | 2. construct the period histogram from the results.
    | 3. find the highest population bin, and select all neighbouring (period)
         values either side until this value becomes less than the median.
    | 4. calculate the number of periods that lie within and outside the period
         range found in part (2)
    | 5. construct another histogram for all periods that are within the range
         calculated in part (2)
    | 6. normalise the histogram so the whole distribution integrates to 1.0.
    | 7. ensure that the number of bins in the new histogram must be greater
         than 3, the number of periods outside the range is less than 0.5, and
         that the histogram must not peak at the start or end points.
         The process only continues if these conditions pass. Otherwise the
         shuffled periodogram value is not returned.
    | 8. fit a Gaussian to the new histogram.
    | 9. calculate "rrmse", the relative root mean square error.
    | 10. if rrmse < 0.5, the FWHM to the Gaussian fit is < 0.05, and the final
          shuffled period (centroid of the Gaussian) differs from the original
          period measurement by more than 10%, then return the shuffled
          periodogram result as the determined period. We set the power
          output=1.0, and the uncertainty is given by the sigma-value
          calculated in the Gaussian fit.
    | 11. finally, replace each set of nth "period, power and error" with the
          (n+1)th set, so the output set from the shuffled periodogram values
          take the highest significance. 

    parameters
    ----------
    cln_lc : `dict`
        The lightcurve data, with any lines containing a False flag removed.
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
        and modified by get_periodogram_peaks
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    n_shuf_runs : `int`, optional, default=5000
        The number of measurements to be made in shuffle_periodogram
    p_min : `float`, optional, default=0.05
        The minimum period for the shuffling method in shuffle_periodogram
    p_max : `float`, optional, default=100.
        The maximum period for the shuffling method in shuffle_periodogram
    n_min : `float`, optional, default=0.1
        The minimum fraction of a group to be used in the periodogram analysis
        in shuffle_periodogram
    n_max : `float`, optional, default=1.0
        The maximum fraction of a group to be used in the periodogram analysis
        in shuffle_periodogram
    bin1 : `int`, optional, default=50
        The number of histogram bins for the initial period distribution
    bin2_fac : `int`, optional, default=10
        The factor to use in calculating the number of histogram bins for the
        refined period distribution. The total number is the product of
        bin2_fac and the number of bins from the initial period distribution
        that are within the region surrounding the maximum bin occupancy.
    n_peaks : `int`, optional, default=4
        The number of peaks calculated in get_periodogram_peaks
    make_shuf_plot : `bool`, optional, default=False
        Choose to plot the outputs of the period distribution.
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period
        analysis. 
    name_plot_shuf : `str`, optional, default='example_plot_shuf.png'
        Choose the file name to save the period distribution plot.

    results
    -------
    Nothing returned, the LS_dict dictionary is updated with new parameters.
    '''
    if shuf_per:
        logger.info(f'Running the shuffle period algorithm.')
        try:
#1) run the shuffle_periodogram function.
            period_arr = shuffle_periodogram(cln_lc, n_shuf_runs=n_shuf_runs,
                                             p_min=p_min, p_max=p_max,
                                             n_min=n_min, n_max=n_max)

#2) construct the period histogram from the results.
            num_log10_per1, log10_per1 = np.histogram(np.log10(period_arr),
                                                      bins=bin1)
            diff_log10_per1 = np.diff(log10_per1)[0]

#3) find the highest population bin, and select all neighbouring (period) 
#   values either side until this value becomes less than the median.
            ind_others = get_next_peak(num_log10_per1,
                                       option=5.*np.median(num_log10_per1))
            ind_log10_per = np.delete(np.arange(len(log10_per1)-1), ind_others)

#4) calculate the number of periods that lie within and outside the period
#   range found in part (2)
            n_others = np.sum(num_log10_per1[ind_others])
            n_log10 = np.sum(num_log10_per1[ind_log10_per])

            x_l = log10_per1[ind_log10_per[0]]
            x_u = log10_per1[ind_log10_per[-1]]
            n_bin = (log10_per1 > (x_l - diff_log10_per1)) & \
                    (log10_per1 < (x_u + diff_log10_per1))

#5) construct another histogram for all periods that are within the range
#   calculated in part (2)
            num_log10_per2, log10_per2 = np.histogram(np.log10(period_arr),
                                                      bins=bin2_fac*np.sum(n_bin),
                                                      range=(x_l, x_u))
            log10_per2 = np.array([(log10_per2[i]+log10_per2[i+1])/2.
                                  for i in range(len(log10_per2)-1)])

#6) normalise the histogram so the whole distribution integrates to 1.0.
            num_log10_per2 = num_log10_per2/np.sum(num_log10_per2)

#7) ensure that the number of bins in the new histogram must be greater than 3,
#   the number of periods outside the range is less than 0.5, and that the
#   histogram must not peak at the start or end of the distribution. The 
#   process only continues if these conditions pass. Otherwise the shuffled
#   periodogram value is not returned.
            crit_1, crit_2 = False, False
            crit_1a = (len(ind_log10_per) > 3)
            crit_1b = (n_log10/(n_log10+n_others) > 0.5)
            crit_1c = (np.argmax(num_log10_per2) != 0) & \
                      (np.argmax(num_log10_per2) != len(num_log10_per2)-1)
            crit_2a, crit_2b, crit_2c = False, False, False
            
            if crit_1a & crit_1b & crit_1c:
                crit_1 = True

#8) fit a Gaussian to the new histogram.
                Gauss_param_log10_per, Gauss_log10_per = get_Gauss_params_pg(log10_per2, num_log10_per2, max_power=1.2*max(num_log10_per2))

#9) calculate "rrmse", the relative root mean square error.
                rrmse = relative_root_mean_squared_error(num_log10_per2,
                                                         Gauss_log10_per)

                p_shuf = 10**(Gauss_param_log10_per[1])
                p_shufu = 10**(Gauss_param_log10_per[1] + Gauss_param_log10_per[2])
                p_shufl = 10**(Gauss_param_log10_per[1] - Gauss_param_log10_per[2])
                p_shuf_err = 0.5*(p_shufu-p_shufl)

#10) if rrmse < 0.5, the FWHM to the Gaussian fit is < 0.05, and the final shuffled period (centroid of the Gaussian) differs from the original period measurement by more than 10%, then return the shuffled periodogram result as the determined period. We set the power output=1.0, and the uncertainty is given by the sigma-value calculated in the Gaussian fit.
                crit_2a = (rrmse < 0.5)
                crit_2b = (Gauss_param_log10_per[2] < 0.05)
                crit_2c = (((p_shuf/LS_dict['period_1'])<0.9) | \
                          ((p_shuf/LS_dict['period_1'])>1.1))

                if crit_2a & crit_2b & crit_2c:
                    crit_2 = True
#11) finally, replace each set of nth "period, power and error" with the (n+1)th set, so the output set from the shuffled periodogram values take the highest significance.
                    LS_dict['shuffle_flag'] = 1
                    LS_dict['period_shuffle'] = p_shuf
                    LS_dict['period_shuffle_err'] = p_shuf_err
                    for i in range(2,n_peaks+1):
                        LS_dict[f'period_{i}'] = LS_dict[f'period_{i-1}']
                        LS_dict[f'Gauss_{i}'] = LS_dict[f'Gauss_{i-1}']
                    LS_dict['period_1'] = p_shuf
                    LS_dict['power_1'] = 1.0
                    LS_dict['Gauss_1'] = [1.0, p_shuf, p_shuf_err]
                else:
                     logger.warning(f'failed second set of criteria: 2a={crit_2a}, 2b={crit_2b}, 2c={crit_2c}')
            else:
                logger.warning(f'failed first set of criteria: 1a={crit_1a}, 1b={crit_1b}, 1c={crit_1c}')            

            if make_shuf_plot:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
                ax[0].set_xlabel(r"$\log_{10}$period [d]")
                ax[0].set_ylabel('number of trials')
                ax[0].hist(np.log10(period_arr), bins=bin1)
                ax[0].axhline(5.*np.median(num_log10_per1), linestyle='--',
                              color='darkorange', linewidth=0.5)
                ax[0].axvline(x_l, color='darkorange', linewidth=0.5)
                ax[0].axvline(x_u, color='darkorange', linewidth=0.5)
                ax[1].set_xlabel(r"$\log_{10}$period [d]")
                ax[1].set_ylabel('normalised PDF')
                ax[1].plot(log10_per2, num_log10_per2)
                
                plotticks_shuffle(crit_1a, 0.94, 0.60, ax[1])
                plotticks_shuffle(crit_1b, 0.94, 0.50, ax[1])
                plotticks_shuffle(crit_1c, 0.94, 0.40, ax[1])
                plotticks_shuffle(crit_2a, 0.99, 0.60, ax[1])
                plotticks_shuffle(crit_2b, 0.99, 0.50, ax[1])
                plotticks_shuffle(crit_2c, 0.99, 0.40, ax[1])
                if crit_2:
                    ax[1].plot(log10_per2, Gauss_log10_per, linestyle='--', color='darkorange')
                    nl = '\n'
                    ax[1].text(0.99, 0.85, f"$P_{{\\rm rot}}$ (shuf) [d]{nl}{p_shuf:.3f}+/-{p_shuf_err:.3f}", transform=ax[1].transAxes, horizontalalignment='right')
                plt.savefig(f'{shuf_dir}/{name_shuf_plot}', bbox_inches='tight')
        except:
            logger.error(f'An error occured with the the period shuffling method.')
            LS_dict['period_shuffle'] = -9
            LS_dict['period_shuffle_err'] = -9
            

def make_phase_curve(LS_dict, ls, n_sca=10):
    '''Generate the phase-folded lightcurve using the peak periodogram period.
    
    This function performs several steps:
    
    1) generate the phase-folded lightcurve
    2) use Aikake Information Criterion to determine whether a sine-fit or a
       straight line is the most appropriate.
    3) Calculate the reduced chi-squared value for the sine fit.
    4) Calculate the typical amplitude and scatter in the phase-folded
       lightcurve.
    
    parameters
    ----------
    LS_dict : `dict`
        The dictionary of periodogram results produced in initialise_LS_dict
        and modified by get_periodogram_peaks and shuffle_check
    ls : `astropy.timeseries.LombScargle`
        The Lomb-Scargle periodogram object for the given lightcurve.
    n_sca : `int`, optional, default=10
        The number of portions to split the phase-folded lightcurve.

    results
    -------
    Nothing returned, the LS_dict dictionary is updated with new parameters.
    '''
    time = LS_dict["time"]
    nflux = LS_dict["nflux"]
    enflux = LS_dict["enflux"]
    freq_best = 1./LS_dict['period_1']

    y_fit_sine = ls.model(time, freq_best)
    y_fit_sine_param = ls.model_parameters(freq_best)
    chisq_model_sine = np.sum((y_fit_sine-nflux)**2/enflux**2)/(len(nflux)-3-1)
    line_fit, _,_,_,_ = np.polyfit(time, nflux, 1, full=True)
    y_fit_line = np.polyval(line_fit, time)
    chisq_model_line = np.sum((y_fit_line-nflux)**2/enflux**2)/(len(nflux)-len(line_fit)-1)
    AIC_sine, AIC_line = 2.*3. + chisq_model_sine, 2.*2. + chisq_model_line

    tdiff = np.array(time-min(time))
    pha, cyc = np.modf(tdiff/LS_dict['period_1'])
    pha, cyc = np.array(pha), np.array(cyc)
    f = np.argsort(pha)
    p = np.argsort(tdiff/LS_dict['period_1'])

    pha_fit, nf_fit, ef_fit, cyc_fit = pha[f], nflux[f], enflux[f], cyc[f].astype(int)
    pha_plt, nf_plt, ef_plt, cyc_plt = pha[p], nflux[p], enflux[p], cyc[p].astype(int)
    try:
        pops, popsc = curve_fit(sin_fit, pha_fit, nf_fit,
                                bounds=(0, [2., 2., 1000.]))
    except Exception:
        logger.warning(Exception)
        pops, popsc = np.array([1., 0.001, 0.5]), 0
        pass

    # order the phase folded lightcurve by phase and split into N even parts.
    # find the standard deviation in the measurements for each bin and use
    # the median of the standard deviation values to represent the final scatter
    # in the phase curve.

    Ndata = len(nflux)
    yp = sin_fit(pha_fit, *pops)
    chi_sq = np.sum(((yp-pha_fit)/ef_fit)**2)/(len(pha_fit)-len(pops)-1)
    chi_sq = np.sum((yp-pha_fit)**2)/(len(pha_fit)-len(pops)-1)
    
    pha_sct = MAD(yp - nflux, scale='normal')
    fdev = 1.*np.sum(np.abs(nflux - yp) > 3.0*pha_sct)/Ndata
    sca_mean, sca_stdev = mean_of_arrays(nf_fit/yp, n_sca)
    sca_median = np.median(sca_stdev)

    LS_dict['y_fit_LS'] = y_fit_sine
    LS_dict['AIC_sine'] = AIC_sine
    LS_dict['AIC_line'] = AIC_line
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


def run_ls(lc_data, lc_type='reg', ref_name='targets', pg_dir='pg', name_pg='pg_target', n_sca=10, p_min_thresh=0.05, p_max_thresh=100., samples_per_peak=10, n_peaks=4, check_jump=False, shuf_per=False, n_shuf_runs=5000, make_shuf_plot=False, shuf_dir='shuf_plots', name_shuf_plot='example_shuf_plot.png'):
    '''Run Lomb-Scargle periodogram and return a dictionary of results.

    parameters
    ----------
    lc_data : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous sections
    lc_type : `str`, optional, default='reg'
        A label designating whether the lightcurve uses the original or CBV-corrected flux.
    ref_name : `str`, optional, default='targets'
        The reference name for each subdirectory which will connect all output
        files.
    pg_dir : `string`, optional, default='pg'
        The name of the directory to store the periodogram data
    name_pg : `string`, optional, default='pg_target'
        A file name which the periodogram output will be saved to.
    n_sca : `int`, optional, default=10
        The number of evenly-split lightcurve parts used to measure the flux scatter.
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : `int`, optional, default=10
        The number of samples to measure in each periodogram peak.
    n_peaks : `int`, optional, default=4
        The number of peaks calculated in get_periodogram_peaks
    check_jump : `bool`, optional, default=False
        Choose to check the lightcurve for jumpy data, using the "check_for_jumps"
        function.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    n_shuf_runs : `int`, optional, default=5000
        The number of measurements to be made in shuffle_periodogram
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='shuf_plots'
        The name of the directory to save the plots of the shuffled period analysis. 
    name_shuf_plot : `str`, optional, default='example_shuf_plot.png'
        A file name which the plots of the shuffled period analysis will be saved to.

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
        | "period_1" : The period corresponding to the highest peak
        | "power_1" : The power corresponding to the highest peak
        | "Gauss_fit_peak_parameters" : Parameters for the Gaussian fit to the highest power peak
        | "Gauss_fit_peak_y_values" : The corresponding y-values for the Gaussian fit
        | "period_around_1" : The period values covered by the Gaussian fit
        | "power_around_1" : The power values across the period range covered by the Gaussian fit
        | "period_not_1" : The period values not covered by the Gaussian fit
        | "power_not_1" : The power values across the period range not covered by the Gaussian fit
        | "period_2" : The period of the second highest peak.
        | "power_2" : The power of the second highest peak.
        | "period_3" : The period of the third highest peak.
        | "power_3" : The power of the third highest peak.
        | "period_4" : The period of the fourth highest peak.
        | "power_4" : The power of the fourth highest peak.
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

    LS_dict, cln_lc, ls = initialise_LS_dict(lc_data, check_jump=check_jump)
    logger.info(f'LS dictionary successfully initialised: {name_pg}, {lc_type}.')

    write_periodogram(LS_dict, name_pg=name_pg, lc_type=lc_type, pg_dir=pg_dir)
    logger.info(f'Periodogram results written to file.')
        
    get_periodogram_peaks(LS_dict, n_peaks=n_peaks)
    logger.info(f'Top {n_peaks} peaks recorded to dictionary.')

    shuffle_check(cln_lc, LS_dict, shuf_per=shuf_per, n_shuf_runs=n_shuf_runs, p_min=p_min_thresh, p_max=p_max_thresh, n_min=1./10., n_max=1., bin1=50, bin2_fac=10, n_peaks=n_peaks, make_shuf_plot=make_shuf_plot, shuf_dir=shuf_dir, name_shuf_plot=name_shuf_plot)
    logger.info('Periodogram successfully shuffled.')

    make_phase_curve(LS_dict, ls, n_sca=n_sca)
    logger.info(f'Phase curve details stored to dictionary.')
    return LS_dict


__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__], predicate = lambda f: inspect.isfunction(f) and f.__module__ == __name__)]
