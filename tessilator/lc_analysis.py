'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

This module contains functions to normalise, detrend, and clean lightcurves.
    
The cleaning and detrending of the lightcurve takes part in 9 steps, each of
which are called by the parent function 'run_make_lc_steps'. The steps are:

| (1) Normalise the raw lightcurve.
| (2) Split the lightcurve into segments of contiguous data (no large time gaps
      between datapoints)
| (3) Remove sparse data, i.e., data segments with only a few data points.
| (4) Detrend the lightcurve.
| (5) Clean the edges: remove flux outliers.
| (6) Clean the edges: scattered, noisy datapoints.
| (7) Remove extreme outliers from the lightcurve
| (8) Normalise the flux in each segment, using the median of qualified data.
| (9) Store the lightcurve to file.
'''

###############################################################################
####################################IMPORTS####################################
###############################################################################
# Internal
import inspect
import sys


# Third party
import numpy as np
import os
import json

from astropy.table import Table
from astropy.stats import akaike_info_criterion_lsq

from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit

import itertools as it
from operator import itemgetter


# Local application
from .file_io import fix_table_format, logger_tessilator
###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__) 
    
    
    
    
def get_time_segments(t, t_fac=10.):
    '''Split the lightcurve into groups of contiguous data
    
    Group data into segments where the time difference between data points is
    less than some threshold factor, `t_fac`.
    
    parameters
    ----------
    t : `Iter`
        The list of time coordinates from the lightcurve
    t_fac : `float`, optional, default=10.
        A time factor, which is multiplied by the median cadence of the full
        lightcurve.
        
    returns
    -------
    ds : `list`
        The start indices of each time segment
    df : `list`
        The final indices of each time segment
    '''
    td     = np.zeros(len(t))
    td[1:] = np.diff(t)
    t_arr = (td <= t_fac*np.median(td)).astype(int)
    groups = (list(group) for key, group in it.groupby(enumerate(t_arr),
                                                       key=itemgetter(1))
                                                       if key)
    ss = [[group[0][0], group[-1][0]] for group in groups
                                      if group[-1][0] > group[0][0]]
    ss = np.array(ss).T
    ds, df = ss[0,:], ss[1,:]
    ds[1:] = [ds[i]-1 for i in range(1,len(ds))]
    df = np.array([(i+1) for i in df])
    return ds, df



    
def remove_sparse_data(x_first, x_last, min_crit_frac=.05, min_crit_num=50):
    '''Removes very sparse data groups, when there are 3 or more groups.

    Calculate the mean (mean_group) and standard deviation (std_group) for the
    number of data points in each group (n_points).
    If std_group > "std_crit", then only keep groups with n_points > std_crit.
    
    parameters
    ----------
    x_first : `Iterable`
        The index values of the first element in each group
    x_last : `Iterable`
        The index values of the last element each group
    min_crit_frac : `float`, optional, default=.05
        The minimum relative size of a flux component when correcting for
        sparse data in the cleaning functions.
    min_crit_num : `int`, optional, default=50
        The minimum number of data points required for a flux component in the
        sparse data cleaning functions.

    returns
    -------
    y_first : `np.array`
        The index values of the first element of the new arrays
    y_last : `np.array`
        The index values of the last element of the new arrays
    '''
    try:
        n_points = np.array([x_last[i] - x_first[i]
                            for i in range(len(x_first))])
        n_tot = np.sum(n_points)
        std_crit = max(min_crit_num, min_crit_frac*n_tot)
        y_first, y_last = np.array(x_first), np.array(x_last)
        if len(x_first) > 2:
            std_group = np.std(n_points)
            if std_group > std_crit:
                g = n_points > std_crit
                y_first, y_last = y_first[g], y_last[g]
                return y_first, y_last
        return y_first, y_last
    except:
        logger.warning("The sparse data removal algorithm failed. Retaining "
                       "the input indices.")
        return y_first, y_last


    
    
    
def aic_selector(x, y, poly_max=3, cov_min=1e-10):
    '''Select the detrending polynomial from the Aikaike Information Criterion
    
    This function uses the Aikaike Information Criterion (AIC) to find the most
    appropriate polynomial order to a set of X, Y data points.
    
    parameters
    ----------
    x : `Iterable`
        The independent variable
    y : `Iterable`
        The dependent variable
    poly_max : `int`, optional, default=3
        The maximum polynomial order to test
    cov_min : `float`
        A threshold value for the first element of the covariance matrix.
        Sometimes the AIC will automatically select a higher-order
        polynomial to a distribution that is clearly best fit by the
        preceeding lower-order fit. For example, a second-order fit provides a
        better fit for a perfect straight line. This is a bug in the numerical
        rounding. Therefore, if the value of the first element of the
        covariance matrix is less than cov_min for the lower order, then the
        lower order fit is selected.

    returns
    -------
    poly_ord : `int`
        The best polynomial order
    coeffs : `list`
        The polynomial coefficients.
    
    '''
    
    q = 0
    N = 1.*len(x)
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
            if (AIC1 < AIC2) | (r1 < cov_min):
                poly_ord, coeffs = q, p1
                return poly_ord, list(coeffs)
            else:
                q += 1
                if q >= poly_max:
                    poly_ord, coeffs = q, p2
                    return poly_ord, list(coeffs)
    except:
        return 0, [1.0]



def relative_root_mean_squared_error(true, pred):
    '''Return the relative root mean squared error (RRMSE)
    
    Given a list of predicted and true values, calculate the RRMSE
    
    parameters
    ----------
    true : `Iter`
        The set of true (measured) values
    pred : `Iter`
        The set of predicted (model) values
        
    returns
    -------
    rrmse : `float`
        The RRMSE value
    '''
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse = np.sqrt(squared_error)
    return rrmse


def smooth_test(time, flux, n_avg=10):
    '''Determine how to detrend a lightcurve, based on a smoothness algorithm
    
    The idea of this function is to catch lightcurves that appear to have
    periods longer than ~15 days, and are notably smooth. The function
    calculates a sine fit to the (linearly) detrended lightcurve, then
    smoothes it using a running mean. Finally, there are three criteria to
    decide how the lightcurve should be detrended. A boolean flag is returned,
    where False=individual groups and True=the whole lightcurve.
    
    parameters
    ----------
    time : `Iter`
        A set of time coordinates
    flux : `Iter`
        A set of flux coordinates
    n_avg : `float`, optional, default=10
        The number of datapoints to be used for the running mean calculation.
        
    returns
    -------
    smooth_flag : `bool`
        A boolean flag to (partially) determine how the lightcurve should be
        detrended (False=individual groups, True=the whole lightcurve)
    '''

    # 1) detrend the whole lightcurve by a linear fit, and calculate the MAD
    t_new = time-time[0]
    p1, r1, _,_,_ = np.polyfit(t_new, flux, 1, full=True)
    f_new = flux/np.polyval(p1, t_new)
    f_MAD = MAD(f_new, scale='normal')

    # 2) make a sine fit to the detrended lightcurve
    pops, popsc = curve_fit(sin_fit_per, t_new, f_new,
                            bounds=([0.5, 0.0, 0.0, 0.0],
                                    [1.5, 0.2, 100., 2.*np.pi]))
    yp = sin_fit_per(t_new, *pops)
    
    # 3) smooth the arrays using a running mean
    yp_sm = np.array(np.convolve(yp, np.ones(n_avg)/n_avg, mode='valid'))
    flux_sm = np.array(np.convolve(f_new, np.ones(n_avg)/n_avg, mode='valid'))
    t_sm = np.array(np.convolve(t_new, np.ones(n_avg)/n_avg, mode='valid'))
    
    # 4) subtract the smoothed "raw" flux by the smoothed sine fit
    diff_flux = flux_sm - yp_sm
    d_MAD = MAD(diff_flux, scale='normal')

    # 5) calculate the RRMSE between the detrended flux and the sine fit 
    rrmse = relative_root_mean_squared_error(f_new, yp)
    
    # 6) false if:
    #       a) the predicted period from the sine fit is > 15. days
    #       b) the RRMSE is < 0.01
    #       c) the ratio of the MAD between the differential and original flux
    #          is < 0.25   
    if (rrmse < 0.01) & \
       (pops[2] > 15.) & (pops[2] < 99.9) & \
       (d_MAD/f_MAD < 0.25):
        return True
    else:
        return False

       


def norm_choice(t_orig, f_orig, lc_part, MAD_fac=2., poly_max=4):
    '''Choose whether to detrend the lightcurve as one by individual groups.
    
    There are always at least two data components in a TESS sector because of
    the finite time needed for data retrieval. This can sometimes lead to
    discontinuities between components because of TESS systematics and the
    temperature gradients across the photometer. These discontinuities can
    cause the phase of the sinusoidal fit to change, leading to low power
    output in the periodograms. Alternatively, if the data are all detrended
    individually, but the data is relatively continuous, this can lead to
    shorter period measurements.
    
    The idea with this function is that a polynomial fit is made to each
    component (chosen using the Aikaike Information Criterion, AIC). The
    extrapolated flux value from component 1 is calculated at the point where
    component 2 starts. If the difference between this value and the actual
    normalised flux at this point is greater than a given threshold, the data
    should be detrended separately. Otherwise the full lightcurve can be
    detrended as a whole.
    
    In addition, if the "smooth" flag (calculated from the "smooth_test"
    function) is True, then the lightcurve is detrended as a whole, regardless
    of the outcome from this function.
    
    parameters
    ----------
    t_orig : `Iterable`
        The time coordinate
    f_orig : `Iterable`
        The original, normalised flux values
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    MAD_fac : `float`, optional, default=2.
        The factor which is to be multiplied by the median absolute deviation.
    poly_max : `float`, optional, default=4
        The maximum polynomial order to test for the AIC evaluation.        

    returns
    -------
    norm_flag : `bool`
        Determines whether the data should be detrended as one whole component
        (False) or by individual groups (True, providing the smooth flag is
        False).
    smooth_flag : `bool`
         Determines whether any detrending should be performed by testing how
         smooth the lightcurve is.
    f1_at_f2_0 : `list`
        The extrapolated fluxes from group 1, calculated at the start point of
        group 2.
    f2_at_f2_0 : `list`
        The first flux values from group 2
    f1_MAD : `list`
        The MAD fluxes from group 1
    f2_MAD : `list`
        The MAD fluxes from group 2
    '''
    norm_flag = False
    Ncomp = len(np.unique(lc_part))
    smooth_flag = smooth_test(t_orig, f_orig)

    f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD = [], [], [], []
    if Ncomp > 1:
        i = 1
        while i < Ncomp:
            g1 = np.array(lc_part == i)
            g2 = np.array(lc_part == i+1)
            try:
                s_fit1, coeff1 = aic_selector(t_orig[g1], f_orig[g1],
                                              poly_max=poly_max)
                s_fit2, coeff2 = aic_selector(t_orig[g2], f_orig[g2],
                                              poly_max=poly_max)
                f1_at_f2_0.append(np.polyval(coeff1, t_orig[g2][0]))
 # The line below IS supposed to be at index "g2"
                f2_at_f2_0.append(np.polyval(coeff2, t_orig[g2][0]))
                f1_n = f_orig[g1]/np.polyval(coeff1, t_orig[g1])
                f2_n = f_orig[g2]/np.polyval(coeff2, t_orig[g2])
                f1_MAD.append(MAD(f1_n, scale='normal'))
                f2_MAD.append(MAD(f2_n, scale='normal'))
                if 2.*abs(f1_at_f2_0[i-1] - f2_at_f2_0[i-i]) > \
                       MAD_fac*((f1_MAD[i-1]+f2_MAD[i-1])/2.):
                    norm_flag = True
                    break
                else:
                    i += 1
            except:
                logger.error('Could not run the AIC selector, '
                             'probably because of a zero-division.')
                f1_at_f2_0.append(np.polyval([1], t_orig[g2][0]))
                f2_at_f2_0.append(np.polyval([1], t_orig[g2][0]))
                f1_n = f_orig[g1]
                f2_n = f_orig[g2]
                f1_MAD.append(MAD(f1_n, scale='normal'))
                f2_MAD.append(MAD(f2_n, scale='normal'))
                norm_flag = False
                break
    if smooth_flag == True:
        norm_flag = False
    return norm_flag, smooth_flag, f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD







def detrend_lc(t,f,lc, MAD_fac=2., poly_max=3):
    '''Detrend and normalise the lightcurves.

    | This function runs 3 operations to detrend the lightcurve, as follows:
    | 1. Choose whether a zeroth- or first-order polynomial is the best fit to
         the full light-curve, using AIC, and detrend the full lightcurve.
    | 2. Decide whether to use the detrended lightcurve from part 1, or to
         detrend individual groups.
    | 3. Return the detrended flux.

    parameters
    ----------
    t : `Iterable`
        the time component of the lightcurve
    f : `Iterable`
        the flux component of the lightcurve.
    lc : `Iterable`
        The index representing the lightcurve component. Note this
        must be indexed starting from 1.
    MAD_fac : `float`, optional, default = 2.
        The factor to multiply the median absolute deviation by.
    poly_max : `int`, optional, default=8
        The maximum order of the polynomial fit.

    returns
    -------
    f_norm : `Iterable`
        The corrected lightcurve after the detrending procedures.
    detr_dict : `dict`
        A dictionary containing the parameters: norm_flag, smooth_flag,
        "f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD" (see norm_choice) and
        "s_fit, coeffs" (see aic_selector)
    '''

    # 1. Choose the best detrending polynomial using the Aikaike Information
    #    Criterion, and detrend the lightcurve as a whole.
    s_fit_0, coeffs_0 = aic_selector(t, f, poly_max=poly_max)
    f_norm = f/np.polyval(coeffs_0, t)

    # 2. Decide whether to use the detrended lightcurve from part 1, or to
    #    separate the lightcurve into individual components and detrend each
    #    one separately
    norm_flag, smooth_flag, f1_at_f2_0, f2_at_f2_0, f1_MAD, f2_MAD = \
                    norm_choice(t, f, lc, MAD_fac=MAD_fac, poly_max=poly_max)
    # 3. Detrend the lightcurve following steps 1 and 2.
    s_fit, coeffs = [], []
    if norm_flag:
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
        f_norm = f# f_norm
    detr_dict = {'norm_flag' : norm_flag,
                 'smooth_flag' : smooth_flag,
                 'f1_at_f2_0' : f1_at_f2_0,
                 'f2_at_f2_0' : f2_at_f2_0,
                 'f1_MAD' : f1_MAD,
                 'f2_MAD' : f2_MAD,
                 's_fit' : s_fit,
                 'coeffs' : coeffs}
    return f_norm, detr_dict




    
    
    
def clean_flux_algorithm(g):
    '''A basic algorithm that trims both sides of a contiguous data string
    if a condition is not satisfied, until the condition is met for the
    first time.
    
    parameters
    ----------
    g : `Iter`
        The outcome for each datapoint, qualified (=1) or not qualified (=0)
    
    returns
    -------
    first : `int`
        The trimmed first point of the array
    last : `int`
        The trimmed last point of the array
    '''
    i, j = 0, len(g)-1
    while i < j:
        if g[i] != 1:
            i+=1
        else:
            first=i
            break
    while j > 0:
        if g[j] != 1:
            j-=1
        else:
            last=j
            break
    if j <= i:
        first, last = 0, len(g)-1
        return first, last
    else:
        return first, last
    
    

    
def clean_edges_outlier(f, MAD_fac=2.):
    '''Remove spurious outliers at the start and end parts of groups.

    The start and end point of each group must have a flux value within a given
    number of MAD from the median flux in the group. This is done because
    during data downlinks, the temperature of the sensors can change notably.
    Therefore the outlying flux points at the group edges are probably from
    temperature instabilities. 

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.

    returns
    -------
    first : `int`
       The start index for the data string.
    last : `int`
       The end index for the data string.
    '''
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    try: 
        g = (np.abs(f-f_med) < MAD_fac*f_MAD).astype(int)
        first, last = clean_flux_algorithm(g)
        
    except:
        logger.error('Something went wrong with the arrays when doing the '
                     'lightcurve edge clipping')
        first, last = 0, len(g)-1
    return first, last


def clean_edges_scatter(f, MAD_fac=2., len_sub_raw=11, num_data_fac=0.1):
    '''Remove highly-scattered data at the edges of each group.

    Some groups have very scattered fluxes at the edges, presumably because the
    sensors are unstable before and after data downlinks. These can degrade the
    quality of the periodogram analysis, or even lead to an incorrect period.
    
    The idea is to group the first "n_sub" datapoints, and calculate the median
    absolute deviation (MAD). If this local MAD value is greater (less) than
    "MAD_fac" times the MAD of the full lightcurve, then the flag at this point
    is 0 (1). The first and last "(n_sub-1)/2" in the lightcurve are given a
    constant value. If the first/last MAD comparison yield a "1" value, then we
    include the full group, including the datapoints replaced with constant
    values -- i.e., no cleaning is necessary.
    
    The value for n_sub is chosen as the minimum value of "len_sub_raw", or
    num_data_fac*(the number of datapoints in the whole set).

    parameters
    ----------
    f : `Iterable`
        The set of normalised flux coordinates
    MAD_fac : `float`, optional, default=2.
        The threshold number of MAD values to allow.
    len_sub_raw : `int`, optional, default=11
        The number of data points to be used in the local MAD value.
    num_data_fac : `float`, optional, default=0.1
        The factor to multiply the number of data points by.

    returns
    -------
    first : `int`
       The start index for the data string.
    last : `int`
       The end index for the data string.
    '''
    n_sub =min(len_sub_raw, int(num_data_fac*len(f)))
    if n_sub // 2 == 0:
        n_sub += 1
    p_e = int((n_sub-1)/2)
    # get the median time and flux, the median absolute deviation in flux
    # and the time difference for each neighbouring point.
    f_med, f_MAD = np.median(f), MAD(f, scale='normal')
    f_diff = np.zeros(len(f))
    f_diff[1:] = np.diff(f)
    f_diff_med = np.median(np.absolute(f_diff))
    f_x = np.array([MAD(f[i:i+n_sub], scale='normal')
                    for i in range(len(f)-n_sub+1)])
    f_diff_run = np.pad(f_x, (p_e, p_e), 'constant',
                        constant_values=(MAD_fac*f_MAD, MAD_fac*f_MAD))

    try:
        g = (np.abs(f_diff_run) < MAD_fac*f_diff_med).astype(int)
        first, last = clean_flux_algorithm(g)
        if first <= p_e:
            first = 0
        elif first > p_e:
            first = np.where(g)[0][p_e]
        if last >= len(g)-1-(2*p_e+1):
            last = len(g)-1
        elif last < len(g)-1-(2*p_e+1):
            last = np.where(g)[0][-p_e]
    except:
        logger.error('Something went wrong with the arrays when doing the '
                     'lightcurve edge clipping')
        first, last = 0, len(g)-1
    return first, last




def run_make_lc_steps(f_lc, f_orig, min_crit_frac=0.1, min_crit_num=50,
                      outl_mad_fac=3.):
    '''Process the lightcurve: cleaning, normalisation and detrending functions
    
    | During each procedure, the function keeps a record of datapoints that are
    | kept or rejected, allowing users to assess the amount of data loss.
    
    | The function makes the following steps...
    | 1. normalise the original flux points
    | 2. split the lightcurve into 'time segments'
    | 3. remove very sparse elements from the lightcurve
    | 4. run the first detrending process to pass to the cleaning function.
    | 5. clean the lightcurve edges from outliers
    | 6. clean the lightcurve edges from scattered data
    | 7. finally cut out data that are extreme outliers.
    | 8. divide each lightcurve component by the median flux value
    | of qualifying data points.
    | 9. return the dictionary

    parameters
    ----------
    f_lc : `dict`
        The initial lightcurve with the minimum following keys required:
        (1) 'time' -> the time coordinate
        (2) 'eflux' -> the error in the flux
        (3) 'f_orig' -> see the f_orig parameter
    f_orig : `str`
        This string determines which of the original flux values to choose.
        It forms the final part of the f_lc keys.
        It could be either 'reg_oflux' (the regular, original flux) or
        'cbv_oflux' (the original flux corrected using co-trending basis
        vectors)
    min_crit_frac : `float`, optional, default=0.1
        The minimum relative size of a flux component when correcting for
        sparse data in the cleaning functions.
    min_crit_num : `int`, optional, default=50
        The minimum number of data points required for a flux component in the
        sparse data cleaning functions.
    outl_mad_fac : `float`, optional, default=3.
        The factor of MAD for the cleaned lightcurve flux values.
        
    returns
    -------
    f_lc : `dict`
        A dictionary storing the full set of results from the lightcurve
        analysis.
        As well as the keys from the inputs, the final keys returned are:
        1: "time" -> the time coordinate.
        2: "mag" -> the TESS magnitude.
        3: "(reg/cbv)_oflux" -> the flux calculated from aperture photometry.
        4: "eflux" -> the error bar on (reg/cbv)_oflux.
        5: "nflux_ori" -> the normalised fluxes from (3).
        6: "nflux_err" -> the error bars on (5).
        7: "nflux_dtr" -> the normalised fluxes after the detrending steps.
        8: "lc_part" -> an index referring to each group in the lightcurve.
        9: "pass_sparse" -> boolean from `remove_sparse_data`
        10: "pass_clean_outlier" -> boolean from clean_edges_outlier.
        11: "pass_clean_scatter" -> boolean from clean_edges_scatter.
        12: "pass_full_outlier" -> boolean from the final outlier rejection.
    detr_dict : `dict`
        The dictionary returned from `detrend_lc`
    '''

    # (1) normalise the original flux points
    f_lc['nflux_ori'] = f_lc[f'{f_orig}']/np.median(f_lc[f'{f_orig}'])
    f_lc['nflux_err'] = f_lc['eflux']/f_lc[f'{f_orig}']
    logger.info('part1: initial normalisation -> done!')

    # (2) split the lightcurve into 'time segments'
    ds1, df1 = get_time_segments(f_lc["time"])
    logger.info('part2: time segmentation -> done!')

    # (3) remove very sparse elements from the lightcurve
    ds2, df2 = remove_sparse_data(ds1, df1, min_crit_frac=min_crit_frac, 
                                  min_crit_num=min_crit_num)
    f_lc["pass_sparse"] = np.array(np.zeros(len(f_lc["time"])), dtype='bool')
    for s, f in zip(ds2, df2):
        f_lc["pass_sparse"][s:f] = True
    logger.info('part3: remove sparse data -> done!')

    # (4) run the first detrending process to pass to the cleaning function.
    f_lc["lc_part"] = np.zeros(len(f_lc["time"]), dtype=int)
    for i, (s, f) in enumerate(zip(ds2, df2)):
        f_lc["lc_part"][s:f] = int(i+1)
    g_cln = f_lc["pass_sparse"]
    f_lc["nflux_dtr"] = np.full(len(f_lc["time"]), -999.)
    f_lc["nflux_dtr"][g_cln], detr_dict = detrend_lc(f_lc["time"][g_cln],
                                                     f_lc["nflux_ori"][g_cln],
                                                     f_lc["lc_part"][g_cln],
                                                     poly_max=1)
    logger.info('part4: detrending -> done!')
    # (5) clean the lightcurve edges from outliers
    ds3, df3 = [], []
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        s_o, f_o = clean_edges_outlier(f_lc["nflux_dtr"][g])
        ds3.append(g[s_o])
        df3.append(g[f_o])
    f_lc["pass_clean_outlier"] = np.array(np.zeros(len(f_lc["time"])),
                                          dtype='bool')
    for s, f in zip(ds3, df3):
        f_lc["pass_clean_outlier"][s:f] = True
    logger.info('part5: clean edges, outliers -> done!')

    # (6) clean the lightcurve edges from scattered data
    ds4, df4 = [], []
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        s_s, f_s = clean_edges_scatter(f_lc["nflux_dtr"][g])
        ds4.append(g[s_s])
        df4.append(g[f_s])
    f_lc["pass_clean_scatter"] = np.array(np.zeros(len(f_lc["time"])),
                                          dtype='bool')
    for s, f in zip(ds4, df4):
        f_lc["pass_clean_scatter"][s:f] = True
    logger.info('part6: clean edges, scatter -> done!')

    # (7) finally cut out data that are extreme outliers.
    med_lc = np.median(f_lc["nflux_dtr"][g_cln])
    MAD_lc = MAD(f_lc["nflux_dtr"][g_cln], scale='normal')
    f_lc["pass_full_outlier"] = np.array(np.zeros(len(f_lc["time"])),
                                         dtype='bool')
    for f in range(len(f_lc["time"])):
        if abs(f_lc["nflux_dtr"][f] - med_lc) < outl_mad_fac*MAD_lc:
            f_lc["pass_full_outlier"][f] = True
    logger.info('part7: remove extreme points -> done!')

    # (8) divide each lightcurve component by the median flux value of
    #     qualifying data points.
    for lc in np.unique(f_lc["lc_part"][g_cln]):
        g = np.where(f_lc["lc_part"] == lc)[0]
        gx = np.logical_and.reduce([
                   f_lc["pass_sparse"][g], 
                   f_lc["pass_clean_scatter"][g],
                   f_lc["pass_clean_outlier"][g],
                   f_lc["pass_full_outlier"][g]
                   ])
        flux_vals = f_lc["nflux_dtr"][g[gx]]
        med_flux = np.median(flux_vals[flux_vals > 0.])

    f_lc["nflux_dtr"][f_lc["nflux_dtr"] < 0] = -999
    logger.info('part8: write the dictionary -> done!')

    # (9) return the dictionary
    logger.info('part9: FINISHED!')
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



def sin_fit_per(t, y0, A, per, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase) to a regular
    sinusoidal function.

    parameters
    ----------
    t : `Iterable`
        list of input values
    y0 : `float`
        The midpoint of the sine curve
    A : `float`
        The amplitude of the sine curve
    per : `float`
        The period of the sine curve
    phi : `float`
        The phase angle of the sine curve

    returns
    -------
    sin_fit_per : `list`
        A list of sin curve values.
    '''
    sin_fit_per = y0 + A*np.sin((2.*np.pi*t/per) + phi)
    return sin_fit_per






def cbv_fit_test(t, of, cf):
    '''Determine whether the cbv-corrected lightcurve should be considered.
    
    Whilst the cbv-corrected flux are designed to eliminate systematic
    artefacts by identifying features common to many stars (using principle 
    component analysis), the routine can overfit the data, and often the cbv
    corrections inject too much unwanted noise (particularly for targets with
    low signal to noise).
    
    Therefore the plan here is to assess lightcurves produced by the cbv
    corrections by comparing basic attributes with the regular (non-corrected)
    lightcurves. These scores come down to:
    
    1: the number of outliers
    2: the size of the median absolute deviation
    3: which lightcurve provides the lowest chi-squared value to a sine fit.
    
    If the "original lightcurve" scores higher, then the cbv-corrected
    lightcurve is not considered for further analysis.
    
    parameters
    ----------
    t : Iterable
        The time component of the lightcurve
    of : Iterable
        The original flux
    cf : Iterable
        The cbv-corrected flux
    
    returns
    -------
    use_cbv : bool
        True if cf score > of score, else False.
    '''
    
    of_score, cf_score = 0, 0

#1) number of outliers test
    of_nflux = np.array(of)/np.median(of)
    cf_nflux = np.array(cf)/np.median(cf)
    of_nMADf = MAD(of_nflux, scale='normal')
    cf_nMADf = MAD(cf_nflux, scale='normal')
    num_of = np.sum(abs(of_nflux-1.) > of_nMADf)
    num_cf = np.sum(abs(cf_nflux-1.) > cf_nMADf)
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
        logger.error('Could not do the sine-fit comparison for ori vs cbv '
                     'lightcurves')
# get the final score - if cbv wins, then a True statement is returned.    
    if of_score >= cf_score:
        use_cbv = False
    else:
        use_cbv = True
    return use_cbv





def make_lc(phot_table, name_lc='target', store_lc=False, lc_dir='lc', cbv_flag=False):
    '''Construct the normalised, detrended, cleaned TESS lightcurve.
    
    This is essentially a parent function that performs all the steps in fixing
    the lightcurve.
    
    The returned product is an array containing the new tabulated lightcurve
    data for the original (unfiltered) aperture photometry, and (if necessary)
    another one for the CBV-corrected fluxes (see the 'cbv_fit_test' function
    for more information.)

    parameters
    ----------
    phot_table : `astropy.table.Table` or `dict`
        | The data table containing aperture photometry. Columns must include:
        | "time" -> The time coordinate for each image
        | "mag" -> The target magnitude
        | "(reg/cbv)_oflux" -> The total flux subtracted by the background flux
        | "flux_err" -> The error on flux_corr
    name_lc : `str`, optional, default='target'
        The name of the file which the lightcurve data will be saved to.
        The target name
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True
    cbv_flag : `bool`, optional, default=False
        Choose whether to analyse the lightcurves for CBV-corrected data.

    returns
    -------
    final_tabs : `list`
        A list of tables containing the lightcurve data
        These are for the original lightcurve, and the cbv-corrected lightcurve
        if required and it satisfies the criteria from cbv_fit_test.
    norm_flags : `list`
        A list of norm_flag values from the detrending algorithm.
    smooth_flags : `list`
        A list of smooth_flag values from the detrending algorithm.
    '''
    logger.info(f'Running the lightcurve analysis for {name_lc}')
    f_labels = ['reg_oflux']
    cbv_ret = False
    
    if cbv_flag:
        if "cbv_oflux" in phot_table.colnames:
            f_labels.append('cbv_oflux')
            use_cbv = cbv_fit_test(phot_table["time"], phot_table["reg_oflux"],
                                   phot_table["cbv_oflux"])
            if use_cbv:
                cbv_ret = True
    final_tabs, norm_flags, smooth_flags = [], [], []
    for f_label in f_labels:
        logger.info(f'using the flux label: {f_label}')
        final_lc = {}
        final_lc["time"] = phot_table["time"].data
        final_lc["mag"] = phot_table["mag"].data
        final_lc[f'{f_label}'] = phot_table[f'{f_label}'].data
        final_lc["eflux"] = phot_table["flux_err"].data
        flux_dict, detr_dict = run_make_lc_steps(final_lc, f_label)
        norm_flag = detr_dict["norm_flag"]
        smooth_flag = detr_dict["smooth_flag"]

        keyorder = ['time','mag',f_label,'eflux','nflux_ori','nflux_err',
                    'nflux_dtr','lc_part','pass_sparse','pass_clean_outlier',
                    'pass_clean_scatter','pass_full_outlier']
        tab_format = ['.6f','.6f','.6f','.6f',
                      '.6f','.4e','.6f','%i',
                      '%s','%s','%s','%s']
        flux_dict = {k: flux_dict[k] for k in keyorder}
        if len(flux_dict["time"]) > 50:
            flux_tab = fix_table_format(Table(flux_dict), keyorder, tab_format)
#            flux_tab = Table(flux_dict)
#            for k, f in zip(keyorder, tab_format):
#                flux_tab[k].info.format = f
#            if f_label == "reg_oflux":
            final_tabs.append(flux_tab)
            norm_flags.append(norm_flag)
            smooth_flags.append(smooth_flag)
#            if (f_label == "cbv_oflux") and (cbv_ret):
#                final_tabs.append(flux_tab)
#                norm_flags.append(norm_flag)
#                smooth_flags.append(smooth_flag)
            if store_lc:
                path_exist = os.path.exists(f'./{lc_dir}')
                if not path_exist:
                    os.makedirs(f'./{lc_dir}')
                flux_tab.write(f'./{lc_dir}/{name_lc}_{f_label}.csv',
                               format='csv', overwrite=True)
                with open(f'./{lc_dir}/{name_lc}_{f_label}.json', 'w') \
                     as convert_file:
                    convert_file.write(json.dumps(detr_dict))
    return final_tabs, norm_flags, smooth_flags



__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__],
           predicate = lambda f: inspect.isfunction(f)
           and f.__module__ == __name__)]
