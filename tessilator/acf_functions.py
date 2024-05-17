# Core SpinSpotter Library
# Created: April 5, 2021
# Last updated: June 2022 
# (or possibly more recently, I honestly forget to update this line often)

# imports
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# UNIVERSAL VARIABLES

# # data points per day at even cadence (=24*30 for TESS, 24*2 for Kepler)
# cad_tess = 24.*30.
# cad_kepler = 24.*2.

cadence = 30*60    # default cadence is 30 minutes (in units of seconds)

# number of seconds in a day
day = 24*60*60


# GENERAL HELPER FUNCTIONS
def days_to_bins(days, bs):
    """
    Converts a time array given in units of days into units of bin number.

    Args:
        days (:obj:`array`): time array in units of days.
        bs (:obj:`float`): the cadence to bin to, in units of seconds.

    Returns:
        :obj:`array`: the time array in units of bin number.
    """
    return days*24*60*60//bs

def bins_to_days(bins, bs):
    """
    Converts a time array given in units of bin number into units of days.

    Args:
        bins (:obj:`array`): time array in units of bin number or (in the case of the acf) lag number.
        bs (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
        :obj:`array`: the time array in units of days.
    """
    return bins*bs/(24*60*60)


# NEW PROCESS HELPER FUNCTIONS


def calc_fft_pgi(corr, bs=cadence):    
    """
    Uses a fast fourier transform to identify an initial guess (pgi) for the dominant periodicity in the autocorrelation function. This is the default pgi-finding function for SpinSpotter.

    Args:
        corr (:obj:`arr`): an autocorrelation function.
        bs (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
    Two parameters.

        - pgi (:obj:`int`): the index in the acf of the initial period guess in units of lags.
        - results (:obj:`array`): dictionary of other relevant paramters from the pgi-finding code.
    """
    results = {}

    # Calculate the fft
    coeff = 10
    corr_fft = np.real(np.fft.rfft(corr,n=coeff*len(corr)))
    fft_period = np.divide(1., np.fft.rfftfreq(coeff*len(corr))[1:])

    # we're only interested in the positive half of the fft
    snip = np.where((fft_period >= 0) & (fft_period <= len(corr)))
    fft_period = fft_period[snip]
    corr_fft = corr_fft[snip]
    
    # add some results to the results dictionary
    results['fft'] = corr_fft
    results['fft_period'] = fft_period  # period array, in units of index of the acf
    results['fft_period_days'] = np.divide(fft_period, (day/bs))  # convert the periods to days
    # results['pgi'] = pgi

    # add the prominence of the fft peak
    pgi, fft_results = fft_find_peaks(results, plot=False)
    results.update(fft_results) 
    
    # if the pgi is more than half the period, automatic fail
    if pgi < 0 :
        results['fail'] = True
        return np.nan, results
    elif fft_period[pgi] > len(corr) / 2:
        results['fail'] = True
        return np.nan, results
    
    return pgi, results

def fft_find_peaks(result, plot=False):
    """
    Helper function used by `calc_fft_pgi` to calculate the prominence of the max 
    peak in the fft.Does this by finding the five tallest peaks, then calculates the standard 
    deviation of the tallest one from the other four. setting plot to True will plot the FFT and
    the five found peaks. Currently, returns the number of standard deviations from
    the mean that the highest peak is.

    Args:
        result (:obj:`dict`):the result dictionary as outputed by calc_fft_pgi(), which contains the fft.
        plot (:obj:`bool`): if True, will plot the FFT.

    Returns:
        results (:obj:`dict`): the updated result dictionary, which now contains info on the pgi and peaks in the fft.
    """
    output = result.copy()
    corr_fft = np.real(result['fft'])
    fft_period = result['fft_period']  # period array, in units of index of the acf

    # find the five tallest peaks, then see how much taller the tallest is
    fft_peaks, fft_properties = find_peaks(corr_fft)
    peak_heights = corr_fft[fft_peaks]

    #initialize the matrix to store the indices of the five peaks
    max_peaks = np.zeros(6,dtype=int)

    # if the max peak is too small, fail the test and exit
    # currently set to cut off when the max peak has index less than 5
    if fft_peaks[np.argmax(peak_heights)] < 5 :
        # print( "period is too short")
        return -1, {'pgi':-1, 'fft_pgi':-1}
    
    # retrieve the peaks
    for i,val in enumerate(max_peaks) :
        index = np.argmax(peak_heights)
        max_peaks[i] = int(fft_peaks[index])
        peak_heights[index] = 0
    
    # add the max_peaks and peak_heights to the results dictionary
    output['max_peaks'] = max_peaks
    output['peak_heights'] = peak_heights
    
    # the pgi is the index of the largest peak
    # pgi = int(np.round(fft_period[max_peaks[0]]))
    fft_pgi = max_peaks[0]  # index of the period guess in the fft
    acf_pgi = int(np.round(fft_period[max_peaks[0]]))  # index of the period guess in the acf
    
    # calculate the mean and std of all but the largest peak
    mean_mp = np.mean(corr_fft[max_peaks[1:]])
    std_mp = np.std(corr_fft[max_peaks[1:]])

    # calc how many stds the max peak is from the rest
    pgi_prom = (corr_fft[max_peaks[0]] - mean_mp) / std_mp

    # add the pgi and peak prominence to the result dict
    output['fft_pgi'] = fft_pgi
    output['pgi'] = acf_pgi
    output['pgi_prom'] = pgi_prom
    
    # if requested, then plot
    if plot :
        plt.figure(figsize=[12,5])
        plt.plot(fft_period,corr_fft)
        plt.plot(fft_period[max_peaks],corr_fft[max_peaks], marker='x',linestyle='')

    return acf_pgi, output



def gaussian(fwhm):
    """
    Creates a gaussian with a given FWHM as preparation for convolution with a light curve.

    Args:
        fwhm (:obj:`float`): the full width half max of the desired gaussian.
        bs (:obj:`float`): the size of the bins, in units of seconds.

    Returns:
        gaussian (:obj:`arr`): a gaussian.
    """
    sigma = fwhm / 2.355
    x = np.arange(-3*sigma, 3*sigma)
    # note that we divide by .997 to preserve the normalization and make the
    # area under the truncated gaussian equal to 1
    return 1./.997 * 1./(np.sqrt(2.*np.pi) * sigma) * np.exp(-(x/sigma)**2./2.)

def parabola(x,a,b,c):
    """
    Creates a parabola of the form y = ax^2 + bx + c. Intended to be used by curvefit in the process Test.

    Args:
        x (:obj:`arr`): the x range of the function.
        a, b, c (:obj:`float`): coefficients

    Returns:
        y (:obj:`arr`): a parabola.
    """
    return a*np.square(x) + b*x + c

def curvefit_peak(func,corr,pgi,peak_num,plot=False):
    """
    Bins a timeseries to the desired cadence. Works much faster than Lightkurve's built in binning function.

    Args:
        func (:obj:`func`): the function to be fitted to peaks in the ACF, intended to be the `parabola()` function.
        corr (:obj:`1D array`): the autocorrelation function.
        pgi (:obj:`int`): period_guess_index, i.e. the intial guess for the index location of the first peak in the `corr` array (aka the acf), usually identified from the highest peak in the FFT of the ACF.
        peak_num (:obj:`int`): which alias you are trying to fit (1 being the original peak, 2 being the first alias.)
        plot (:obj:`bool`, optional): if True, will plot the ACF and the fitted parabolas.

    Returns:
    Two parameters, or three if a flux_err is also provided.

        - fit_params (:obj:`array`): [a0,b0,c0], the coeffitients of the fitted parabola.
        - R_adj (:obj:`float`): the adjusted R^2 value of the parabola fit.
        - fitted_parabola (:obj:`array`): y-values of the fitted parabola.
        - hwhm (:obj:`float`): the half-width half-max of the fitted parabola. Used in error calculations.
    """
    # clip the acf to a length of a period guess, centered on the period guess
    k_snip = np.arange(pgi*peak_num - pgi/4, pgi*peak_num + pgi/4).astype(int)
    corr_snip = corr[k_snip]
    x_snip = np.arange(len(k_snip)).astype(int)  # synthetic x-axis
    
    # do a regression to fit the function
    # make an array for the weights
    # gauss = 1.0 - gaussian(len(x_snip)/2.4)
    # start = (len(gauss)-len(x_snip))/2
    # weights = gauss[start:(start+len(x_snip))]
    # params_opt stands for optimal parameters
    [a0,b0,c0], pcov = curve_fit(func, x_snip, corr_snip)
    # [a0,b0,c0], pcov = curve_fit(func, x_snip, corr_snip, sigma=weights, absolute_sigma=True)

    # calculate the expected curve from the fit parameters
    # note the the output is in lags, need to do conversions to get period in days again
    fitted_parabola = func(x_snip, *[a0,b0,c0])

    # calculate the other params you want
    # note correction to P_rot0 to make up for the shift in the window that was fit
    P_rot0 = -1.*b0 / (2*a0) + 3*pgi/4.   # note that P_rot0 is in lags, NOT time units
    A0 = c0 - np.square(b0) / (4*a0)
    
    # B0 = -1. * a0 * np.square(P_rot0)
    # new value of B0 is the width of the parabola at the zero crossing    
    # to avoid warning, check if there is an intercept    
    if (b0**2 - 4*a0*c0) < 0 :
        intercept1 = np.nan 
        intercept2 = np.nan
    else :
        intercept1 = (-b0 + np.sqrt(b0**2 - 4*a0*c0)) / (2 * a0)
        intercept2 = (-b0 - np.sqrt(b0**2 - 4*a0*c0)) / (2 * a0)
    B0 = np.abs((intercept1 - intercept2) / P_rot0)
    
    # gather things up
    fit_params = [a0,b0,c0]
    peak_params = [P_rot0, A0, B0]

    # check how good the fit is with the adjusted R^2 statistic
    R_adj = adjusted_R_sq(corr_snip, fitted_parabola)

    # calculate the half-width half-max of the peak
    if np.isnan(intercept1) or np.isnan(intercept2) :
        hwhm = np.nan
    else :
        x1 = (-b0 + np.sqrt(b0**2 - 4*a0*(c0-A0/2))) / (2 * a0)
        x2 = (-b0 - np.sqrt(b0**2 - 4*a0*(c0-A0/2))) / (2 * a0)
        hwhm = np.abs(x2-x1) / 2

    # plot if requested
    if plot:
        if peak_num == 1:
            plt.figure(figsize=[9,5])
            plt.plot(corr[:(len(corr))])
        plt.plot(k_snip,fitted_parabola)
        
    # # print the resulting params and R_adj if desired
    # if print_result:
    #     print( "Optimal params: " + str(params_opt))
    #     print( "Adjusted R^2 for curvefit: " + str(R_adj))

    return fit_params, peak_params, R_adj, fitted_parabola, hwhm

def adjusted_R_sq(obs,exp,num_param=3):
    """
    Calculates the adjusted R^2 statistic to estimate the success of a model. Formula implemented from equation 7.62 in Modern Statistical Methods for Astronomy by Feigelson and Babu.

    Args:
        obs (:obj:`float`): the observed value.
        exp (:obj:`float`): the expected value.
        num_param (:obj:`int`): the number of parameters used in the fit.

    Returns:
        R_adj (:obj:`float`): the adjusted R^2 statistic.
    """
    if len(obs) != len(exp):
        print( "The length of the observed and expected arrays should be equal.")
    n = len(obs)
    exp_mean = np.mean(obs) / float(n)

    # calculate the R^2 statistic first
    numerator = 0.
    denominator = 0.
    for i in range(n):
        numerator += np.square(obs[i] - exp[i])
        denominator += np.square(obs[i] - exp_mean)
    R_sq_inv = np.divide(numerator, denominator) # this is actually 1-R^2, not R^2

    # now calculate the adjusted R^2 to take into account the number of model parameters
    R_adj = 1. - (n - 1.)/(n - num_param) * R_sq_inv
    return R_adj


# # NEW PROCESS FUNCTIONS

def calc_acf(lc, bs=cadence, max_lag=None, smooth=None, sector_label=False):
    """
    Caculates the autocorrelation function (ACF) of a light curve..

    Args:
        lc (:obj:`LightCurve obj`): the cleaned lightcurve.
        bs (:obj:`float`): the size of the bins, in units of seconds.
        max_lag (:obj:`float`, optional): the maximum lag to which to calculate the ACF in units of days.
        smooth (:obj:`int`, optional): if supplied, will apply smoothing to the LC before fitting parabolas by convolving with a gaussian with a FWHM equal to this value.
        sector_label (:obj:`int` or :obj:`str`, optional): tallows you to set a custom sector label, must be castable to a string

    Returns:
        fits_result (:obj:`dict`): A dictionary containing information on the light curve and it's ACF.
    """
    fits_result = {'time_even':lc['time'], 'flux_even':lc['nflux_dtr'], 'flux_err_even':lc['nflux_err'],
       'acf_lags':np.array([]), 'acf':np.array([]), 'acf_smooth':np.array([])}

    # set the nlags
    if max_lag is None :
        nlags = days_to_bins(np.floor(lc['time'][-1] - lc['time'][0]), bs)
    else :
        nlags = days_to_bins(max_lag, bs)

    # calculate acf
    acf_corr = acf(lc['nflux_dtr'], missing='conservative',nlags=nlags,fft=True)
    lag_times = bins_to_days(np.arange(len(acf_corr)), bs)
    
    # try convolving with a gaussian
    if isinstance(smooth, int) :
        acf_smooth = np.convolve(acf_corr, gaussian(smooth), mode="same")
    else :
        acf_smooth = np.array([])

    # update fits_result
    fits_result['acf_lags'] = lag_times
    fits_result['acf'] = acf_corr
    fits_result['acf_smooth'] = acf_smooth

    return fits_result

def calc_parabolas(corr, TICID=None, bs=cadence, smooth=None, prot_prior_func=calc_fft_pgi, prot_prior_func_kwargs={}):
    """
    Calculates the best fit parabolas to peaks in an acf.

    Args:
        corr (:obj:`1D array`): the autocorrelation function.
        bs (:obj:`float`): the size of the bins, in units of seconds.
        TICID (:obj:`int` or :obj:`str`, optional): the ID for the object.
        smooth (:obj:`int`, optional): if supplied, will apply smoothing to the LC before fitting parabolas by convolving with a gaussian with a FWHM equal to this value.
        prot_prior_func (:obj:`func`, optional): the function to be used to idntify the period_guess_index (pgi) for the rotation period. Defaults to `calc_fft_pgi()`
        prot_prior_func_kwargs (:obj:`dict`, optional): keyword arguments for the `prot_prior_func` function.

    Returns:
        results (:obj:`dict`): a dictionary of info on the parabola fits.
    """
    # make a dictionary to store results in 
    # any key ending with _k is an array length 5, where the value at each index
    # is associated with a fit to a different peak in the acf. fitted_parabola_k is an
    # array of arrays, each of which is the parabola fit to one of the peaks.
    # 'fail', when set to True, indicates that the test could not be completed
    # due to finding a pgi greater than half the sample length.
    # 'half_period' describes whether there peaks in the ACF at half periods due to
    # having spots in opposite hemispheres. 'half_period_check' means that the peak
    # height difference is less than 5% and needs to be checked by hand.
    results = {'smooth':smooth, 'fft':None, 'fft_period':None, 'pgi':np.nan,
               'a_k':np.array([]), 'b_k':np.array([]), 'c_k':np.array([]), 
               'Rsq_k':np.array([]), 'hwhm_k':np.array([]), 'fitted_parabola_k':[[],[],[],[],[]],
               'P_k':np.array([]), 'A_k':np.array([]), 'B_k':np.array([]), 
               'P_avg':np.nan, 'A_avg':np.nan, 'B_avg':np.nan, 'R_avg':np.nan, 'fft_prom':np.nan,
               'P_err':np.nan,
               'half_period': False, 'half_period_check':False,
               'fail':False }
        
    # if smoothing of the acf is requested, apply it
    if smooth :
        corr_smooth = np.convolve(corr, gaussian(smooth), mode="same")
        corr = corr_smooth
    
    # if the pgi is given as a number, use that
    if (type(prot_prior_func)==float) or (type(prot_prior_func)==int):
        # the pgi will be the index of the acf_lags closes to the provided number in days
        prot_prior_lags = int(days_to_bins(prot_prior_func, bs))
        pgi = min(range(len(corr)), key=lambda i: abs(range(len(corr))[i]-prot_prior_lags))
        results['pgi'] = pgi
    else :
        # Calculate the pgi (initial guess for the period)
        # by default, this will use the funciton calc_fft_pgi, which selects the highest peak in the 
        # FFT of the ACF. You can also write a custom pgi-finding function and pass it in to process_test_raw
        pgi, pgi_results = prot_prior_func(corr, bs=bs, **prot_prior_func_kwargs)
        
        # update the results dictionary
        results.update(pgi_results)
        results['pgi'] = pgi
     
    # run curvefit_peaks for the first peak and up to four aliases
    for peak_num in range(1,6) :
        # check that the alias won't extend beyond end of the acf
        if peak_num*pgi + pgi/4 < len(corr) :
            # run curvefit
            fit_params, peak_params, R_adj, fitted_parabola, hwhm = curvefit_peak(parabola, corr, pgi, peak_num)
            
            # add results to the appropriate dictionary
            results['a_k'] = np.append(results['a_k'], fit_params[0])
            results['b_k'] = np.append(results['b_k'], fit_params[1])
            results['c_k'] = np.append(results['c_k'], fit_params[2])
            results['P_k'] = np.append(results['P_k'], peak_params[0])
            results['A_k'] = np.append(results['A_k'], peak_params[1])
            results['B_k'] = np.append(results['B_k'], peak_params[2])
            results['Rsq_k'] = np.append(results['Rsq_k'], R_adj)
            results['hwhm_k'] = np.append(results['hwhm_k'], hwhm)
            results['fitted_parabola_k'][peak_num-1] = fitted_parabola
        else :
            # trim fitted_parabola_k to the appropriate length
            results['fitted_parabola_k'] = results['fitted_parabola_k'][:peak_num]

    # add the averaged values to the results dictionary
    results['A_avg'] = np.nanmean(results['A_k'])
    results['B_avg'] = np.nanmean(results['B_k'])
    results['R_avg'] = np.nanmean(results['Rsq_k'])
    
    # calculate the error bar on P_avg
    if len(results['P_k']) >= 3 :
        results['P_err'] = np.std(results['P_k'])/np.sqrt(len(results['P_k']))
    else :
        results['P_err'] = np.nanmean(results['hwhm_k'])

    
    # Select the rotation period, keeping in mind that there may be spots in opposite hemispheres
    # Check for alternating peak heights in the ACF.
    if len(results['A_k']) > 2 :
        # check if the second peak is higher than the 1st or 3rd by more than 5%
        if results['A_k'][1]*.95 > results['A_k'][0] and results['A_k'][1]*.95 > results['A_k'][2] :
            # this is the unambiguous case, definitely a half period
            try: 
                results['P_avg'] = np.nanmean([results['P_k'][1], results['P_k'][3]]) * 2
            except:
                results['P_avg'] = results['P_k'][1]
            
            results['half_period'] = True
            
        elif results['A_k'][1] > results['A_k'][0] and results['A_k'][1] > results['A_k'][2]:
            # this is the ambiguous case, less than 5% difference in peak height
            # does NOT automatically update P_avg, this will have to be done by hand when checked
            results['P_avg'] = np.nanmean(results['P_k'])
            results['half_period_check'] = True
        else:
            results['P_avg'] = np.nanmean(results['P_k'])
              
    # now, everything in the results dictionary should be taken care of
    return results

def process_LightCurve(lc, bs=cadence, precleaned=False,
                        transit=None,
                        max_lag=None, smooth=None, sector_label=None,
                        prot_prior='fft', prot_prior_func=None, prot_prior_func_kwargs={}):
    """
    Bins a timeseries to the desired cadence. Works much faster than Lightkurve's built in binning function.

    Args:
        lc (:obj:`LightCurve obj`): the cleaned lightcurve.
        bs (:obj:`float`): the size of the bins, in units of seconds.
        precleaned (:obj:`bool`, optional): set to True if the provided `lc` argument has already been cleaned and normalized.
        transit (:obj:`array`, optional): array of transit paramters like [period, epoch, duration] in units of days each entry can be an array if there are multiple planets. Also used by `default_cleaning_func`.
        max_lag (:obj:`float`, optional): the maximum lag to which to calculate the ACF in units of days.
        smooth (:obj:`int`, optional): if supplied, will apply smoothing to the LC before fitting parabolas by convolving with a gaussian with a FWHM equal to this value.
        sector_label (:obj:`int` or :obj:`str`, optional): tallows you to set a custom sector label, must be castable to a string.
        prot_prior_func (:obj:`func`, optional): the function to be used to idntify the period_guess_index (pgi) for the rotation period. Defaults to `calc_fft_pgi()`
        prot_prior_func_kwargs (:obj:`dict`, optional): keyword arguments for the `prot_prior_func` function.

    Returns:
    Two parameters

        - fits_result (:obj:`dict`): dictionary containing information on the LC and ACF.
        - process_result (:obj:`dict`): dictionary containing information on the parabola fits.
    """
    lc_clean = lc
    
    # calculate the acf
    fits_result = calc_acf(lc_clean, bs=bs, max_lag=max_lag, smooth=smooth)

    # add the raw light curve to the fits_result for ease of inspection later on
    fits_result['time_raw'] = lc['time']
    fits_result['flux_raw'] = lc['nflux_dtr']
    fits_result['flux_err_raw'] = lc['nflux_err']
    
    # calculate parabola fits
    # first, check what kind of prior was passed
    if prot_prior == 'fft':
        process_result = calc_parabolas(fits_result['acf'], bs=bs, prot_prior_func=calc_fft_pgi)
    elif prot_prior == 'custom':
        process_result = calc_parabolas(fits_result['acf'], bs=bs, prot_prior_func=prot_prior_func, 
                                prot_prior_func_kwargs=prot_prior_func_kwargs)
    elif type(prot_prior)==int :
        process_result = calc_parabolas(fits_result['acf'], bs=bs, prot_prior_func=prot_prior)
    else:
        process_resutl = {}
        print('Invalid argument passed to prot_prior. Please use \'fft\', \'custom\', or a float.')

    return fits_result, process_result


# # PLOTTING FUNCTIONS

def custom_plot(x, y, ax=None, **plt_kwargs):
    """
    (Under construction) Custom plotting function.

    Args:
        x (:obj:`arr`): the x array.
        y (:obj:`array`): the y array.
        ax (:obj:`axis object`, optional): the axis object on which to plot.
        peak_num (:obj:`int`): which alias you are trying to fit (1 being the original peak, 2 being the first alias.)
        **plt_kwargs (:obj:`dict`): keyword arguments.

    Returns:
        ax (:obj:`obj`): the plotted function.
    """
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **plt_kwargs) ## example plot here
    return(ax)

def plot_fft(fits_result,process_result, plot_peaks=True, **plt_kwargs):
    """
    Given the result dictionaries from process_LightCurve, plots the FFT of the ACF. Returns a figure object.

    Args:
        fits_result (:obj:`dict`): dictionary containing information on the LC and ACF, as returned by `process_LightCurve()`.
        process_result (:obj:`dict`): dictionary containing information on the parabola fits, as returned by `process_LightCurve()`.
        plot_peaks (:obj:`dict`):  if True, places a marker on the five tallest peaks in the FFT.
        plt_kwargs (:obj:`dict`):  keyword arguments for `matplotlib.plt.plot()`.
    """
    fft = process_result['fft']
    fft_period = process_result['fft_period_days']
    pgi = process_result['pgi']
    fft_pgi = process_result['fft_pgi']
    
    # make le plot!
    fig, ax = plt.subplots(1, 1, figsize=[12,5], facecolor='white')
    ax.plot(fft_period,fft, color='black', **plt_kwargs)

    # also mark the highest peaks, with the highest one marked in red and the rest in green
    if plot_peaks :
        peaks_x = process_result['max_peaks']
        peaks_y = fft[peaks_x]
        ax.plot(fft_period[peaks_x], peaks_y, marker='o', color='green',linestyle='', markersize=8, fillstyle='none')
        ax.plot(fft_period[fft_pgi], fft[fft_pgi], marker='o', color='red', markersize=8, fillstyle='none')

    # labels
    ax.set_title("FFT of the ACF", fontsize=14)
    ax.set_xlabel("Period (days)", fontsize=14)
    ax.set_ylabel("FFT Power", fontsize=14)

    return fig, ax


# # PLOTTING AND PRINTING FUNCTIONS

def print_summary(fits_result, process_result, bs=cadence):
    """
    Prints a summary of the descriptive parameters calculated by `process_LightCurve()`.
    
    Args:
        fits_result (:obj:`dict`): dictionary containing information on the LC and ACF, as returned by `process_LightCurve()`.
        process_result (:obj:`dict`): dictionary containing information on the parabola fits, as returned by `process_LightCurve()`.    
    """
    print('pgi: ' + "%.3f" % bins_to_days(process_result['pgi']/(24*30), bs=bs))
    print('A_avg: ' + "%.3f" % (process_result['A_avg']))
    print('B_avg: ' + "%.3f" % (process_result['B_avg']))
    print('Rsq_avg: ' + "%.3f" % (process_result['R_avg']))
    print('P_avg (days): ' + "%.3f" % bins_to_days(process_result['P_avg'], bs=bs))
    print('P_err (days): ' + "%.3f" % bins_to_days(process_result['P_err'], bs=bs))
    print('HWHM_err (days): ' + "%.3f" % bins_to_days(np.nanmean(process_result['hwhm_k']), bs=bs))
    print()
    

def plot_acf(fits_result,process_result, plot_peaks=True, plot_line=None, cut=10):
    """
    Prints a summary of the descriptive parameters calculated by `process_LightCurve()`.
    
    Args:
        fits_result (:obj:`dict`): dictionary containing information on the LC and ACF, as returned by `process_LightCurve()`.
        process_result (:obj:`dict`): dictionary containing information on the parabola fits, as returned by `process_LightCurve()`.   
        plot_peaks (:obj:`bool`): if set to True, will overplot the parabola fits to the ACF peaks on the ACF plot
        plot_line (:obj:`float`): if given a lag time in days, will plot a vertical line on the ACF at that x-value, indended to plot the found period for visual comparison
        cut (:obj:`int`): plots look better when you cut the first few points off the ACF, to avoid the high peak at (0,1). This keyword lets you adjust how many points get cut off the front.
    
    Returns:
        fig (:obj:`obj`): the figure object. 
        ax (:obj:`obj`): the axis object with the plotted function. 
    """
    # make the base plot
    # fig_num=plt.figure().number + 1
    fig, ax = plt.subplots(1, 1, figsize=[10,5], facecolor='white')
    ax.plot(fits_result['acf_lags'][cut:],fits_result['acf'][cut:])
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("ACF")
    ax.set_title("ACF")

    pgi = process_result['pgi']
    # check if the test failed
    if pgi <= 0 or np.isnan(pgi) :
        print('Cannot plot peaks, no plausible rotation period detected.')
        return fig, ax
    if pgi > len(fits_result['acf_lags']//2) :
        print("Test failed due to pgi > 1/2*sample length") 
        return fig, ax

    # if desired, plot the fitted peaks to the acf
    if plot_peaks :
        # extract the needed info
        # plots look better when you cut off the first few points in the ACF
        lag_times = fits_result['acf_lags']
        pgi = process_result['pgi']
        fitted_parabolas = process_result['fitted_parabola_k']
        try : 
            acf_snip = pgi * 7
        except :
            acf_snip = pgi * len(process_result['a_k']) + pgi/2
        acf_snip = min(acf_snip, len(lag_times)-1)

        # Set a reasonable limit
        ax.set_xlim([0, lag_times[acf_snip]])
        
        # now plot each peak
        for i in range(len(process_result['a_k'])):
            curve = fitted_parabolas[i]
            window = len(curve)
            snip = np.arange(pgi*(i+1) - window/2, pgi*(i+1) + window/2).astype(int)
            lag_snip = lag_times[snip]
            ax.plot(lag_snip, curve,linewidth=4,color='r',linestyle='-')
            
        
        # now make it pretty
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.set_title("ACF")

        # if provided, plot a line where requested
    if plot_line :
        line_y = np.arange(-1.5,1.5,.1)
        line_x = np.ones(len(line_y)) * (plot_line)# / (cad/bs)
        ax.plot(line_x, line_y, c='g', scaley=False)
    return fig, ax
