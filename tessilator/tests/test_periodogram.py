
import warnings

# Third party imports
import numpy as np
import os

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy.io import ascii, fits
import astropy.units as u
from astropy.stats import akaike_info_criterion_lsq
from glob import glob
from ..file_io import logger_tessilator


from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit
import itertools as it

import matplotlib.pyplot as plt 
from collections.abc import Iterable

# Local application imports
from ..fixedconstants import pixel_size, exprf, Zpt, eZpt, sec_max
from ..periodogram import check_for_jumps, gauss_fit, gauss_fit_peak, get_next_peak, get_Gauss_params_pg, logger, mean_of_arrays, run_ls
from ..contaminants import is_period_cont
from ..lc_analysis import sin_fit

start, stop, typical_timestep = 0, 27, 0.007 # in days
t = np.linspace(start=start+typical_timestep, stop=stop, num=int(stop/typical_timestep), endpoint=True)


def test_single_sine_fit():
    period, amp, y_err = 5.0, 0.1, 0.005
    y = 1. + (amp+(y_err*np.random.randn(len(t))))*np.sin(2.*np.pi*t/period)
    ls = LombScargle(t, y, dy=y_err)
    frequency, power = ls.autopower(minimum_frequency=1./100.,
                                    maximum_frequency=1./0.05,
                                    samples_per_peak=50)
    p_m = np.argmax(power)
    period_1 = 1.0/frequency[p_m]
    power_1 = power[p_m]
    assert(np.isclose(period_1, period, rtol=0.1))
    

def test_multi_peak_fit():
    periods = [5.0, 2.8, 0.8, 10.3]
    amplitudes = [0.3, 0.2, 0.15, 0.1]
    phases = 2.*np.pi*np.random.rand(4)
    y_err = 0.000005
    y = np.ones(len(t))
    for i in range(4):
        y += (amplitudes[i]+(y_err*np.random.randn(len(t))))*np.sin((2.*np.pi*t/periods[i])+phases[i])
    ls = LombScargle(t, y, dy=y_err)
    plt.plot(t, y)
#    plt.show()
    frequency, power = ls.autopower(minimum_frequency=1./100.,
                                    maximum_frequency=1./0.05,
                                    samples_per_peak=50)

    p_m = np.argmax(power)
    period_1 = 1.0/frequency[p_m]
    power_1 = power[p_m]
    period = 1./frequency[::-1]
    power = power[::-1]

#    plt.plot(period, power)
#    plt.xlim([0,20])
#    plt.show()
    LS_dict = {}
    LS_dict['a_1'] = np.arange(len(power))
    LS_dict['period_a_1'] = period
    LS_dict['power_a_1'] = power
    LS_dict['period_1'] = period_1
    LS_dict['power_1'] = power_1
    n_peaks = 3
    for i in 1+np.arange(n_peaks):
        try:
            # get the indices of all the peaks that were not part of the last peak
            LS_dict[f'a_{i+1}'] = get_next_peak(LS_dict[f'power_a_{i}'])
            # all the indices that 'are' part of the peak
            LS_dict[f'a_g_{i}'] = np.delete(np.array(LS_dict[f'a_{i}']), np.array(LS_dict[f'a_{i+1}']))
            LS_dict[f'Gauss_{i}'], LS_dict[f'Gauss_y_{i}'] = get_Gauss_params_pg(period, power, LS_dict[f'a_g_{i}'])
           # find all the new period values in the new array
            LS_dict[f'period_a_{i+1}'] = LS_dict[f'period_a_{i}'][LS_dict[f'a_{i+1}']]
            # find all the new power values in the new array
            LS_dict[f'power_a_{i+1}'] = LS_dict[f'power_a_{i}'][LS_dict[f'a_{i+1}']]
            # calculate the period of the maximum power peak
            LS_dict[f'period_{i+1}'] = LS_dict[f'period_a_{i+1}'][np.argmax(LS_dict[f'power_a_{i+1}'])]
            # return the maximum power peak value
            LS_dict[f'power_{i+1}'] = LS_dict[f'power_a_{i+1}'][np.argmax(LS_dict[f'power_a_{i+1}'])]
        except:
            logger.error(f'Something went wrong with the periods/powers of subsequent peaks. Probably an empty array of values.')
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
    for i in range(3):
        print(LS_dict[f'period_{i+1}'])
        assert((LS_dict[f'period_{i+1}']/periods[i] > 0.8) and LS_dict[f'period_{i+1}']/periods[i] < 1.2)
        
def test_real_targets(p_min_thresh=0.05, p_max_thresh=100., samples_per_peak=10):
    target_roots = glob('./targets_tests/*')

    lcs = sorted([glob(f'{x}/lc*reg*csv')[0] for x in target_roots])

    periods = [4.604,0.660,0.356,14.773,10.802]
    for i, (lc, period) in enumerate(zip(lcs, periods)):
        print(lc)
        converters = {'pass_sparse' : bool,
                      'pass_clean_scatter' : bool,
                      'pass_clean_outlier' : bool,
                      'pass_full_outlier' : bool}
        lc_data = Table.read(lc, converters=converters)
        cln_cond = np.logical_and.reduce([
                   lc_data["pass_clean_scatter"],
                   lc_data["pass_clean_outlier"],
                   lc_data["pass_full_outlier"]
                   ])
        cln = lc_data[cln_cond]

        time = np.array(cln["time"])
        nflux = np.array(cln["nflux_dtr"])
        enflux = np.array(cln["nflux_err"])
        ls = LombScargle(time, nflux, dy=enflux)
        frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                        maximum_frequency=1./p_min_thresh,
                                        samples_per_peak=samples_per_peak)
        p_m = np.argmax(power)
        period_1 = 1.0/frequency[p_m]
        power_1 = power[p_m]
        print(period_1, power_1)
        if i >= 3:
            assert(period_1 > 5.)
        else:
            assert((period_1/period > 0.8) and (period_1/period < 1.2))
