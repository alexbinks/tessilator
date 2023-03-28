import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from ..lc_analysis import clean_lc

start, stop, typical_timestep = 0, 27, 0.007 # in days
period = 3.5
times = np.linspace(start=start+typical_timestep, stop=stop, num=int(stop/typical_timestep), endpoint=True)


def test_flat():
    '''TRY A COMPLETELY FLAT LIGHTCURVE'''
    x, y = clean_lc(times, np.ones(len(times)))
    assert(x[0] == 0)
    assert(y[0] == len(times)-1)


def test_sine():
    '''TRY A PERFECTLY SINUSOIDAL LIGHTCURVE'''
    x, y = clean_lc(times, 1.0 + 0.1*np.sin(2.*math.pi*times/period))
    assert(x[0] == 0)
    assert(y[0] == len(times)-1)


def test_simulated():
    '''MAKE A FAKE LIGHTCURVE WITH "HEATED" STARTS (SIMILAR TO WHAT IS OFTEN SEEN IN TESS DATA)
    This is made by fitting an exponential at the start of each data string and a sine wave'''
    t_use0, t_use1 = (times < np.median(times) - 1.0), (times > np.median(times) + 1.0)
    times_fin = times[t_use0 | t_use1]
    ts0, ts1 = times[t_use0][0], times[t_use1][0]

    flux_sin = 1.0 + 0.1*np.sin(2.*math.pi*times_fin/period)
    flux_exp0 = np.piecewise(times_fin, [times_fin < ts1, times_fin >= ts1], [lambda x: np.exp(-4.0*(times[t_use0]-ts0)), lambda x: 0])
    flux_exp1 = np.piecewise(times_fin, [times_fin < ts1, times_fin >= ts1], [lambda x: 0, lambda x: np.exp(-4.0*(times[t_use1]-ts1))])
    LCpart = np.piecewise(times_fin, [times_fin < ts1, times_fin >= ts1], [lambda x: 0, lambda x: 1])

    flux_fin = flux_sin+flux_exp0+flux_exp1

    i = 0
    MAD_test = 1.0
    while abs(flux_fin[LCpart == 0][i] - np.median(flux_fin)) >= MAD_test*median_abs_deviation(flux_fin, scale='normal'):
        i += 1
    i0 = i

    i = sum(LCpart == 0)-1
    while abs(flux_fin[LCpart == 0][i] - np.median(flux_fin)) >= MAD_test*median_abs_deviation(flux_fin, scale='normal'):
        i -= 1
    i1 = i

    i = 0
    while abs(flux_fin[LCpart == 1][i] - np.median(flux_fin)) >= MAD_test*median_abs_deviation(flux_fin, scale='normal'):
        i += 1
    i2 = i + sum(LCpart == 0)

    i = sum(LCpart == 1)-1
    while abs(flux_fin[LCpart == 1][i] - np.median(flux_fin)) >= MAD_test*median_abs_deviation(flux_fin, scale='normal'):
        i -= 1
    i3 = i + sum(LCpart == 0)


    x, y = clean_lc(times_fin, flux_fin, MAD_fac=MAD_test, time_fac=10., min_num_per_group=50)

    assert(x[0] == i0)
    assert(y[0] == i1)
    assert(x[1] == i2)
    assert(y[1] == i3)


