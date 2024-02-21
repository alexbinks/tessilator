import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from ..lc_analysis import clean_flux_edges
from scipy.stats import median_abs_deviation as MAD

# make an artificial lightcurve that has a regular sinusoid for 9 days, then increases linearly in flux from 9-10 days.

start, stop, typical_timestep = 0, 10, 0.01 # in days
period = 3.5
times = np.linspace(start=start+typical_timestep, stop=stop, num=int(stop/typical_timestep), endpoint=True)
y_err = 0.02
Prot, amp = 1., 0.1
MAD_fac = 2.

y = np.piecewise(times, [times < 9, times >= 9], [lambda times: 1 + amp*np.sin(2.*np.pi*times/Prot), lambda times: 2*(times - 8.5)])

y_med, y_MAD = np.median(y), MAD(y, scale='normal')

y_min, y_max = y_med+(MAD_fac*y_MAD), y_med-(MAD_fac*y_MAD)

s, f = clean_flux_edges(y)
