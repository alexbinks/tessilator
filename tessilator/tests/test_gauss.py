# import the remaining modules required for the TESS analysis

import os, glob
import numpy as np
from astropy.io import ascii
from ..lc_analysis import get_second_peak
from ..lc_analysis import gauss_fit_peak

def test_gaussfit_periodogram():
    dirname = os.path.dirname(__file__)
    AUMic_Table = ascii.read(dirname +"/AUMic_tests/AU_Mic_sec1_per_pow.dat")
    period = AUMic_Table["period"]
    power = AUMic_Table["power"]
    a_g, a_o = get_second_peak(power)
    pow_r = max(power[a_g])-min(power[a_g])
    a_g_fit = a_g[power[a_g] > min(power[a_g]) + .05*pow_r]

    popt, ym = gauss_fit_peak(period[a_g_fit], power[a_g_fit])

    assert np.isclose(popt[0], 0.08549122, rtol=0.0, atol=1e-2)
    assert np.isclose(popt[1], 9.83170108, rtol=0.0, atol=1e-1)
    assert np.isclose(popt[2], 1.87462235, rtol=0.0, atol=1e-2)
