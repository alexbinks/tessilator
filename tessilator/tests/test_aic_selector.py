import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from ..lc_analysis import aic_selector

start, stop, typical_timestep = 0, 27, 0.007 # in days
period = 3.5
times = np.linspace(start=start+typical_timestep, stop=stop, num=int(stop/typical_timestep), endpoint=True)
y_err = 0.02

def test_flat_no_err():
    '''TRY A COMPLETELY FLAT LIGHTCURVE'''
    x, y = times, np.ones(len(times))
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert(np.isclose(poly_ord, 0.0, rtol=1e-05))
    assert(len(coeffs) == 1)

def test_flat_err():
    '''TRY A FLAT LIGHTCURVE WITH RANDOM GAUSSIAN Y-ERRORS'''
    x, y = times, np.ones(len(times))
    y = [i+np.random.normal(0, y_err) for i in y]
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    print(poly_ord)
    assert(np.isclose(poly_ord, 0.0, rtol=1e-05))
    assert(len(coeffs) == 1)

def test_linear_no_err():
    '''TRY A COMPLETELY LINEAR LIGHTCURVE'''
    x, y = times, 1 + 0.001*np.arange(len(times))
    slope = (y[-1]-y[0])/(x[-1]-x[0])
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    print(coeffs, slope)
    assert(np.isclose(coeffs[0], slope, rtol=1e-03))
    assert(np.isclose(coeffs[1], 1.0, rtol=1e-03))
    assert(len(coeffs) == 2)

def test_linear_err():
    '''TRY A LINEAR LIGHTCURVE WITH RANDOM GAUSSIAN Y-ERRORS'''
    x, y = times, 1 + 0.001*np.arange(len(times))
    y = [i+np.random.normal(0, y_err) for i in y]
    slope = (y[-1]-y[0])/(x[-1]-x[0])
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert(np.isclose(coeffs[0], slope, rtol=1e-01))
    assert(np.isclose(coeffs[1], 1.0, rtol=1e-01))
    assert(len(coeffs) == 2)

def test_parabola_no_err():
    '''TRY A COMPLETELY PARABOLIC LIGHTCURVE'''
    x, y = times, 1.5*np.arange(len(times))**2 + 0.75*np.arange(len(times)) + 3.
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert(len(coeffs) == 3)

def test_parabola_no_err():
    '''TRY A PARABOLIC LIGHTCURVE WITH RANDOM GAUSSIAN Y-ERRORS'''
    x, y = times, 1.5*np.arange(len(times))**2 + 0.75*np.arange(len(times)) + 3.
    y = [i+np.random.normal(0, y_err) for i in y]
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert(len(coeffs) == 3)
    
def test_parabola_fail():
    '''ENSURE THE DEFAULT VALUES ARE RETURNED IF THE FUNCTION CRASHES'''
    x, y = times, np.ones(len(times)+50)
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert(np.isclose(poly_ord, 0.0, rtol=1e-05))
    assert(np.isclose(coeffs[0], 1.0, rtol=1e-05))
    assert(len(coeffs) == 1)

