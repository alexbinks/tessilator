import inspect
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from ..lc_analysis import aic_selector
from ..file_io import logger_tessilator


logger = logger_tessilator('aic_tests')

start, stop, typical_timestep = 0, 27, 0.007 # in days
period = 3.5
times = np.linspace(start=start+typical_timestep, stop=stop, num=int(stop/typical_timestep), endpoint=True)
y_err = 0.000000000005

text_x, text_y = 0.05, 0.95

flat_coeffs = [4.]
line_coeffs = [1., .1]
para_coeffs = [1.5, 0.75, 3.]



#########################################
######FUNCTIONS NEEDED TO RUN TESTS######
#########################################
def makeplot(x, y, def_name, coeffs):
    fig, ax = plt.subplots(figsize=(10,3))    
    ax.scatter(x, y)
    ax.text(text_x, text_y, coeffs, transform=ax.transAxes)
    plt.savefig(f'{def_name}.png', bbox_inches='tight')


def get_coords(curve, err=False):
    x = times
    y = np.zeros(len(times))
    for i, c in enumerate(curve):
        y += c*times**(len(curve)-i-1)
    if err:
        y = [i+np.random.normal(0, y_err) for i in y]
    return x, y



def run_aic_tests(coeff_type, err, def_name='aic_test'):
    x, y = get_coords(coeff_type, err=err)
    print('len: ', len(coeff_type))
    poly_ord, coeffs = aic_selector(x, y, poly_max=len(coeff_type))
    makeplot(x, y, def_name, coeffs)
    print(len(coeff_type))
    for i in range(len(coeff_type)):
        assert(np.isclose(coeffs[i], coeff_type[i], rtol=1e-01))
    assert(len(coeffs) == len(coeff_type))


#########################################
##############RUN EACH TEST##############
#########################################
def test_aic_parabola_fail():
    '''ENSURE THE DEFAULT VALUES ARE RETURNED IF THE FUNCTION CRASHES'''
    def_name = inspect.stack()[0][0].f_code.co_name
    x, y = times, np.ones(len(times)+50)
    poly_ord, coeffs = aic_selector(x, y, poly_max=3)
    assert(np.isclose(poly_ord, 0.0, rtol=1e-05))
    assert(np.isclose(coeffs[0], 1.0, rtol=1e-05))
    assert(len(coeffs) == 1)

def test_aic_flat_no_err():
    '''TRY A COMPLETELY FLAT LIGHTCURVE'''
    run_aic_tests(flat_coeffs, err=False, def_name=inspect.stack()[0][0].f_code.co_name)

def test_aic_flat_with_err():
    '''TRY A FLAT LIGHTCURVE WITH RANDOM GAUSSIAN Y-ERRORS'''
    run_aic_tests(flat_coeffs, err=True, def_name=inspect.stack()[0][0].f_code.co_name)

def test_aic_linear_no_err():
    '''TRY A COMPLETELY LINEAR LIGHTCURVE'''
    run_aic_tests(line_coeffs, err=False, def_name=inspect.stack()[0][0].f_code.co_name)

def test_aic_linear_with_err():
    '''TRY A LINEAR LIGHTCURVE WITH RANDOM GAUSSIAN Y-ERRORS'''
    run_aic_tests(line_coeffs, err=True, def_name=inspect.stack()[0][0].f_code.co_name)

def test_aic_parabola_no_err():
    '''TRY A COMPLETELY PARABOLIC LIGHTCURVE'''
    run_aic_tests(para_coeffs, err=False, def_name=inspect.stack()[0][0].f_code.co_name)

def test_aic_parabola_with_err():
    '''TRY A PARABOLIC LIGHTCURVE WITH RANDOM GAUSSIAN Y-ERRORS'''
    run_aic_tests(para_coeffs, err=True, def_name=inspect.stack()[0][0].f_code.co_name)
