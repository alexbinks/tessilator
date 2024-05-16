from ..lc_analysis import aic_selector, clean_edges_outlier, clean_edges_scatter, detrend_lc, get_time_segments, logger, make_lc, norm_choice, remove_sparse_data, sin_fit, cbv_fit_test

from ..file_io import logger_tessilator
# Third party imports
import numpy as np
import os
import json


from astropy.table import Table
import astropy.units as u
from astropy.stats import akaike_info_criterion_lsq

from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit
from scipy.stats import iqr

import itertools as it
from operator import itemgetter

from astropy.table import Table
from astropy.io import ascii
from glob import glob

makelog=False
if makelog:
    logger = logger_tessilator('clean_lightcurve_tests')


targ_dir = './targets_tests'

target_roots = glob('./targets_tests/*')

aps = [glob(f'{x}/ap*csv')[0] for x in target_roots]
lcs = [glob(f'{x}/lc*reg*csv')[0] for x in target_roots]

for ap_file, lc_file in zip(aps, lcs): 
    ap_in = ascii.read(ap_file)
    ap_in = ap_in[ap_in["flux"] > 0.0]
    lc_test = ascii.read(lc_file)
    lc_exam = make_lc(ap_in)[0]
    print(lc_exam)
    for i in range(len(lc_exam)):#, lc_exam):
        print(i)
        print(lc_test['pass_sparse'][i])
#        print(lc_test['pass_sparse'][i])
#        print(str(lc_test['pass_sparse'][i]), str(lc_exam['pass_sparse'][i]))
#        assert(str(lc_test['pass_sparse'][i]) == str(lc_exam['pass_sparse'][i]))
#        assert(str(lc_test['pass_clean_outlier'][i]) == str(lc_exam['pass_clean_outlier'][i]))
#        assert(str(lc_test['pass_clean_scatter'][i]) == str(lc_exam['pass_clean_scatter'][i]))
#        assert(str(lc_test['pass_full_outlier'][i]) == str(lc_exam['pass_full_outlier'][i]))

