from ..lc_analysis import aic_selector, clean_edges_outlier, clean_edges_scatter, detrend_lc, get_time_segments, logger, make_lc, norm_choice, remove_sparse_data, sin_fit, cbv_fit_test

from ..logger import logger_tessilator
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

targ_dir = './tessilator/tests/targets_tests'

aps = [f'{targ_dir}/Gaia_DR3_2314778985026776320_tests/ap_2314778985026776320_0029_1_2.csv',
       f'{targ_dir}/BD+20_2465_tests/ap_BD+20_2465_0045_3_4.csv',
       f'{targ_dir}/GESJ08065664-4703000_tests/ap_GESJ08065664-4703000_0061_3_1.csv',
       f'{targ_dir}/ABDor_tests/ap_AB_Dor_0036_4_3.csv']
lcs = [f'{targ_dir}/Gaia_DR3_2314778985026776320_tests/lc_2314778985026776320_0029_1_2_reg_oflux.csv',
       f'{targ_dir}/BD+20_2465_tests/lc_BD+20_2465_0045_3_4_reg_oflux.csv',
       f'{targ_dir}/GESJ08065664-4703000_tests/lc_GESJ08065664-4703000_0061_3_1_reg_oflux.csv',
       f'{targ_dir}/ABDor_tests/lc_AB_Dor_0036_4_3_reg_oflux.csv']

for ap_file, lc_file in zip(aps, lcs): 
    ap_in = ascii.read(ap_file)
    ap_in = ap_in[ap_in["flux"] > 0.0]
    lc_test = ascii.read(lc_file)
    lc_exam = make_lc(ap_in)[0]
    for i in range(len(lc_exam)):#, lc_exam):
        assert(str(lc_test['pass_sparse'][i]) == str(lc_exam['pass_sparse'][i]))
        assert(str(lc_test['pass_clean_outlier'][i]) == str(lc_exam['pass_clean_outlier'][i]))
        assert(str(lc_test['pass_clean_scatter'][i]) == str(lc_exam['pass_clean_scatter'][i]))
        assert(str(lc_test['pass_full_outlier'][i]) == str(lc_exam['pass_full_outlier'][i]))

