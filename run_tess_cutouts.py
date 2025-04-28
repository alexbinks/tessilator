from tessilator import tessilator

import numpy as np
import logging
import ast

fileRef, tFile = tessilator.setup_input_parameters()
periodFile = tessilator.setup_filenames(fileRef)


logging.basicConfig(filename="output.log", level=logging.INFO)


print(f"Reading the table and formatting into astropy table structure for {tFile}")
tTargets = tessilator.read_data(tFile, gaia_sys=True, type_coord='icrs', name_is_source_id=False)

print("Done reading the table and formatting...now iterating over each source.")

try:
    with open('tessilator_cutout_inputs.txt') as tci:
        data = tci.read() 
        tessilator_inputs = ast.literal_eval(data)

except:
    tessilator_inputs = {'gaia_sys':True,
                         'xy_pos':(10.,10.),
                         'ap_rad':1.,
                         'sky_ann':(6.,8.),
                         'fix_rad':False,
                         'n_cont':5,
                         'cont_rad':10.,
                         'mag_lim':3.,
                         'choose_sec':None,
                         'save_phot':True,
                         'cbv_flag':False,
                         'store_lc':True,
                         'tot_attempts':10,
                         'cap_files':None,
                         'res_ext':'results',
                         'lc_ext':'lc',
                         'pg_ext':'pg',
                         'fits_ext':'fits',
                         'clean_fail_modes':True,
                         'keep_data':True,
                         'fix_noise':False,
                         'shuf_per':True,
                         'shuf_ext':'shuf_plots',
                         'make_plot':True,
                         'calc_cont':True,
                         'lc_cont':False,
                         'make_shuf_plot':True
                         }
tessilator.all_sources_cutout(tTargets, periodFile, fileRef,
                              **tessilator_inputs)

