from tessilator import tessilator

import numpy as np
import logging

fluxCon, lcCon, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
periodFile = tessilator.setup_filenames(fileRef)


logging.basicConfig(filename="output.log", level=logging.INFO)

print(f"Reading the table and formatting into astropy table structure for {tFile}")
tTargets = tessilator.read_data(tFile, gaia_sys=True, type_coord='icrs')
print("Done reading the table and formatting...now iterating over each source.")
tessilator.all_sources_cutout(tTargets, periodFile, lcCon, fluxCon, makePlots,
                              fileRef, gaia_sys=True, xy_pos=(10.,10.),
                              ap_rad=1., sky_ann=(6.,8.), fix_rad=False,
                              n_cont=5, cont_rad=10., mag_lim=3.,
                              choose_sec=[42,43,44], save_phot=True, cbv_flag=False,
                              store_lc=True, tot_attempts=10, cap_files=10,
                              res_ext='results', lc_ext='lc', pg_ext='pg',
                              fits_ext='fits', keep_data=True, fix_noise=False,
                              shuf_per=True, shuf_ext='shuf_plots',
                              make_shuf_plot=True)
