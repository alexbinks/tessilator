from TESSilator import tess_cutouts

import numpy as np


flux_con, LC_con, make_plots, file_ref, t_filename = tess_cutouts.setupInputParameters()
con_file, period_file, store_file = tess_cutouts.setupFilenames(file_ref)

Rad, SkyRad = 1.0, np.array([6.0,8.0])


print("Reading the table and formatting into astropy table structure.")
t_targets = tess_cutouts.readData(t_filename, store_file=store_file)
print("Done reading the table and formatting.")

print("...now calculating the contamination.")
t_targets = tess_cutouts.collectContaminationData(t_targets, flux_con, LC_con, con_file, Rad=Rad)
print("Done calculating the contamination.")

print("...now iterating over each source.")
tess_cutouts.iterate_sources(t_targets, period_file, store_file, LC_con, flux_con, con_file, make_plots)