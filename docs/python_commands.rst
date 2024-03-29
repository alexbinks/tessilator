.. _python_commands:

Running the tessilator as part of a python script
=================================================

The two sets of python commands below provide minimal working examples of how the tessilator functions may be used as part of a python script. These commands can be copied into a python module or passed as separate commands in an interactive python shell.


1) **Using the tessilator with TESScut**
----------------------------------------

Import the tessilator, logging and numpy module

   >>> from tessilator import tessilator
   >>> import logging
   >>> import numpy as np

Provide tessilator with the input parameters and store these as variables to be passed to further tasks.

   >>> fluxCon, lcCon, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
   >>> periodFile = tessilator.setup_filenames(fileRef)

Define the aperture radius and the radii of the annulus for background calculation

   >>> Rad, SkyRad = 1.0, np.array([6.0,8.0])

Instantiate a logging script to store all message with a severity level of "info" or above.

   >>> logging.basicConfig(filename="output.log", level=logging.INFO)

Read the input data and ensure this is in the correct format.

   >>> tTargets = tessilator.read_data(tFile, gaia_sys=True)

If required, perform the contamination calculation and append the results to the input table.

   >>> tTargets, conDir = tessilator.collect_contamination_data(tTargets, fluxCon, conFile
   ...                                                          fileRef, Rad=Rad)

Perform the tessilator analysis for all targets provided in the input table.

   >>> tessilator.all_sources_cutout(tTargets, periodFile, lcCon, fluxCon, makePlots, fileRef,
   ...                               Rad=Rad, choose_sec=None, save_phot=True, cbv_flag=False,
   ...                               store_lc=True, tot_attempts=10, cap_files=10, fix_noise=False)

2) **Using the tessilator with sectors**
----------------------------------------

The following python commands provide a minimal working example of how to use the tessilator
for a given sector of data using the calibrated full-frame images.

Import the tessilator and necessary modules

    >>> from tessilator import tessilator
    >>> import os
    >>> import logging
    >>> from astropy.table import join
    >>> from astropy.io import ascii

Provide tessilator with the input parameters and store these as variables to be passed to further tasks.

    >>> fluxCon, scc, makePlots, fileRef, tFile = tessilator.setup_input_parameters()
    >>> periodFile = tessilator.setup_filenames(fileRef, scc=scc)

Define the aperture radius and the radii of the annulus for background calculation

    >>> Rad, SkyRad = 1.0, [6.0,8.0]

Instantiate a logging script to store all message with a severity level of "error" or above.

    >>> logging.basicConfig(filename="output.log", level=logging.ERROR)

Test whether the input table is already correctly formatted for tessilator analysis

    >>> t_large_sec_check = tessilator.test_table_large_sectors(tFile)
    >>> if t_large_sec_check is not None:
    >>>     tTargets = t_large_sec_check
    >>> else:
    >>>     gaia_data = tessilator.read_data(tFile, name_is_source_id=True)
    >>>     xy_pixel_data = tessilator.get_tess_pixel_xy(gaia_data)
    >>>     tTargets = join(gaia_data, xy_pixel_data, keys='source_id')
    >>>     ascii.write(tTargets, tFile, format='csv', overwrite=True)

Select the targets from the input file that land in the required sector

    >>> tTargets = tTargets[tTargets['Sector'] == scc[0]]

Perform the tessilator analysis for all targets within the required sector.

    >>> tessilator.all_sources_sector(tTargets, scc, makePlots, periodFile, fileRef,
    ...                               keep_data=False, fix_noise=False, store_lc=False)
