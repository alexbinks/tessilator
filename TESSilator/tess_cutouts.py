'''
Alexander Binks and Moritz Guenther, 2023

The TESSilator

Licence: MIT

This is a python3 program designed to provide an all-in-one module to measure lightcurves and stellar rotation periods from the Transiting Exoplanet Survey Satellite (TESS). Whilst there are many useful (and powerful) software tools available for working with TESS data, they are mostly provided as various steps in the data reduction process --- to our knowledge there are no programs that automate the full process from downloading the data (start) to obtaining rotation period measurements (finish). The software provided here fills this gap. The user provides a table of targets with basic Gaia DR3 information (source ID, sky positions, parallax and Gaia G magnitude) and simply allows the TESSilator to do the rest! The steps are:

(1) download photometric time-series data from TESS.

(2) scan the Gaia DR3 catalogue to quantify the level of background contamination from nearby sources.

(3) clean the lightcurves for poor quality data caused by systematic and instrumental effects.

(4) normalize and detrend lightcurves over the whole sector of observations.

(5) measure stellar rotation periods using two processes: the Lomb-Scargle periodogram and the Auto-Correlation Function.

(6) quantify various data quality metrics from photometric time-series data which can be used by the user to assess data reliability

The calling sequence is:

python tess_cutouts.py flux_con LC_con make_plots file_ref t_filename

where
      "flux_con" is the toggle for applying the contamination calculation (yes=1, no=0)
      "LC_con" determines if lightcurve/periodogram analyses should be carried out for neighbouring contaminants (yes=1, no=0)
      "make_plots" gives the user the option to make plots (yes=1, no=0)
      "file_ref" is a string expression used to reference the files produced.
      "t_filename" is the name of the input file required for analysis.

In this module, the data is downloaded from TESSCut (Brasseur et al. 2019) -- a service which allows the user to acquire a stack of n 20x20 pixel "postage-stamp" image frames ordered in time sequence and centered using the celestial coordinates provided by the user, where n represents the number of TESS sectors available for download. It uses modules from the TesscutClass (https://astroquery.readthedocs.io/en/latest/api/astroquery.mast.TesscutClass.html - part of the astroquery.mast package) to download the data, then applies steps 2-6, sector-by-sector.

Since there are no requirements to download the full image frames (0.5-0.7TB and 1.8-2.0TB per sector for 30 and 10 min cadence observations, respectively) this software is recommended for users who require a relatively fast extraction for a manageable number of targets (i.e., < 5000). With the correct pre-requisite Python modules and an uninterrupted internet connection, for a target with 5 sectors of TESS data, the processing time is approximately 1-2 minutes (depending whether or not the user wants to measure contamination and/or generate plots). If the user is interested in formulating a much larger survey, we recommend they obtain the bulk downloads of the TESS calibrated full-frame images (https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html) and run the "tess_large_sectors.py" module, which has been used to measure rotation periods for catalogues of >1 million targets.

The module "tess_functions.py" contains the functions called to run both tess_cutouts.py and tess_large_sectors.py

Should there be any problems in using this software please contact Alex Binks (lead author) at abinks@mit.edu

If this package is useful for research leading to publication we would appreciate the following acknowledgement:

"The data from the Transiting Exoplanet Survey Satellite (TESS) was acquired using the TESSilator software package (Binks et al. 2023)."
'''

from .modules_to_import import *
from .tess_functions2 import *


#TESS pixel size in arcseconds
pixel_size = 21.0 
exprf = 0.65
#Zero-point magnitude from Vanderspek et al. 2018
Zpt, eZpt = 20.44, 0.05




# Aperture size, background annulus and positional uncertainty (all in pixels)
# A 1-pixel aperture is considered an optimal extraction following experimentation.
# The details are described in Appendix B3 of Binks et al. 2022.

def setupFilenames(file_ref):
    '''Set up the file names to store data
    
    parameters
    ----------
    
    file_ref : str
    
    returns
    -------
    con_file : to store contamination values.
    period_file : for recording parameters measured by the periodogram analysis.
    store_file : to record any logging information.
    '''    
    con_file = '_'.join(['contamination', file_ref, 'tesscut'])+'.ecsv'
    period_file = '_'.join(['periods', file_ref, 'tesscut'])+'.ecsv'
    store_file = '_'.join(["cur", file_ref, 'tesscut'])+".txt"
    return con_file, period_file, store_file

def readData(t_filename):
    '''Read input data and convert to an formatted astropy table ready for analysis
    
    The input data must be in the form of a comma-separated variable and may
    take 3 forms:
    a) a 1-column table of source identifiers
    b) a 2-column table of celestial sky coordinates
    c) a pre-prepared table of 5 columns consisting of
       source_id, ra, dec, parallax, Gmag
       
    parameters
    ----------
    t_filename : `astropy.table.Table`
        name of the file containing the input data

    returns
    -------
    t_targets : `astropy.table.Table`
        a formatted astropy table ready for further analysis
    '''
# Produce an astropy table with the relevant column names and data types. These will be filled in list comprehensions later in the program.
    final_table = Table(names=['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag', 'Sector', 'Camera', 'CCD', 'log_tot_bg_star', 'log_max_bg_star', 'n_contaminants', 'Period_Max', 'Period_Gauss', 'e_Period', 'Period_2', 'power1', 'power1_power2', 'FAP_001', 'amp', 'scatter', 'fdev', 'Ndata', 'cont_flags'],
                        dtype=(str, str, float, float, float, float, int, int, int, float, float, int, float, float, float, float, float, float, float, float, float, float, int, str))


    start_date = time.asctime(time.localtime(time.time()))
    start = time.time()
    print(start_date)

    t_input = readascii(t_filename)
    t_targets = getGaiaData(t_input)
    return t_targets

# Generate a new store_file if one exists previously. This is used as a debugging log. 
#    with open(store_file, 'w') as file1:
#        file1.write('Starting Time: '+str(start_date)+'\n')
#    return t_targets

# If the user selects sys.argv[1] = 1, then the contamination method
# will be implemented. Warning --- if the contamination file already
# exists and a value of 1 is selected, the file will be overwritten.
# Alternatively, if this toggle is not selected, the contamination
# parameters are all initially assigned dummy values. Then if a pre-
# existing contamination file exists, these results will be used.
# The idea here is to save the user time if they have already used
# the contamination module in a previous run. If the pre-existing
# table is only partially completed, only the targets that match the
# input list will be changed (the missing data will still have the
# dummy values).

def collectContaminationData(t_targets, con_file, flux_con=True, LC_con=0):
    '''Collect data on contamination sources around selelcted targets
    
    This function takes a target table and, if requested, prints out details of the total flux contribution from neighbouring contaminating sources for each target. It also returns a table (in descending order) of (up to) 5 neighbouring contaminants that contribute the most flux in the target aperture, if requested.
    
    Parameters
    ----------
    t_targets : `astropy.table.Table`
        target table with columns "name, source_id, ra, dec, parallax, Gmag" 
    con_file : string
        file name to save the contamination table
    flux_con : bool
        Decides if the contamination 
    chose_con : int
        set to 1 to calcualation more details on the contmination. This 
        parameter has no effect is ``use_con=False``.
        
    Returns
    -------
    t_targets : `astropy.table.Table`
        Input target table with added columns xyx, which describe TEXT HERE...
    coon_table_full : `astropy.table.Table` or None
        aas;lfjasldkfjasd;lkjf
    '''
    print(t_targets)
    if flux_con:
        t_targets, t_contam, t_con_table = contamination(t_targets)
        t_contam.write(con_file, overwrite=True)
        if LC_con:
            t_con_table.write(con_file[:-5]+'_individiual'+con_file[-5:], overwrite=True)
    else:
        t_targets["log_tot_bg"] = -999.
        t_targets["log_max_bg"] = -999.
        t_targets["num_tot_bg"] = -999.
    
        if os.path.exists(con_file):
            t_contam = Table.read(con_file)
            for i in range(len(t)):
                g = (t_contam["source_id"] == t_targets["source_id"].astype(str)[i])
                if len(g) >= 1:
                    t_targets["log_tot_bg"][i] = t_contam["log_tot_bg"][g][0]
                    t_targets["log_max_bg"][i] = t_contam["log_max_bg"][g][0]
                    t_targets["num_tot_bg"][i] = t_contam["num_tot_bg"][g][0]
    return t_targets


def single_epoch(file_in, t_target, cutout_size, store_file):
    '''Run the aperture photometry, lightcurve cleaning and periodogram analysis
    
    parameters
    ----------
    file_in : str
        name of the input TESS cutout fits file
    t_target : `astropy.table.Table`
        details of the target star
    cutout_size : float
        the pixel length of the downloaded cutout
    store_file : str
        name of the file to save logging information
        
    returns
    -------
    the final period file
    a plot of the lightcurve (if requested)
    '''
    print(file_in)
    t_sp = file_in.split('_')
    scc = [int(t_sp[-3][1:]), int(t_sp[-2]), int(t_sp[-1][0])] # simply extract the sector, ccd and camera numbers from the fits file.
    print(t_target)
# create a numpy array which holds the photometric measurements for each timestep from the aper_run_cutouts routine.
    
    XY_ctr = (cutout_size/2., cutout_size/2.)
    phot_targ = np.array(aper_run_cutouts(file_in, XY_ctr, store_file=store_file, make_slices=0))
    phot_targ_table = make_phot_table(phot_targ)
    if phot_targ_table is None:
        return
    clean_norm_lc, original_norm_lc = make_lc(phot_targ_table)
    if len(clean_norm_lc) != 0:
        d_target = run_LS(clean_norm_lc, store_file)
        labels_cont = 'z'
    else:
        return

    LC_con=0
    if LC_con == 1:
        if flux_con != 1:
            print("Contaminants not identified! Please toggle LC_con=1\nContinuing program using only the target.")
        else:
            phot_cont = {}
            labels_cont = ''
            con_table = ascii.read(con_file[:-5]+'_individiual'+con_file[-5:])
            con_table = con_table[con_table["source_id_target"] == t_target["source_id"]]
            X_con, Y_con = find_XY_cont(file_in, con_table)
            X_con += XY_ctr[0]
            Y_con += XY_ctr[1]
            XY_arr = np.array([X_con, Y_con]).T
            phot_cont['XY'] = XY_arr
            print(XY_arr)
            for z in range(len(XY_arr)):
                XY_con = tuple((XY_arr[z][0], XY_arr[z][1]))
                phot_cont[z] = np.array(aper_run_cutouts(file_in, XY_con, store_file=store_file))
                phot_cont_table = make_phot_table(phot_cont[z])
                if phot_cont_table is None:
                    labels_cont += 'd'
                else:
                    clean_norm_lc_cont, original_norm_lc_cont = make_lc(phot_cont_table)
                    if len(clean_norm_lc_cont) != 0:
                        d_cont = run_LS(clean_norm_lc_cont)
                        labels_cont += isPeriodCont(d_target, d_cont, con_table[z])
                    else:
                        labels_cont += 'd'

# make an astropy table out of the array
    if make_plots == 1:
        LC_con = 0
        if LC_con == 1:
            make_LC_plots(file_in, clean_norm_lc, original_norm_lc, d_target, scc, t_target, phot_cont['XY'])
        else:
            make_LC_plots(file_in, clean_norm_lc, original_norm_lc, d_target, scc, t_target)
    datarow = [
               str(t_target["name"]),
               str(t_target["source_id"]),
               t_target["ra"],
               t_target["dec"],
               t_target["parallax"],
               t_target["Gmag"],
               scc[0],
               scc[1],
               scc[2],
               t_target["log_tot_bg"],
               t_target["log_max_bg"],
               t_target["num_tot_bg"],
               d_target['period_best'],
               d_target['Gauss_fit_peak_parameters'][1],
               d_target['Gauss_fit_peak_parameters'][2],
               d_target['period_second'],
               d_target['power_best'],
               d_target['power_best']/d_target['power_second'],
               d_target['FAPs'][2],
               d_target['pops_vals'][1],
               d_target['phase_scatter'],
               d_target['frac_phase_outliers'],
               d_target['Ndata'],
               labels_cont
              ]
    final_table.add_row(datarow)
    final_table.write(period_file, overwrite=True)


def iterate_source(coord, target, store_file, cutout_size=20):
    manifest = Tesscut.download_cutouts(coordinates=coord, size=cutout_size, sector=None)
    for j in range(len(manifest)):
         print(f"{j+1} of {len(manifest)} sectors")
         file_in = manifest["Local Path"][j]
         name_underscore = str(target['name']).replace(" ", "_")
         f_sp = file_in.split('-')
         file_new = '_'.join([name_underscore, f_sp[1][1:], f_sp[2], f_sp[3][0]])+'.fits'
         os.rename(file_in, file_new)
         with open(store_file, 'a') as file1:
             file1.write(f'{target["source_id"]}, {j+1}/{len(manifest)}\n')
         t = single_epoch(file_new, target, cutout_size, store_file)

def iterate_sources(t_targets, store_file):
    for i, target in enumerate(t_targets):
        print(f"{target['name']}, star # {i+1} of {len(t_targets)}")
        coo = SkyCoord(target["ra"], target["dec"], unit="deg")
        table_one_source = iterate_source(coo, target, store_file)
