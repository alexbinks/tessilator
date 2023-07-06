'''
Functions used by the tessilator.
'''

from .lc_analysis import *
from .contaminants import *
from .maketable import *
from .makeplots import *
from .tess_stars2px import tess_stars2px_function_entry
from datetime import datetime
import numpy as np
import pyinputplus as pyip
import sys, os
from glob import glob
import logging
from astropy.nddata.utils import Cutout2D
from collections.abc import Iterable
from astropy.table import Table
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Tesscut
import matplotlib.pyplot as plt
import matplotlib as mpl

__all__ = ['start', 'logger',
           'create_table_template', 'setup_input_parameters',
           'setup_filenames', 'test_table_large_sectors', 'read_data',
           'collect_contamination_data', 'make_datarow', 'make_failrow',
           'fix_noise_lc_local', 'fix_noise_lc_sim', 'make_lc_corr_plot', 
           'get_name_target', 'get_median_lc', 'full_run_lc',
           'print_time_taken', 'find_xy_cont', 'run_test_for_contaminant', 
           'get_tess_pixel_xy', 'get_fits', 'make_2d_cutout',
           'cutout_allsecs', 'cutout_onesec', 'cutout_chosensecs',
           'get_cutouts', 'one_source_cutout', 'all_sources_cutout',
           'one_cc', 'all_sources_sector']

start = datetime.now()
print("\n")
print("**********************************************************************")
print("****|******_*********_*********_*********_*********_*********_********")
print("****|*****/*\*******/*\*******/*\*******/*\*******/*\*******/*\*******")
print("****|****/***\*****/***\*****/***\*****/***\*****/***\*****/***\******")
print("****|***/*****\***/*****\***/*****\***/*****\***/*****\***/*****\*****")
print("****|**/*******\_/*******\_/*******\_/*******\_/*******\_/*******\****")
print("****|_____________________________________________________________****")
print("**********************************************************************")
print("**********************WELCOME TO THE TESSILATOR***********************")
print("********The one-stop shop for measuring TESS rotation periods*********")
print("**********************************************************************")
print("**********************************************************************")
print("\n")
print("If this package is useful for research leading to publication we")
print("would appreciate the following acknowledgement:")
print("'The data from the Transiting Exoplanet Survey Satellite (TESS) was")
print("acquired using the tessilator software package (Binks et al. 2023).'")
print("\n")
print("Start time: ", start.strftime("%d/%m/%Y %H:%M:%S"))

# Create custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)



def create_table_template():
    '''Create a template astropy table to store tessilator results.

    returns
    -------
    final_table : `astropy.table.Table`
        A template table to store tessilator results.
    '''
    final_table = Table(names=['name', 'source_id', 'ra', 'dec', 'parallax',\
                           'Gmag', 'Sector', 'Camera', 'CCD',\
                           'log_tot_bg_star', 'log_max_bg_star',\
                           'n_contaminants', 'Period_Max', 'Period_Gauss',\
                           'e_Period', 'Period_2', 'power1', 'power1_power2',\
                           'FAP_001', 'AIC_line', 'AIC_sine', 'amp', 'scatter',\
                           'chisq_phase', 'fdev', 'Ndata','jump_flag','cont_flags'],\
                        dtype=(str, str, float, float, float, float, int, int,\
                               int, float, float, int, float, float, float,\
                               float, float, float, float, float, float, float, float,\
                               float, float, int, int, str))
    return final_table
    
    
def setup_input_parameters():
    '''Retrieve the input parameters to run the tessilator program.

    | The input parameters are:

    | 1) "flux_con": the toggle for applying the contamination calculation
       (yes=1, no=0).

    | 2) either "LC_con", if using the cutout functions or "sector_num" if sectors are needed.       
    * "LC_con" determines if lightcurve/periodogram analyses should be carried out for neighbouring contaminants (yes=1, no=0).
    * "sector_num" prompts the user to enter the sector number needed. If command line arguments are not used, the program will ask if a specific Camera and CCD is needed (1=yes, 0=no). If not required, the whole sector is analysed. If this is a command line argument, if the user enters just the sector number (maximum 2 digits) the whole sector will be analysed, and if the Camera and CCD number are given right after the sector number with no spaces, then a specific Camera and CCD configuration will be used. E.G: if "sector_num = 8", the entire sector 8 is analysed, whereas if "sector_num = 814" then the program will analyse only Camera 1 and CCD 4 in sector 8.

    | 3) "make_plots" gives the user the option to make plots (yes=1, no=0)

    | 4) "file_ref" is a string expression used to reference the files produced.

    | 5) "t_filename" is the name of the input file (or target) required for analysis.

    | If a program is called from the command line without all five input parameters, a set of prompts are initiated to receive input. If just one target is needed, then the user can simply supply either the target name, as long as it is preceeding by a hash (#) symbol.
    | Otherwise, if the full set of command line parameters are supplied, the function will use these as the inputs, however, if they have the wrong format the program will return a warning message and exit.

    parameters
    ----------
    Either arguments supplied on the command line, or the function will prompt
    the user to provide input.
    
    returns
    -------
    flux_con : `bool`
        Run lightcurve analysis for contaminant sources
    scc : `list`, size=3, only if sector data is used
        List containing the sector number, camera and CCD
    LC_con : `bool`, only if cutout data is used
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    file_ref : `str`
        A common string to give all output files the same naming convention
    t_filename : `str`
        The name of the input table containing the targets (or a single target)
     '''
    max_sec = 65
    if len(sys.argv) != 6:
        flux_con = pyip.inputInt("Do you want to search for contaminants? "
                   "1=yes, 0=no : ", min=0, max=1)
        if 'cutout' in sys.argv[0]:
            LC_con = pyip.inputInt("Do you want to calculate period data for "
                     "the contaminants? 1=yes, 0=no : ", min=0, max=1)
        elif 'sector' in sys.argv[0]:
            sector_num = pyip.inputInt("Which sector of data do you require? "
                         f"(1-{max_sec}) : ", min=1, max=max_sec)
            cc_request = pyip.inputInt("Do you want a specific Camera/CCD? "
                                        "1=yes, 0=no : ", min=0, max=1)
            if cc_request:
                cam_num = pyip.inputInt("Which Camera? "
                                        "(1-4)", min=1, max=4)
                ccd_num = pyip.inputInt("Which CCD? "
                                        "(1-4)", min=1, max=4)
                scc = [sector_num, cam_num, ccd_num]
            else:
                scc = [sector_num]
        make_plots = pyip.inputInt("Would you like to make some plots? 1=yes, "
                     "0=no : ", min=0, max=1)
        file_ref = pyip.inputStr("Enter the unique name for referencing the "
                   "output files : ")
        while True:
            t_filename = pyip.inputStr("Enter the file name of your input "
                         "table or object.\nIf this is a single target please enter "
                         "a hash (#) symbol before the identifier : ")
            if t_filename.startswith('#'):
                t_name = t_filename[1:]
                t_name_joined = t_name.replace(' ','_').replace(',', '_')+'.dat'
                if os.path.exists(t_name_joined):
                    os.remove(t_name_joined)
                with open(t_name_joined, 'a') as single_target:
                    single_target.write(t_name)
                t_filename = t_name_joined
                break
            if os.path.exists(t_filename) == False:
                logger.error(f'The file "{t_filename}" does not exist.')
            else:
                break
    else:
        flux_con = int(sys.argv[1])
        if 'cutout' in sys.argv[0]:
            LC_con = int(sys.argv[2])
        elif 'sector' in sys.argv[0]:
            scc_in = str(sys.argv[2])
            if len(scc_in) > 4:
                logger.critical("Incorrect format for sector/camera/ccd values")
                sys.exit()
            elif len(scc_in) < 3:
                scc = [int(scc_in)]
            elif len(scc_in) in [3,4]:
                sec = int(scc_in[:-2])
                cam = int(scc_in[-2])
                ccd = int(scc_in[-1])
                scc = [sec, cam, ccd]
        make_plots = int(sys.argv[3])
        file_ref = sys.argv[4]
        t_filename = sys.argv[5]

        true_vals = [0, 1]
        sec_vals = np.arange(1,max_sec+1)
        cam_ccd_vals = np.arange(1,5)
        if flux_con not in true_vals:
            logger.critical(f"flux_con value {flux_con} not a valid input: "
                            f"choose either 0 or 1.\nExiting program.")
            sys.exit()
        if 'cutout' in sys.argv[0]:
            if LC_con not in true_vals:
                logger.critical(f"LC_con value {LC_con} not a valid input: "
                            f"choose either 0 or 1.\nExiting program.")
                sys.exit()
        elif 'sector' in sys.argv[0]:
            if scc[0] not in sec_vals:
                logger.critical(f"sector_num value {scc[0]} not a valid input: "
                            f"choose integer between 1 and {max_sec}.\nExiting program.")
                sys.exit()
            if len(scc) == 3:
                if scc[1] not in cam_ccd_vals:
                    logger.critical(f"Camera value {scc[1]} out of range: "
                                    f"choose integer between 1 and 4.\nExiting program.")
                    sys.exit()
                if scc[2] not in cam_ccd_vals:
                    logger.critical(f"CCD value {scc[2]} out of range: "
                                    f"choose integer between 1 and 4.\nExiting program.")
                    sys.exit()                        
        if make_plots not in true_vals:
            logger.critical(f"make_plots not a valid input: "
                            f"choose either 0 or 1.\nExiting program.")
            sys.exit()

        if t_filename.startswith('#'):
            logger.info(t_filename)
            t_name = t_filename[1:]
            t_name_joined = t_name.replace(' ','_').replace(',', '_')+'.dat'
            if os.path.exists(t_name_joined):
                os.remove(t_name_joined)
            with open(t_name_joined, 'a') as single_target:
                if t_name.startswith("Gaia DR3 "):
                    single_target.write(t_name[9:])
                else:
                    single_target.write(t_name)
            t_filename = t_name_joined
        if os.path.exists(t_filename) == False:
            logger.critical(f'The file "{t_filename}" does not exist.')
            sys.exit()
    if 'cutout' in sys.argv[0]:
        return flux_con, LC_con, make_plots, file_ref, t_filename
    elif 'sector' in sys.argv[0]:
        return flux_con, scc, make_plots, file_ref, t_filename


def setup_filenames(file_ref, scc=None):
    '''Set up the file names to store data

    parameters
    ----------
    file_ref : `str`
        A common string to give all output files the same naming convention
    scc : `list` or `None`, size=3, optional, default = `None` 
        A list containing the Sector, Camera and CCD.

    returns
    -------
    con_file : `str`
        Name of file to store contamination values.
    period_file : `str`
        Name of file for recording parameters measured by the periodogram
        analysis.
    '''    
    if scc:
        sn = 'sector'+f"{scc[0]:02d}"
        con_file = '_'.join(['contamination', file_ref, sn])
        period_file = '_'.join(['periods', file_ref, sn])
    else:
        con_file = '_'.join(['contamination', file_ref, 'tesscut'])
        period_file = '_'.join(['periods', file_ref, 'tesscut'])

    return con_file, period_file


def test_table_large_sectors(t_filename):
    '''Check if the input file needs modifying at all.

    |If running the tessilator for a whole sector, read the input file and if the format is ready for analysis, make a couple of adjustments, then simply pass the file.
    
    | For a straight pass, the columns must be ordered in two ways. Either:
    * exactly set out with the following columns:
    
        #. source_id: name of the Gaia DR3 source identifier
        #. ra: right ascension
        #. dec: declination
        #. parallax: parallax
        #. Gmag: Gaia DR3 apparent G-band magnitude
        #. Sector: The TESS sector
        #. Camera: The TESS camera
        #. CCD: The TESS CCD number
        #. Xpos: The TESS X-pixel
        #. Ypos: The TESS Y-pixel
        
        
    * The same, but with a preceding column entitled "name", which refers to a target identifier name. This can be any string.
    | In any other case, None is returned and other functions are used to get the table into the correct format.

    parameters
    ----------
    table_in : `str`
        The file name of the table which will be checked for formatting.

    returns
    -------
    t : `astropy.table.Table` or `None`
        either a ready-prepared table or nothing.
    '''

    t = Table.read(t_filename, format='csv')
    # The list of column names which must be spelled exactly.
    cnc = ['source_id', 'ra', 'dec', 'parallax', 'Gmag', 'Sector',
                       'Camera', 'CCD', 'Xpos', 'Ypos']
    if cnc == t.colnames:
        t["name"] = t["source_id"]
        cnc.insert(0, 'name')
        t = t[cnc]
        # Ensure the dtype for "name" and "source_id" are strings.
        t["name"] = t["name"].astype(str)
        t["source_id"] = t["source_id"].astype(str)
        return t
    elif cnc == t.colnames[1:]:
        if t.colnames[0] == "name":
            # Ensure the dtype for "name" and "source_id" are strings.
            t["name"] = t["name"].astype(str)
            t["source_id"] = t["source_id"].astype(str)
            return t
    else:
        t = None
        # return nothing if neither of the above two conditions are met.
        return t


def read_data(t_filename, name_is_source_id=False, type_coord='icrs', gaia_sys=True):
    '''Read input data and convert to an astropy table ready for analysis
    
    | The input data must be in the form of a comma-separated variable and may take 3 forms:
    | (a) a 1-column table of source identifiers
    | (b) a 2-column table of decimal sky coordinates (celestial or galactic)
    | (c) a pre-prepared table of 5 columns consisting of source_id, ra, dec, parallax, Gmag (without column headers)
    
    Note that a single target can be quickly analysed directly from the command line by using option (a) with double-quotation marks around the source identifier.
    
    E.G. >>> python run_tess_cutouts 0 0 0 files "#AB Doradus"
    
    parameters
    ----------
    t_filename : `astropy.table.Table`
        name of the file containing the input data
    name_is_source_id : `bool`, optional, Default=False
        when running option (c), the "name" column will automatically be set as
        the Gaia DR3 identifiers if this parameter is True. This avoids long sql
        queries for very large input tables.
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input data. Choose from 'icrs', 'galactic' or
        'barycentricmeanecliptic', where the latter is the conventional coordinate
        system used by TESS.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination can
        be calculated if this is False.

    returns
    -------
    t_targets : `astropy.table.Table`
        a formatted astropy table ready for further analysis
    '''
    
    logger.info(f"Starting Time: {start}")
    if isinstance(t_filename, str):
        t_input = ascii.read(t_filename, delimiter=',', format='no_header')
    elif isinstance(t_filename, Table):
        t_input = t_filename

    t_targets = get_gaia_data(t_input, name_is_source_id=name_is_source_id, type_coord=type_coord, gaia_sys=gaia_sys)

    return t_targets


def collect_contamination_data(t_targets, flux_con, con_file, Rad=1., gaia_sys=True):
    '''Collect data on contamination sources around selected targets

    This function takes a target table and, if requested, prints out details of
    the total flux contribution from neighbouring contaminating sources for
    each target.
    It also returns a table (in descending order) of (up to) 5 neighbouring
    contaminants that contribute the most flux in the target aperture, if
    requested.
    
    Parameters
    ----------
    t_targets : `astropy.table.Table`
        target table with columns "name, source_id, ra, dec, parallax, Gmag".
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    con_file : `str`
        The name of the file to store data from the total flux contribution
        from contaminants.
    Rad : `float`, optional, default=True
        The aperture radius from the aperture photometry (in pixels)
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination can
        be calculated if this is False.

    Returns
    -------
    t_targets : `astropy.table.Table`
        Input target table with 3 columns added containing details of the
        contamination: "log_tot_bg", "log_max_bg", "num_tot_bg"
    '''
    if gaia_sys & flux_con:
        t_targets, t_con_table = contamination(t_targets, Rad=Rad)
        t_contam = t_targets[['source_id', 'log_tot_bg', 'log_max_bg',\
                              'num_tot_bg']]
        t_contam.write(con_file+'.ecsv', overwrite=True)
        if t_con_table is not None:
            t_con_table.write(con_file+
                              '_individiual.ecsv', overwrite=True)
    else:
        t_targets["log_tot_bg"] = -999.
        t_targets["log_max_bg"] = -999.
        t_targets["num_tot_bg"] = -999.
        if os.path.exists(con_file+'.ecsv'):
            t_contam = Table.read(con_file+'.ecsv')
            for i in range(len(t_contam)):
                g = (t_contam["source_id"] == \
                     t_targets["source_id"].astype(str)[i])
                if len(g) >= 1:
                    t_targets["log_tot_bg"][i] = t_contam["log_tot_bg"][g][0]
                    t_targets["log_max_bg"][i] = t_contam["log_max_bg"][g][0]
                    t_targets["num_tot_bg"][i] = t_contam["num_tot_bg"][g][0]
    return t_targets



def make_datarow(t_target, scc, d_target, labels_cont):
    '''Once the tessilator has analysed a target, the results are printed line
    by line to a table.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The Gaia and contamination details (see 'getGaiaData' and
        'contamination' functions in tessilator.tess_functions.)
    d_target : `dict`
        The dictionary containing details of the periodogram analysis, which is
        returned by 'run_LS' in tessilator.tess_functions.)
    scc : `list`, size=3
        A list containing the Sector, Camera and CCD.
    labels_cont : `str`
        A string of labels listing details of any contaminant sources.
        
    returns
    -------
    dr : `dict`
        A dictionary entry for the target star containing tessilator data
    '''
    dr = [
          t_target["name"],
          t_target["source_id"],
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
          d_target['AIC_line'],          
          d_target['AIC_sine'],
          d_target['pops_vals'][1],
          d_target['phase_scatter'],
          d_target['phase_chisq'],
          d_target['frac_phase_outliers'],
          d_target['Ndata'],
          d_target['jump_flag'],
          labels_cont
          ]
    return dr

def make_failrow(t_target, scc):
    '''Print a line with input data if tessilator fails for a given target

    parameters
    ----------
    t_targets : `astropy.table.Table`
        | Table of input data for the tessilator, with the following columns:
        #. name: name of the target (str)
        #. source_id: Gaia DR3 source identifier (str)
        #. ra: right ascension
        #. dec: declination
        #. parallax: parallax
        #. Gmag: Gaia DR3 apparent G-band magnitude
        #. log_tot_bg_star: log-10 value of the flux ratio between contaminants and target (optional)
        #. log_max_bg_star: log-10 value of the flux ratio between the largest contaminant and target (optional)
        #. n_contaminants: number of contaminant sources (optional)
        
        Note that if the contaminantion is not calculated, the final three columns are automatically
        filled with -999 values.
    scc : `list`, size=3
        A list containing the Sector, Camera and CCD.

    returns
    -------
    dr : `dict`
        A dictionary entry for the target star containing input data
    '''
    if 'log_tot_bg' not in t_target.colnames:
        t_targets.add_column(-999, name='log_tot_bg')
        t_targets.add_column(-999, name='log_max_bg')
        t_targets.add_column(0,    name='n_contaminants')

    dr = [
          t_target["name"],
          t_target["source_id"],
          t_target["ra"],
          t_target["dec"],
          t_target["parallax"],
          t_target["Gmag"],
          scc[0],
          scc[1],
          scc[2],
          t_target["log_tot_bg"],
          t_target["log_max_bg"],
          t_target["num_tot_bg"]]
    for i in range(13):
        dr.append(np.nan)
    dr.append(0)
    dr.append(0)
    dr.append('z')
    return dr


def fix_noise_lc_local(targ_lc, med_lc, scc, targ_name, make_plots=False):
    '''Apply a flux-corrected key to the target lightcurve dictionary, and make
    a plot of the corrections if required.
    
    This function uses the lightcurve data from neighbouring sources.
    
    parameters
    ----------
    targ_lc : `dict`
        a dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    med_lc : `dict`
        the "median" lightcurve data from the neigbourhing sources
    scc : `list`, size=3
        List containing the sector number, camera and CCD
    targ_name : `str`
        The name of the target
    make_plots : `bool`, optional, default=False
        Choose to make plots of the noise corrections

    returns
    -------
    targ_lc : `dict`
        The updated target lightcurve dictionary, with an extra key containing
        the noise-corrected flux.
    '''
    targ_time = np.array([t-min(targ_lc["time"]) for t in targ_lc["time"]])
    sim_flux = []
    for t in range(len(targ_time)):
        sim_bit = np.interp(targ_time[t], med_lc["time"], med_lc["nflux"])
        targ_lc["nflux_corr"].append(targ_lc["nflux"][t]/sim_bit)
        sim_flux.append(sim_bit)
    if make_plots:
        plot_name = f"{targ_name}_{scc[0]:04d}_{scc[1]}_{scc[2]}_corr_local_lc.png"
        make_lc_corr_plot(plot_name, targ_time, targ_lc["nflux"], sim_flux)
    return targ_lc


def fix_noise_lc_sim(targ_lc, targ_name, t_targets, scc, mag_extr_lim=3., make_plots=False):
    '''Divide the target lightcurve by the noise lightcurve
    
    For a given sector, camera and CCD configuration, this
    function will search for the simulated noisy lightcurve 
    and divide the target lightcurve by the noise lightcurve,
    which should mitigate some of the systematic features.
    
    The noisy lightcurves are described in the tessimulation.py
    module.
    
    parameters
    ----------
    targ_lc : `dict`
        a dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    targ_name : `str`
        The name of the target
    t_targets : `astropy.row.Row`
        details of the target star   
    scc : `list`
        a list containing the sector number, camera and CCD    
    mag_extr_lim : `float`, optional, default=3.
        The tolerated difference in magnitude between the target
        and the range of calculated simulated lightcurves if the
        magnitude is out of range.
    make_plots : `bool`, optional, default=False
        Choose to make plots of the noise corrections

    returns
    -------
    targ_lc : `dict`
        The updated target lightcurve dictionary, with an extra key containing
        the noise-corrected flux.
    '''
    mag_files = sorted(glob(f'./tesssim/lc/{scc[0]:02d}_{scc[1]}_{scc[2]}/mag*'))
    if not mag_files:
        logger.warning(f"No simulated lightcurve for {targ_name}, Sector {scc[0]:02d}, Camera {scc[1]}, CCD {scc[2]}")
        return targ_lc

    mag_target = t_targets['Gmag'][0]
    mag0 = np.array([float(mag_file.split("_")[-2]) for mag_file in mag_files])
    mag1 = np.array([float(mag_file.split("_")[-1]) for mag_file in mag_files])
    g = np.where((mag0 <= mag_target) & (mag1 > mag_target))[0]

    if g.size > 0:
        sim_tab = f'{mag_files[g[0]]}/flux_fin.csv'
    elif (mag_target < mag0[0]) & (np.abs(mag0[0] - mag_target) < mag_extr_lim):
        logger.info(f"Target {targ_name} is brighter than the magnitude range of simulated files, but within the extrapolation threshold")
        sim_tab = f'{mag_files[0]}/flux_fin.csv'
    elif (mag_target > mag1[-1]) & (np.abs(mag1[-1] - mag_target) < mag_extr_lim):
        logger.info(f"Target {targ_name} is fainter than the magnitude range of simulated files, but within the extrapolation threshold")
        sim_tab = f'{mag_files[-1]}/flux_fin.csv'
    else:
        logger.warning(f"Target {targ_name} is out of the magnitude range of simulated files, and out of the extrapolation threshold")
        return targ_lc

    sim_lc = Table.read(sim_tab)
    targ_time = np.array([t-min(targ_lc["time_o"]) for t in targ_lc["time_o"]])
    sim_flux = []
    for t in range(len(targ_time)):
        sim_bit = np.interp(targ_time[t], sim_lc["time"], sim_lc["nflux"])
        targ_lc["nflux_corr"].append(targ_lc["nflux"][t]/sim_bit)
        sim_flux.append(sim_bit)
    if make_plots:
        plot_name = f"{targ_name}_{scc[0]:04d}_{scc[1]}_{scc[2]}_corr_sim_lc.png"
        make_lc_corr_plot(plot_name, targ_time, targ_lc["nflux"], sim_flux)
    return targ_lc




def make_lc_corr_plot(plot_name, targ_time, targ_flux, sim_flux, im_dir='plots'):
    '''Make a plot of the noise corrections to the lightcurve.
    
    parameters
    ----------
    plot_name : `str`
        The file name which will be used to save the plot.
    targ_time : `Iterable`
        The time coordinate of the data
    targ_flux : `Iterable`
        The flux coordinate of the target data
    sim_flux : `Iterable`
        The flux coordinate of the simulated noisy lightcurve.
    im_dir : `str`, optional, default='plots'
        The directory to save the plots.

    returns
    -------
    None. Plots are saved to file.
    '''    
    fig, ax = plt.subplots(figsize=(15,7))
    mpl.rcParams.update({'font.size': 20})
    ax.set_xlabel('time [days]')
    ax.set_ylabel('normalised flux')
    ax.plot(targ_time, targ_flux, '.', c='r', label='target flux')
    ax.plot(targ_time, sim_flux, '.', c='g', label='systematic flux')
    ax.legend()
    path_exist = os.path.exists(f'./{im_dir}')
    if not path_exist:
        os.mkdir(f'./{im_dir}')
    plt.savefig(f'./{im_dir}/{plot_name}', bbox_inches='tight')
    plt.close('all')



def get_name_target(t_target):
    '''Quick function to capture a string containing the name of the target
    
    parameters
    ----------
    t_target : `astropy.row.Row`
        The target data, which contains the target name
        
    returns
    -------
    name_target : `str`
        The formatted target name to be used for reference.
    '''
    name_target = t_target.replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]
    return name_target



def get_median_lc(files, directory, scc, n_bin=10):
    '''A function that makes the final noisy lightcurve
    
    For a given number of lightcurves, this function
    calculates the median flux at each time step if there are
    more than 2 measurements (or the mean if 2 or less).
    
    parameters
    ----------
    files : `list`
        a list of lightcurves in a given directory
    directory : `str`
        the name of the directory containing the lightcurves
    scc : `list`
        a list containing the sector number, camera and CCD        
    n_bin : `int`, optional, default=10
        the maximum number of lightcurves to be used in the analysis.
        
    returns
    -------
    t_fin : `astropy.table.Table`
        a table containing the data for the final noisy lightcurve
    '''
    f_name = files[0].split("/")[2].split("_")[1]
    num, time, flux, eflux = [], [], [], []
    if len(files) > n_bin:
        chosen_indices = np.random.choice(len(files), n_bin)
        files_chosen = [files[n] for n in chosen_indices]
    else:
        files_chosen = files
    for f, file in enumerate(files_chosen):
        tab = ascii.read(file)
        tab['enflux'][np.where(tab['enflux'] < 0)] = .01
        for t in tab:
            num.append(f+1)
            time.append(t['time'])
            flux.append(t['nflux'])
            eflux.append(t['enflux'])
    t_uniq = np.unique(np.array(time))
    t_fin = Table(names=('time', 'nflux', 'enflux', 'n_lc'), dtype=(float, float, float, int))
    for t in t_uniq:
        g = np.where(time == t)[0]
        flux_med = np.median(np.array(flux)[g])
        eflux_med = np.median(np.array(eflux)[g])
        flux_mean = np.mean(np.array(flux)[g])
        eflux_mean = np.mean(np.array(eflux)[g])
        num_lc = len(g)
        if num_lc > 2:
            t_fin.add_row([t, flux_med, eflux_med, num_lc])
        else:
            t_fin.add_row([t, flux_mean, eflux_mean, num_lc])
    t_fin.write(f'./{directory}/flux_fin_{f_name}_{scc[0]:04d}_{scc[1]}_{scc[2]}.csv', overwrite=True)
    return t_fin




def full_run_lc(file_in, t_target, make_plots, scc, final_table, cutout_size=20, store_lc=False, lc_dir='lc', keep_data=False, flux_con=False, LC_con=False, con_file=False, XY_pos=(10.,10.), Rad=1., SkyRad=[6.,8.], fix_noise=False):
    '''Aperture photometry, lightcurve cleaning and periodogram analysis.

    This function calls a set of functions in the lc_analysis.py module to
    perform aperture photometry, clean the lightcurves from spurious data and
    runs the Lomb-Scargle periodogram to measure rotation periods.

    parameters
    ----------
    file_in : `str`
        name of the input TESS fits file
    t_target : `astropy.table.Table`
        details of the target star
    make_plots: `bool`
        decides if plots are made 
    scc : `list`, size=3
        sector, camera, ccd
    cutout_size : `int`
        the pixel length of the downloaded cutout
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True
    keep_data : `bool`
        Choose to save the input data to file.
    flux_con : `bool`, optional, default=False
        Decides if the flux contribution from contaminants is to be calculated.
    LC_con : `bool`, optional, default=0
        Decides if a lightcurve analysis is to be performed for the n strongest
        contaminants, where n is a keyword in the "contamination" function from
        tess_functions2.py.
    con_file : `str`
        The name of the file to store data from the total flux contribution
        from contaminants.
    XY_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    Rad : `float`, optional, default=1.0
        The aperture radius from the aperture photometry
    SkyRad : `Iterable`, optional, default=[6.,8.]
        The inner and outer background annuli from aperture photometry
    fix_noise : `bool`, optional, default=False
        Choose to apply corrections accounting for systematic noise.

    returns
    -------
    * A data entry for the final period file.
    * A plot of the lightcurve (if requested).
    '''
    nc = 'nc'
    print(file_in, t_target)
    try:
        tpf = aper_run(file_in, t_target, Rad=Rad,
                       SkyRad=SkyRad, XY_pos=XY_pos)
    except Exception as e:
        logger.error(f"aperture photometry: of {file_in} failed to run")
    if len(tpf) < 10:
        logger.error(f"aperture photometry: failed to produce enough data "
                     f"points for {t_target['source_id']}")
        for t in t_target:
            final_table.add_row(make_failrow(t, scc))
        return None
    phot_targets = tpf.group_by('id')
    for key, group in zip(phot_targets.groups.keys, phot_targets.groups):
        g_c = group[group["flux"] > 0.0]
        if isinstance(t_target, Table):
            t_targets = t_target[t_target["source_id"] == key[0]]
        else:
            t_targets = Table(t_target)
            t_targets["source_id"] = t_targets["source_id"].astype(str)
            
        name_target = get_name_target(t_targets["name"][0])

        if len(g_c) >= 50:
            name_lc = f'lc_{name_target}_{scc[0]:02d}_{scc[1]}_{scc[2]}.csv'
            clean_norm_lc, original_norm_lc = make_lc(g_c, name_lc, store_lc=store_lc, lc_dir=lc_dir)
        else:
            logger.error(f"No photometry was recorded for this group.")
            final_table.add_row(make_failrow(t_targets, scc))
            continue
        if len(clean_norm_lc) == 0:
            logger.error(f"no datapoints to make lightcurve analysis for "
                         f"{t_targets['source_id']}")
            final_table.add_row(make_failrow(t_targets, scc))
            continue

        if fix_noise and not LC_con:
            logger.info('fixing the noise!')
            clean_norm_lc = fix_noise_lc_sim(clean_norm_lc, name_target, t_targets, scc, make_plots=make_plots)
            nc = 'corr_sim'
        d_target = run_ls(clean_norm_lc, check_jump=True)
        if d_target['period_best'] == -999:
            logger.error(f"the periodogram did not return any results for "
                         f"{t_targets['source_id']}")
            final_table.add_row(make_failrow(t_targets, scc))
            continue

        
        if LC_con:
            lc_cont_dir = 'lc_cont'
            if flux_con != 1:
                logger.warning("Contaminants not identified! Please toggle LC_con=1")
                logger.warning("Continuing program using only the target.")
            else:
                logger.info('calculating contaminant lightcurves')
                ind_file = f"./{con_file}_individiual.ecsv"
                if os.path.isfile(ind_file):
                    con_table = ascii.read(con_file+
                                           '_individiual.ecsv')
                    con_table = con_table[con_table["source_id_target"] == \
                                          t_targets["source_id"].astype(str)]
                    XY_con = find_xy_cont(file_in, con_table, cutout_size)
                    labels_cont = ''
                    lc_cont_files = []
                    for z in range(len(XY_con)):
                        name_lc, lab_lc, d_lc = run_test_for_contaminant(XY_con[z],\
                                                                         file_in,\
                                                                         con_table[z],\
                                                                         d_target, scc, lc_cont_dir=lc_cont_dir)
                        labels_cont += lab_lc
                        if d_lc is not None:
                            if d_lc["AIC_sine"] > d_lc["AIC_line"]+1:
                                lc_cont_files.append(f'./{lc_cont_dir}/{name_lc}')
                    if not labels_cont:
                        labels_cont = 'e'
                else:
                    labels_cont = 'o'
                    XY_con = None
                if fix_noise:
                    logger.info('fixing the noise!')
                    if len(lc_cont_files) > 0:
                        nc = 'corr_local'
                        t_median_lc = get_median_lc(lc_cont_files, lc_cont_dir, scc)
                        clean_norm_lc = fix_noise_lc_local(clean_norm_lc, t_median_lc, scc, name_target, make_plots=make_plots)
                        d_target = run_ls(clean_norm_lc, check_jump=True)
                    else:
                        nc = 'corr_sim'
                        clean_norm_lc = fix_noise_lc_sim(clean_norm_lc, name_target, t_targets, scc, make_plots=make_plots)
                        d_target = run_ls(clean_norm_lc, check_jump=True)
                else:
                    nc = 'local'
        else:
            labels_cont = 'z'
            XY_con = None
        if make_plots:
            im_plot, XY_ctr = make_2d_cutout(file_in, group, \
                                             im_size=(cutout_size+1,\
                                                      cutout_size+1))
            make_plot(im_plot, clean_norm_lc, original_norm_lc,\
                         d_target, scc, t_targets, name_target, XY_contam=XY_con,\
                         p_min_thresh=0.1, p_max_thresh=50., Rad=1.0,\
                         SkyRad=[6.,8.], nc=nc)
                         
        final_table.add_row(make_datarow(t_targets, scc, d_target,\
                                         labels_cont))
        if not keep_data:
            if len(file_in) == 1:
                os.remove(file_in)



def print_time_taken(start, finish):
    '''Calculate the time taken for a process.

    This function takes a start and finish point a calculates the time taken in
    hours, minutes and seconds

    parameters
    ----------
    start : `datetime.datetime`
        The start point of the process
    
    finish : `datetime.datetime`
        The end point of the process

    returns
    -------
    time_taken : `str`
        The time taken for the process to complete
    '''
    time_in_secs = (finish - start).seconds
    mins, secs = divmod(time_in_secs, 60)
    hrs, mins = divmod(mins, 60)
    time_taken = f"{hrs} hours, {mins} minutes, {secs} seconds"
    return time_taken





def find_xy_cont(f_file, con_table, cutout_size):
    '''Identify the pixel X-Y positions for contaminant sources.

    If the user requests a periodogram analysis of neighbouring potential
    contaminants (LC_con=1), this function returns their X-Y positions, which are
    used as the centroids for aperture photometry.

    parameters
    ----------
    f_file : `str`
        The name of the fits file.
    con_table : `astropy.table.Table`
        The table containing Gaia data of the contaminants.
    cutout_size : `int`
        The length size of the TESS cutout image.

    returns
    -------
    cont_positions : `np.array`
        A 2-column array of the X-Y positions of the contaminants.
    '''
    XY_ctr = (cutout_size/2., cutout_size/2.)
    with fits.open(f_file) as hdul:
        head = hdul[0].header
        RA_ctr, DEC_ctr = head["RA_OBJ"], head["DEC_OBJ"]
        RA_con, DEC_con = con_table["RA"], con_table["DEC"]
        X_abs_con, Y_abs_con = [], []
        _, _, _, _, _, _, Col_ctr, Row_ctr, _ = \
            tess_stars2px_function_entry(1, RA_ctr, DEC_ctr,\
                                         trySector=head["SECTOR"])
        for i in range(len(RA_con)):
            _, _, _, _, _, _, Col_con, Row_con, _ = \
                tess_stars2px_function_entry(1, RA_con[i], DEC_con[i],\
                                             trySector=head["SECTOR"])
            X_abs_con.append(Col_con[0])
            Y_abs_con.append(Row_con[0])
        X_con = np.array(X_abs_con - Col_ctr[0]).flatten() + XY_ctr[0]
        Y_con = np.array(Y_abs_con - Row_ctr[0]).flatten() + XY_ctr[1]
        cont_positions = np.array([X_con, Y_con]).T
        return cont_positions
        

def run_test_for_contaminant(XY_arr, file_in, con_table, d_target, scc, store_lc=True, lc_cont_dir='lc_cont'):
    '''Run the periodogram analyses for neighbouring contaminants if required.

    parameters
    ----------
    XY_arr : `list`, size=2
        The X and Y positions of the contaminant (the output form the "find_XY_cont" function).
    file_in : `str`
        The name of the fits file containing the contaminant.
    con_table : `astropy.table.Table`
        A single row from the contamination table which has
        details of the flux contribution.
    d_target : `dict`
        The dictionary returned from the periodogram analysis of
        the target star (the output from the "run_LS" function in the lc_analysis.py module)

    returns
    -------
    labels_cont : `str` (a, b, c or d)
        A single character which assess if the calculated period for the target
        could actually come from the contaminant. 
    '''

    XY_con = tuple((XY_arr[0], XY_arr[1]))
    phot_cont = aper_run(file_in, con_table, Rad=1.,
                         SkyRad=(6.,8.), XY_pos=XY_con)
    clean_norm_lc_cont, name_lc, d_cont = None, None, None
    if phot_cont is None:
        labels_cont = 'd'
    else:
        name_lc = f'lc_{con_table["source_id_target"]}_{con_table["source_id"]}_{scc[0]:04d}_{scc[1]}_{scc[2]}.csv'
        clean_norm_lc_cont, original_norm_lc_cont = make_lc(phot_cont, name_lc, store_lc=True, lc_dir=lc_cont_dir)
        if len(clean_norm_lc_cont) != 0:
            d_cont = run_ls(clean_norm_lc_cont)
            labels_cont = is_period_cont(d_target, d_cont, con_table)
        else:
            labels_cont = 'd'
    logger.info(f"label for this contaminant: {labels_cont}")
    return name_lc, labels_cont, d_cont







def get_tess_pixel_xy(t_targets):
    '''Get the pixel X-Y positions for all targets in a Sector/Camera/CCD mode.

    For a given pair of celestial sky coordinates, this function returns table
    rows containing the sector, camera, CCD, and X/Y position of the full-frame
    image fits file, so that all stars located in a given (large) fits file can
    be processed simultaneously. After the table is returned, the
    input table is joined to the input table on the source_id, to ensure this
    function only needs to be called once.

    This function is only used if the tessilator program runs over the calibrate
    full-frame images - i.e., when the "all_cc" function is called.
    
    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table created by the function get_gaia_data.py

    returns
    -------
    xy_table : `astropy.table.Table`
        Output table containing the (*x*, *y*) pixel positions for each target.
    '''
    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
           outCPix, outRPix, scinfo = tess_stars2px_function_entry(
           t_targets['source_id'], t_targets['ra'], t_targets['dec'])
    xy_table = Table([outID, outSec, outCam, outCcd, outCPix, outRPix],
            names=('source_id', 'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos'))
    return xy_table


def get_fits(scc):
    '''Function which returns a list of fits files corresponding to a
    given Sector, Camera and CCD configuration

    parameters
    ----------
    sector_num : `int`
        Sector number required

    scc : `list`, size=3
        List of [a, b, c], where a is the Sector number, b is the Camera
        number (1-4), and c is the CCD number (1-4)
    
    file_dir : `str`
        The name of the base directory containin the fits files.

    returns
    -------
    fits_files : `list`
        A list of the fits files to be used for aperture photometry
    '''
    
    list_fits = sorted(glob(f"./tess_fits_files/sector{scc[0]:02d}/*.fits"))
    l_cam = np.array([int(j.split('-')[2]) for j in list_fits])
    l_ccd = np.array([int(j.split('-')[3]) for j in list_fits])
    fits_indices = (l_cam == scc[1]) & (l_ccd == scc[2])
    fits_files = np.array(list_fits)[fits_indices]
    print(fits_files)
    return fits_files


def make_2d_cutout(file_in, phot_table, im_size=(20,20)):
    '''Makes a 2D cutout object of a target using the median time-stacked image

    parameters
    ----------
    file_in : `astropy.table.Table`
        The astropy table containing the output from the aperture photometry
        for a given target.
    phot_table : `list`
        The list of fits files used to make the aperture photometry
    im_size : `tuple`, optional, default=(20,20)
        The required size of the 2D-cutout object

    returns
    -------
    cutout : `astropy.nddata.Cutout2D`
        A 2D-cutout object
    XY_ctr : `tuple`
        A tuple containing the X, Y position of the median time-stacked image.
    '''

    if isinstance(file_in, np.ndarray):
        image_index = math.floor((len(phot_table))/2)
        image_fits  = file_in[image_index]
        table_slice = phot_table[image_index]
        with fits.open(image_fits) as hdul:
            data = hdul[1].data
            head = hdul[1].header
            error = hdul[2].data
        ctr_pt = (table_slice["xcenter"], table_slice["ycenter"])
        cutout = Cutout2D(data, ctr_pt, im_size)
    elif isinstance(file_in, str):
        with fits.open(file_in) as hdul:
            data = hdul[1].data
        data_slice = data["FLUX"][:][:][int(data.shape[0]/2)]
        ctr_pt = ((im_size[0]-1)/2., (im_size[1]-1)/2.)
        cutout = Cutout2D(data_slice, ctr_pt, im_size)
    else:
        logger.error("Fits file {file_in} has the invalid type: {type(file_in)}")
        return None
    return cutout, ctr_pt




def cutout_allsecs(coord, cutout_size, name_target, tot_attempts=3, cap_files=None, fits_dir='fits'):
    '''Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.
    
    This function in particular will return all available sectors of data.
    
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    cutout_size : `float`
        The pixel length of the downloaded cutout
    name_target : `str`
        Name of the target
    tot_attempts : `int`, optional, default=3
        The number of attempts to download the fits files in case of request or
        server errors
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.

    returns
    -------
    manifest : `list`
        A list of the fits files for lightcurve analysis.
    '''
    num_attempts, manifest = 0, []
    while num_attempts < tot_attempts:
        logger.info(f'attempting download request {num_attempts+1} of {tot_attempts}...')
        try:
            sectors = Tesscut.get_sectors(coordinates=coord)['sector'].data
            if len(sectors) == 0:
                logger.error(f'Sorry, no TESS data available for {name_target}')
                break
            np.random.shuffle(sectors)
            if cap_files:
                sectors=sectors[:cap_files]
            for n_s, sec in enumerate(sectors):
                logger.info(f'getting sector {sec}, {n_s+1} of {len(sectors)}')
                s = f'{sec:04d}'
                fits_file = glob(f'./{fits_dir}/{name_target}_{s}*')
                if fits_file:
                    manifest.append(fits_file[0])
                else:
                    dl = Tesscut.download_cutouts(coordinates=coord,\
                                                  size=cutout_size,\
                                                  sector=s,\
                                                  path=fits_dir)
                    manifest.append(dl['Local Path'][0])
            if manifest:
                logger.info(f'...done!')
                break
        except:
            num_attempts += 1
    if num_attempts == tot_attempts:
        logger.error(f"Timeout error for {name_target}")
    return manifest



def cutout_onesec(coord, cutout_size, name_target, choose_sec, tot_attempts=3, cap_files=None, fits_dir='fits'):
    '''Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.

    This function in particular will return just one selected sector of data.
    
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    cutout_size : `float`
        The pixel length of the downloaded cutout
    target_name : `str`
        Name of the target
    choose_sec : `int`
        The number of the TESS sector required for download.
    tot_attempts : `int`, optional, default=3
        The number of attempts to download the fits files in case of request or
        server errors
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.

    returns
    -------
    manifest : `list`
        A list of the fits files for lightcurve analysis.
    '''
    manifest = []
    if (choose_sec < 1 or choose_sec > 70):
        print(f"Sector {choose_sec} is out of range.")
        logger.error(f"Sector {choose_sec} is out of range.")
        return manifest
    else:
        num_attempts = 0
        while num_attempt < tot_attempts:
            try:
                fits_file = glob(f'./{fits_dir}/{name_target}_{choose_sec:04d}*')
                if fits_file:
                     manifest.append(fits_file[0])
                else:
                    dl = Tesscut.download_cutouts(coordinates=coord,\
                                                  size=cutout_size,\
                                                  sector=choose_sec,\
                                                  path=fits_dir)
                    manifest.append(dl["Local Path"][0])
                return manifest
            except:
                print(f"Didn't get data for {name_target} in {choose_sec}, attempt {num_attempt+1}")
                logger.error(f"Didn't get data for {name_target} in {choose_sec}, attempt {num_attempt+1}")
                num_attempts += 1
        if num_attempts == tot_attempts:
            print(f"No data found for {name_target} in {choose_sec}")
            logger.error(f"No data found for {name_target} in {choose_sec}")
            return manifest


def cutout_chosensecs(coord, cutout_size, name_target, choose_sec, tot_attempts=3, cap_files=None, fits_dir='fits'):
    '''Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.
    
    This function in particular will return the sectors of data defined from a list of integers.
    
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    cutout_size : `float`
        The pixel length of the downloaded cutout
    name_target : `str`
        Name of the target
    choose_sec : `Iterable`
        The TESS sectors required for download.
    tot_attempts : `int`, optional, default=3
        The number of attempts to download the fits files in case of request or
        server errors
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.

    returns
    -------
    manifest : `list`
        A list of the fits files for lightcurve analysis.
    '''
    manifest = []
    cs = np.array(list(set(choose_sec)))
    if all(isinstance(x, np.int64) for x in cs):
        cs_g = cs[np.where((cs > 0) & (cs < 70))[0]]
        if len(cs) != len(cs_g):
            logger.warning(f"Sectors {np.setdiff1d(cs, cs_g)} are out of "
                           f"range.")
        if cap_files:
            cs_g = cs_g[:cap_files]

        c_fail = []
        for c in cs_g:
            num_attempts = 0
            while num_attempts < tot_attempts:
                print(f'attempting download request {num_attempts+1} of {tot_attempts}...')
                try:
                    fits_file = glob(f'./{fits_dir}/{name_target}_{c}*')[0]
                    if os.exists(fits_file):
                        manifest.append(fits_file)
                    else:
                        dl = Tesscut.download_cutouts(coordinates=coord,\
                                                      size=cutout_size,\
                                                      sector=c,\
                                                      path=fits_dir)
                        manifest.append(dl["Local Path"][0])
                except:
                    print(f"Didn't get Sector {c} data for {name_target}, attempt {num_attempts+1} of {tot_attempts}")
                    logger.error(f"Didn't get Sector {c} data for {name_target}, attempt {num_attempts+1} of {tot_attempts}")
                if num_attempts == tot_attempts:
                    print(f"No data for {name_target} in Sector {c}")
                    logger.error(f"No data for {name_target} in Sector {c}")
                    c_fail.append(c)
        return manifest
    else:
        print("Some sectors not of type `int'. Fix and try again.")
        logger.error("Some sectors not of type `int'. Fix and try again.")
        return manifest


def get_cutouts(coord, cutout_size, name_target, choose_sec=None, tot_attempts=3, cap_files=None, fits_dir='fits'):
    '''Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.
    
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    cutout_size : `float`
        The pixel length of the downloaded cutout
    choose_sec : `None`, `int` or `Iterable`
        | The sector, or sectors required for download.
        * If `None`, TESScut will download all sectors available for the target.
        * If `int`, TESScut will attempt to download this sector number.
        * If `Iterable`, TESScut will attempt to download a list of sectors.
    name_target : `str`
        Name of the target
    tot_attempts : `int`, optional, default=3
        The number of attempts to download the fits files in case of request or
        server errors
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.

    returns
    -------
    manifest : `list`
        A list of the fits files for lightcurve analysis.
    '''
    name_target = get_name_target(name_target)
    if choose_sec is None:
        manifest = cutout_allsecs(coord, cutout_size, name_target, tot_attempts=tot_attempts, cap_files=cap_files, fits_dir=fits_dir)
    elif isinstance(choose_sec, int):
        manifest = cutout_onesec(coord, cutout_size, name_target, tot_attempts=tot_attempts, cap_files=cap_files, fits_dir=fits_dir)
    elif isinstance(choose_sec, Iterable):
        manifest = cutout_chosensecs(coord, cutout_size, name_target, choose_sec, tot_attempts=tot_attempts, cap_files=cap_files, fits_dir=fits_dir)
    else:
        print(f"The choose_sec parameter has an invalid format type: "
              f"{type(choose_sec)}")
        logger.error(f"The choose_sec parameter has an invalid format type: "
                     f"{type(choose_sec)}")
        manifest = []
    return manifest

def one_source_cutout(target, LC_con, flux_con, con_file, make_plots,\
                      final_table, choose_sec=None, store_lc=False, cutout_size=20, \
                      tot_attempts=3, cap_files=None, fits_dir='fits', lc_dir='lc', fix_noise=False):
    '''Download cutouts and run lightcurve/periodogram analysis for one target.

    Called by the function "all_sources".

    parameters
    ----------
    target : `astropy.table.row.Row`
        A row of data from the astropy table
    LC_con : `bool`
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    con_file : `str`
        The name of the file to store data from the total flux contribution
        from contaminants.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    final_table : `astropy.table.Table`
        The table to store the final tessilator results.
    choose_sec : `None`, `int` or `Iterable`, optional, default=None
        | The sector, or sectors required for download.
        * If `None`, TESScut will download all sectors available for the target.
        * If `int`, TESScut will attempt to download this sector number.
        * If `Iterable`, TESScut will attempt to download a list of sectors.
    cutout_size : `float`, optional, default=20
        the pixel length of the downloaded cutout
    tot_attempts : `int`, optional, default=3
        The number of attempts to download the fits files in case of request or
        server errors
    cap_files : `None`, `int`, optional, default=None
        The maximum number of sectors for each target.
    targ_name : `G` or `T`, optional, default=G
        The prefix in the names of the fits files are changed to either the Gaia source
        identifier if targ_name = G, or the target name if targ_name = T.

    returns
    -------
    Nothing returned. Results are saved to table and plots are generated (if
    specified). 
    '''
    # Set the contaminant parameters to the default values in case
    # they have not been added
    if 'log_tot_bg' not in target.colnames:
        target.add_column(-999, name='log_tot_bg')
        target.add_column(-999, name='log_max_bg')
        target.add_column(0,    name='n_contaminants')
    name_target = target['name'].replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]

    coo = SkyCoord(target["ra"], target["dec"], unit="deg")

    # use Tesscut to get the cutout fits files for the target star
    # there may be more than 1 fits file if the target lands in
    # multiple sectors!
    fits_files = get_cutouts(coo, cutout_size, name_target, tot_attempts=tot_attempts, choose_sec=choose_sec, cap_files=cap_files, fits_dir='fits')
    if fits_files is None:
        logger.error(f"could not download any data for {target['name']}. "
                     f"Trying next target.")
    else:
        for m, file_in in enumerate(fits_files):
            print(f'working on {file_in}, #{m+1} of {len(fits_files)}')
            try:
    # rename the fits file to something more legible for users
                f_sp = file_in.split('/')[-1].split('-')
                if (len(f_sp) >=3) & (f_sp[0] == 'tess'):
                    f_new = f'./{fits_dir}/'+'_'.join([name_target, f_sp[1][1:], f_sp[2],\
                                                       f_sp[3][0]])+'.fits'
                    os.rename(f'./{file_in}', f_new)
                    logger.info(f"target: {target['source_id']}, "
                                f"{m+1}/{len(fits_files)}")
    # run the lightcurve analysis for the given target/fits file
                elif len(f_sp) == 1:
                    f_new = f'./{fits_dir}/{f_sp[0]}'
                else:
                    f_new = file_in
                t_sp = f_new.split('_')
    # simply extract the sector, ccd and camera numbers from the fits file.
                scc = [int(t_sp[-3][1:]), int(t_sp[-2]), int(t_sp[-1][0])]
                full_run_lc(f_new, target, make_plots, scc, final_table, 
                                   flux_con=flux_con, store_lc=store_lc, LC_con=LC_con,
                                   con_file=con_file, XY_pos=(10.,10.), lc_dir=lc_dir, fix_noise=fix_noise)
            except Exception as e:
                logger.error(f"Error occurred when processing {file_in}. "
                             f"Trying next target.")

        
def all_sources_cutout(t_targets, period_file, LC_con, flux_con, con_file,\
                       make_plots, choose_sec=None, store_lc=False, tot_attempts=3, cap_files=None, res_dir='results', lc_dir='lc', fits_dir='fits', keep_data=False, fix_noise=False):
    '''Run the tessilator for all targets.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        | Table of input data for the tessilator, with the following columns:
        #. name: name of the target (str)
        #. source_id: Gaia DR3 source identifier (str)
        #. ra: right ascension
        #. dec: declination
        #. parallax: parallax
        #. Gmag: Gaia DR3 apparent G-band magnitude
        #. log_tot_bg_star: log-10 value of the flux ratio between contaminants and target (optional)
        #. log_max_bg_star: log-10 value of the flux ratio between the largest contaminant and target (optional)
        #. n_contaminants: number of contaminant sources (optional)
    period_file : `str` 
        Name of the file to store periodogram results.
    LC_con : `bool`
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    con_file : `str`
        The name of the file to store data from the total flux contribution
        from contaminants.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    choose_sec : `None`, `int`, or `Iterable`, optional, default=None
        | The sector, or sectors required for download.
        * If `None`, TESScut will download all sectors available for the target.
        * If `int`, TESScut will attempt to download this sector number.
        * If `Iterable`, TESScut will attempt to download a list of sectors.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    tot_attempts : `int`, optional, default=3
        The number of attempts to download the fits files in case of request or
        server errors
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    res_dir : `str`, optional, default='results'
        The directory to store the final results file.
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.
    keep_data : `bool`
        Choose to save the input data to file.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.

    returns
    -------
    Nothing returned. The final table is saved to file and the program
    terminates.
    '''
    fits_path_exist = os.path.exists(f'./{fits_dir}')
    if not fits_path_exist:
        os.makedirs(f'./{fits_dir}')

    final_table = create_table_template()
    if 'log_tot_bg' not in t_targets.colnames:
        t_targets.add_column(-999, name='log_tot_bg')
        t_targets.add_column(-999, name='log_max_bg')
        t_targets.add_column(0,    name='num_tot_bg')
    for i, target in enumerate(t_targets):
        print(f"{target['name']} (Gaia DR3 {target['source_id']}), star # {i+1} of {len(t_targets)}")
        one_source_cutout(target, LC_con, flux_con, con_file,
                          make_plots, final_table, store_lc=store_lc, choose_sec=choose_sec,
                          tot_attempts=tot_attempts, cap_files=cap_files, lc_dir=lc_dir, fix_noise=fix_noise)
    finish = datetime.now()
    dt_string = finish.strftime("%b-%d-%Y_%H:%M:%S")
    res_path_exist = os.path.exists(f'./{res_dir}')
    if not res_path_exist:
        os.makedirs(f'./{res_dir}')
    final_table.write(f'./{res_dir}/{period_file}_{dt_string}.ecsv')

    hrs_mins_secs = print_time_taken(start, finish)
    print(f"Finished {len(t_targets)} targets in {hrs_mins_secs}")



def one_cc(t_targets, scc, make_plots, final_table, Rad=1.0,
                      SkyRad=[6.,8.], keep_data=False, fix_noise=False):
    '''Run the tessilator for targets in a given Sector/Camera/CCD configuration

    This routine finds the full-frame calibrated fits files and targets which
    land in a given Sector/Camera/CCD configuration (SCC). Aperture photometry
    is carried out simultaneously for all stars in a given SCC for each fits
    file in chronological order. This makes the method run much faster than
    doing it star-by-star (i.e. vectorisation). The output is a table for each
    SCC and plots for each target (if required).

    parameters
    ----------
    cc : `list` size=2
        List of [a, b], where a is the Camera number (1-4)
        and b is the CCD number (1-4)
    t_targets : `astropy.table.Table`
        Table containing the targets to be analysed
    sector_num : `int`
        Sector number required
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    final_table : `astropy.table.Table`
        The table to store tessilator results.
    Rad : `float`, optional, default=1.0
        The pixel radius of the flux collecting area for aperture photometry
    SkyRad: `Iterable`, size=2, optional, default=[6.,8.]
        The inner and outer background annuli used for aperture photometry
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.

    returns
    -------
    Nothing returned, but the final tessilator table is saved to file.
    '''
    fits_files = get_fits(scc)
    ind = (t_targets['Sector'] == scc[0]) & \
          (t_targets['Camera'] == scc[1]) & \
          (t_targets['CCD'] == scc[2])
    if ind.any() == False:
        return
    print(fits_files)
    full_run_lc(fits_files, t_targets[ind], make_plots, scc, final_table, keep_data=keep_data, fix_noise=fix_noise)




def all_sources_sector(t_targets, scc, make_plots, period_file, keep_data=False, fix_noise=False):
    '''Iterate over all cameras and CCDs for a given sector
    
    parameters
    ----------
    t_targets : `astropy.table.Table`
        Input data for the targets to be analysed
    scc : `list`, size=3
        List containing the sector number, camera and CCD
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    period_file : `str`
        Name of file for recording parameters measured by the periodogram
        analysis.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.

    returns
    -------
    Nothing returned. The Tessilator data for each camera/CCD configuration
    is saved to file.
    ''' 
    start = datetime.now()
    if len(scc) == 3:
        final_table = create_table_template()
        one_cc(t_targets, scc, make_plots, final_table, keep_data=keep_data, fix_noise=fix_noise)
        final_table.write(f"{period_file}_{scc[1]}_{scc[2]}.ecsv", overwrite=True)
        finish = datetime.now()
        hrs_mins_secs = print_time_taken(start, finish)
        print(f"Finished {len(final_table)} targets for Sector {scc[0]}, "
              f"Camera {scc[1]}, CCD {scc[2]} in {hrs_mins_secs}")
    else:
        cam_ccd = np.array([[i,j] for i in range(1,5) for j in range(1,5)])
        for cc in cam_ccd:
            final_table = create_table_template()
            start_ccd = datetime.now()
            one_cc(t_targets, [scc[0], cc[0], cc[1]], make_plots, final_table, keep_data=keep_data, fix_noise=fix_noise)
            final_table.write(f"{period_file}_{cc[0]}_{cc[1]}.ecsv", overwrite=True)
            finish_ccd = datetime.now()
            hrs_mins_secs = print_time_taken(start_ccd, finish_ccd)
            print(f"Finished {len(final_table)} targets for Sector {scc[0]}, "
                  f"Camera {cc[0]}, CCD {cc[1]} in {hrs_mins_secs}")
        finish = datetime.now()
        hrs_mins_secs = print_time_taken(start, finish)
        print(f"Finished the whole of Sector {scc[0]} in {hrs_mins_secs}")
