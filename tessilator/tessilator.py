"""

Alexander Binks, Moritz Guenther, 2024

License: MIT 2024

The TESSILATOR

"""
###############################################################################
####################################IMPORTS####################################
###############################################################################
#Internal

from datetime import datetime
import sys
import os
import inspect
from glob import glob

# Third party imports
import numpy as np
import pyinputplus as pyip
from astropy.nddata.utils import Cutout2D
from astropy.table import Table, MaskedColumn
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

# Local application imports
from .aperture import aper_run
from .lc_analysis import make_lc
from .periodogram import run_ls
from .detrend_cbv import get_cbv_scc
from .contaminants import contamination, is_period_cont
from .maketable import get_gaia_data
from .makeplots import make_plot
from .file_io import logger_tessilator, make_dir
from .tess_stars2px import tess_stars2px_function_entry

###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__)


STARTUP_STRING = r"""
**********************************************************************
****|******_*********_*********_*********_*********_*********_********
****|*****/*\*******/*\*******/*\*******/*\*******/*\*******/*\*******
****|****/***\*****/***\*****/***\*****/***\*****/***\*****/***\******
****|***/*****\***/*****\***/*****\***/*****\***/*****\***/*****\*****
****|**/*******\_/*******\_/*******\_/*******\_/*******\_/*******\****
****|_____________________________________________________________****
**********************************************************************
**********************WELCOME TO THE TESSILATOR***********************
********The one-stop shop for measuring TESS rotation periods*********
**********************************************************************
**********************************************************************
If this package is useful for research leading to publication we
would appreciate the following acknowledgement:
'The data from the Transiting Exoplanet Survey Satellite (TESS) was
acquired using the tessilator software package (Binks et al. 2024).'
"""

print(STARTUP_STRING)


import requests
from bs4 import BeautifulSoup
import re

tess_web = requests.get('https://tess.mit.edu/observations/')
soup = BeautifulSoup(tess_web.text, 'html.parser')
tess_para = soup(text=re.compile("TESS is in Orbit"))
sec_max = int(tess_para[0].split(',')[1].split(' ')[2].split('.')[0])

_Template_table_format = [
    # column name, description, data type, format, fill_value
    ("original_id", r"Target identifier", str, None, "N/A"),
    ("source_id", r"Gaia DR3 source identifier", str, None, "N/A"),
    ("ra", r"Right ascension (epoch J2000)", float, ".12f", np.nan),
    ("dec", r"Declination (epoch J2000)", float, ".12f", np.nan),
    ("parallax", r"Gaia DR3 parallax", float, ".6f", np.nan),
    ("Gmag",r"Gaia DR3 $G$-band magnitude",float,".6f",np.nan,),
    ("BPmag", r"Gaia DR3 $G_{\rm BP}$-band magnitude", float, ".6f", np.nan),
    ("RPmag", r"Gaia DR3 $G_{\rm RP}$-band magnitude", float, ".6f", np.nan),
    ("Tmag_MED", r"Median TESS $T$-band magnitude", float, ".6f", 0),
    ("Tmag_MAD", r"MAD TESS $T$-band magnitude", float, ".6f", 0),
    ("Sector", r"TESS sector number", int, None, 0),
    ("Camera", r"TESS camera number", int, None, 0),
    ("CCD", r"TESS CCD number", int, None, 0),
    ("log_tot_bg", r"$\Sigma\eta$", float, ".6f", -999),
    ("log_max_bg", r"$\eta_{\rm max}$", float, ".6f", -999),
    ("num_tot_bg", r"Number of contaminating sources", int, None, 0),
    ("ap_rad", r"Aperture radius (pixels)", float, ".3f", -np.inf),
    ("false_flag", r"Test if a contaminant is the $P_{\rm rot}$ source", int, None, 4),
    ("reliable_flag", r"Test if the $P_{\rm rot}$ source is reliable", int, None, 4),
    ("CBV_flag", r"The CBV-correction category", int, None, 9),
    ("smooth_flag", r"Flag for detrending step 1", int, None, 9),
    ("norm_flag", r"Flag for detrending step 2$", int, None, 9),
    ("jump_flag", r"Test for jumps in the lightcurve", int, None, 9),
    ("AIC_line", r"AIC score: linear fit to the lightcurve", float, ".6f", np.nan),
    ("AIC_sine", r"AIC score: sine fit to the lightcurve", float, ".6f", np.nan),
    ("Ndata", r"Number of datapoints in the periodogram analysis", int, None, 0),
    ("FAP_001", r"1\% False Alarm Probability power", float, ".6f", np.nan),
    ("period_1", r"Primary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_1_fit", r"Primary $P_{\rm rot}$ (Gaussian fit centroid)", float, ".6f",np.nan),
    ("period_1_err", r"Primary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_1", r"Power output of the primary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_2", r"Secondary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_2_fit",r"Secondary $P_{\rm rot}$ (Gaussian fit centroid)",float,".6f",np.nan),
    ("period_2_err", r"Secondary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_2", r"Power output of the secondary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_3", r"Tertiary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_3_fit",r"Tertiary $P_{\rm rot}$ (Gaussian fit centroid)",float,".6f",np.nan),
    ("period_3_err", r"Tertiary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_3", r"Power output of the tertiary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_4", r"Quaternary $P_{\rm rot}$ (peak)", float, ".6f", np.nan),
    ("period_4_fit", r"Quaternary $P_{\rm rot}$ (Gaussian fit centroid)", float, ".6f", np.nan),
    ("period_4_err", r"Quaternary $P_{\rm rot}$ uncertainty", float, ".6f", np.nan),
    ("power_4", r"Power output of the quaternary $P_{\rm rot}$", float, ".6f", np.nan),
    ("period_shuffle", r"$P_{\rm shuff}$", float, ".6f", np.nan),
    ("period_shuffle_err", r"Uncertainty in $P_{\rm shuff}$", float, ".6f", np.nan),
    ("shuffle_flag", r"Indicates if $P_{\rm shuff}$ was adopted", int, None, 9),
    ("amp", r"Amplitude of the PFL", float, ".6f", np.nan),
    ("scatter", r"Scatter of the PFL", float, ".6f", np.nan),
    ("chisq_phase",r"$\chi^{2}$ value of the sinusoidal fit to the PFL",float,".6f",np.nan),
    ("fdev", r"Number of extreme outliers in the PFL", float, ".6f", np.nan),
]


def create_table_template():
    '''Create a template astropy table to store tessilator results.

    returns
    -------
    res_table : `astropy.table.Table`
        A template table to store tessilator results.
    '''
    cols = []
    for name, description, dtype, format, fill_value in _Template_table_format:
        cols.append(
            MaskedColumn(
                name=name,
                description=description,
                dtype=dtype,
                format=format,
                fill_value=fill_value,
            )
        )
    return Table(cols, masked=True)


def setup_input_parameters():
    '''Retrieve the input parameters to run the tessilator program.

    The input parameters are:

    1) "flux_con": the toggle for applying the contamination calculation
       (yes=1, no=0).

    2) either "lc_con", if using the cutout functions or "sector_num" if
       sectors are needed.
              
       * "lc_con" determines if lightcurve/periodogram analyses should be
         carried out for neighbouring contaminants (yes=1, no=0).

       * "sector_num" prompts the user to enter the sector number needed.
         If command line arguments are not used, the program will ask if a
         specific Camera and CCD is needed (1=yes, 0=no). If not required, the
         whole sector is analysed. If this is a command line argument, if the
         user enters just the sector number (maximum 2 digits) the whole sector
         will be analysed, and if the Camera and CCD number are given right
         after the sector number with no spaces, then a specific Camera and CCD
         configuration will be used. E.G: if "sector_num = 8", the entire
         sector 8 is analysed, whereas if "sector_num = 814" then the program
         will analyse only Camera 1 and CCD 4 in sector 8.

    3) "make_plots" gives the user the option to make plots (yes=1, no=0)

    4) "file_ref" is a string expression used to reference the files produced.

    5) "t_filename" is the name of the input file (or target) required for
       analysis.

    If a program is called from the command line without all five input
    parameters, a set of prompts are initiated to receive input. If just one
    target is needed, then the user can simply supply either the target name,
    as long as it is preceeding by a hash (#) symbol. Otherwise, if the full
    set of command line parameters are supplied, the function will use these
    as the inputs, however, if they have the wrong format the program will
    return a warning message and exit.

    parameters
    ----------
    Either arguments supplied on the command line, or the function will prompt
    the user to provide input.
    
    returns
    -------
    flux_con : `bool`
        Run lightcurve analysis for contaminant sources.
    scc : `list`, size=3, only if sector data is used
        List containing the sector number, camera and CCD.
    lc_con : `bool`, only if cutout data is used
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    file_ref : `str`
        A common string to give all output files the same naming convention.
    t_filename : `str`
        The name of the input table containing the targets (or a single
        target).
     '''
    # first, set parameters in the case where inputs are not defined on the
    # command line
    if len(sys.argv) != 6:
        flux_con = pyip.inputInt("Do you want to search for contaminants? "
                   "1=yes, 0=no : ", min=0, max=1)
        if 'cutout' in sys.argv[0]:
            lc_con = pyip.inputInt("Do you want to calculate period data for "
                     "the contaminants? 1=yes, 0=no : ", min=0, max=1)
        elif 'sector' in sys.argv[0]:
            sector_num = pyip.inputInt("Which sector of data do you require? "
                         f"(1-{sec_max}) : ", min=1, max=sec_max)
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
                         "table or object.\nIf this is a single target please "
                         "enter a hash (#) symbol before the identifier : ")
            if t_filename.startswith('#'):
                t_name = t_filename[1:]
                t_name_joined = \
                t_name.replace(' ','_').replace(',', '_')+'.dat'
                if os.path.exists(t_name_joined):
                    os.remove(t_name_joined)
                with open(t_name_joined, 'a') as single_target:
                    single_target.write(t_name)
                t_filename = t_name_joined
                break
            if not os.path.exists(t_filename):
                logger.error(f'The file "{t_filename}" does not exist.')
            else:
                break
    # second, set parameters in the case where command line inputs are given
    else:
        flux_con = int(sys.argv[1])
        if 'cutout' in sys.argv[0]:
            lc_con = int(sys.argv[2])
        elif 'sector' in sys.argv[0]:
            scc_in = str(sys.argv[2])
            if len(scc_in) > 4:
                logger.critical("Incorrect format for sector/camera/ccd " "values")
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
        sec_vals = np.arange(1,sec_max+1)
        cam_ccd_vals = np.arange(1,5)
        if flux_con not in true_vals:
            logger.critical(f"flux_con value {flux_con} not a valid input: "
                            f"choose either 0 or 1.\nExiting program.")
            sys.exit()
        if 'cutout' in sys.argv[0]:
            if lc_con not in true_vals:
                logger.critical(f"lc_con value {lc_con} not a valid input: "
                            f"choose either 0 or 1.\nExiting program.")
                sys.exit()
        elif 'sector' in sys.argv[0]:
            if scc[0] not in sec_vals:
                logger.critical(f"sector_num value {scc[0]} not a valid input: "
                                f"choose integer between 1 and {sec_max}.\n"
                                f"Exiting program.")
                sys.exit()
            if len(scc) == 3:
                if scc[1] not in cam_ccd_vals:
                    logger.critical(f"Camera value {scc[1]} out of range: "
                                    f"choose integer between 1 and 4.\n"
                                    f"Exiting program.")
                    sys.exit()
                if scc[2] not in cam_ccd_vals:
                    logger.critical(f"CCD value {scc[2]} out of range: "
                                    f"choose integer between 1 and 4.\n"
                                    f"Exiting program.")
                    sys.exit()                        
        if make_plots not in true_vals:
            logger.critical(
                "make_plots not a valid input: "
                "choose either 0 or 1.\nExiting program."
            )
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
        if not os.path.exists(t_filename):
            logger.critical(f'The file "{t_filename}" does not exist.')
            sys.exit()
    if 'cutout' in sys.argv[0]:
        return flux_con, lc_con, make_plots, file_ref, t_filename
    elif 'sector' in sys.argv[0]:
        return flux_con, scc, make_plots, file_ref, t_filename


def setup_filenames(file_ref, scc=None):
    '''Set up the file names to store data.

    parameters
    ----------
    file_ref : `str`
        A common string to give all output files the same naming convention.
    scc : `list` or `None`, size=3, optional, default = `None` 
        A list containing the Sector, Camera and CCD.

    returns
    -------
    period_file : `str`
        Name of file for recording parameters measured by the periodogram
        analysis.
    '''    
    if scc:
        name_str = 'sector'+f"{scc[0]:02d}"
    else:
        name_str = 'tesscut'
    period_file = '_'.join(['periods', file_ref, name_str])
    return period_file




def test_table_large_sectors(t_filename):
    '''Check if the input file needs modifying at all.

    If running the tessilator for a whole sector, read the input file and if
    the format is ready for analysis, make a couple of adjustments, then simply
    pass the file.
    
    For a straight pass, the columns must be ordered in two ways. Either:

    * exactly set out with the following columns:
    
      1) source_id: name of the Gaia DR3 source identifier
      
      2) ra: right ascension
      
      3) dec: declination
      
      4) parallax: parallax
      
      5) Gmag: Gaia DR3 apparent G-band magnitude
      
      6) BPmag: Gaia DR3 apparent BP-band magnitude
      
      7) RPmag: Gaia DR3 apparent RP-band magnitude
      
      8) Sector: The TESS sector
      
      9) Camera: The TESS camera
      
      10) CCD: The TESS CCD number
      
      11) Xpos: The TESS X-pixel
      
      12) Ypos: The TESS Y-pixel
        
    * The same, but with a preceding column entitled "name", which refers to a
      target identifier name. This can be any string.
    
    In any other case, None is returned and other functions are used to get the
    table into the correct format.

    parameters
    ----------
    t_filename : `str`
        The file name of the table which will be checked for formatting.

    returns
    -------
    t : `astropy.table.Table` or `None`
        either a ready-prepared table or nothing.
    '''

    t = Table.read(t_filename, format='csv')
    # The list of column names which must be spelled exactly.
    cnc = ['source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag',
           'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos']
    if cnc == t.colnames:
        t["name"] = t["source_id"]
        cnc.insert(0, 'name')
        t = t[cnc]
        # Ensure the dtype for "name" and "source_id" are strings.
        t["name"] = t["name"]
        t["source_id"] = t["source_id"]
    elif cnc == t.colnames[1:]:
        if t.colnames[0] == "name":
            # Ensure the dtype for "name" and "source_id" are strings.
            t["name"] = t["name"]
            t["source_id"] = t["source_id"]
    else:
        t = None
        # return nothing if neither of the above two conditions are met.
    return t


def read_data(t_filename, name_is_source_id=False, type_coord='icrs',
              gaia_sys=True):
    '''Read input data and convert to an astropy table ready for analysis.
    
    The input data must be in the form of a comma-separated variable and may
    take 3 forms:
    
    (a) a 1-column table of source identifiers.
    
    (b) a 2-column table of decimal sky coordinates (celestial or galactic).
    
    (c) a pre-prepared table of 7 columns consisting of source_id, ra, dec,
        parallax, Gmag, BPmag, RPmag (without column headers).
    
    Note that a single target can be quickly analysed directly from the
    command line by using option (a) with a #-sign preceding the target name,
    and then encompassed with double-quotation marks around the source
    identifier.
    
    E.G. >>> python run_tess_cutouts 0 0 0 files "#AB Doradus"
    
    parameters
    ----------
    t_filename : `astropy.table.Table`
        Name of the file containing the input data.
    name_is_source_id : `bool`, optional, default=False
        When running option (c), the "name" column will automatically be set as
        the Gaia DR3 identifiers if this parameter is True. This avoids long
        sql queries for very large input tables.
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input data. Choose from 'icrs', 'galactic'
        or 'barycentricmeanecliptic', where the latter is the conventional
        coordinate system used by TESS.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    returns
    -------
    t_targets : `astropy.table.Table`
        a formatted astropy table ready for further analysis
    '''
    if isinstance(t_filename, str):
        t_input = ascii.read(t_filename, delimiter=',', format='no_header')
    elif isinstance(t_filename, Table):
        t_input = t_filename
    t_targets = get_gaia_data(t_input, name_is_source_id=name_is_source_id,
                              type_coord=type_coord, gaia_sys=gaia_sys)
    print(t_targets)
    return t_targets


def collect_contamination_data(t_targets, ref_name, targ_name,
                               gaia_sys=True, ap_rad=1., n_cont=10, cont_rad=10.,
                               mag_lim=3., tot_attempts=3):
    '''Collect data on contamination sources around selected targets.

    This function takes a target table and, if requested, prints out details of
    the total flux contribution from neighbouring contaminating sources for
    each target.
    It also returns a table (in descending order) of neighbouring contaminants
    that contribute the most flux in the target aperture, if requested.
    
    Parameters
    ----------
    t_targets : `astropy.table.Table`
        A target table with columns:        

        * name: name of the target (`str`)

        * source_id: Gaia DR3 source identifier (`str`)

        * ra: right ascension

        * dec: declination

        * parallax: parallax

        * Gmag: Gaia DR3 apparent G-band magnitude

        * BPmag: Gaia DR3 apparent BP-band magnitude

        * RPmag: Gaia DR3 apparent RP-band magnitude

    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    ap_rad : `float`, optional, default=1.
        The aperture radius from the aperture photometry (in pixels).
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    Returns
    -------
    t_targets : `astropy.table.Table`
        Input target table with 3 columns added containing details of the
        contamination: "log_tot_bg", "log_max_bg", "num_tot_bg".
    t_cont : `astropy.table.Table`
        Table containing details of the contamination flux from nearby sources,
        in descending order of "log_tot_bg".
    cont_dir : `str`
        The directory that contains the contamination files.
    '''

    if gaia_sys:
        t_targets, t_cont = contamination(t_targets, ap_rad=ap_rad, n_cont=n_cont,
                                          cont_rad=cont_rad, mag_lim=mag_lim, tot_attempts=tot_attempts)
        t_contam = t_targets[['source_id', 'log_tot_bg', 'log_max_bg',\
                              'num_tot_bg']]
        cont_dir = make_dir("contaminants", ref_name)
        path_exist = os.path.exists(cont_dir)
        if not path_exist:
            os.mkdir(cont_dir)
        t_contam.write(f'{cont_dir}/{targ_name}.csv', overwrite=True)
        if t_cont is not None:
            t_cont.write(f'{cont_dir}/{targ_name}_individiual.csv',
                         overwrite=True)
    else:
        t_targets["log_tot_bg"] = -999.
        t_targets["log_max_bg"] = -999.
        t_targets["num_tot_bg"] = -999.
        if os.path.exists(f'{cont_dir}/{targ_name}.csv'):
            t_contam = Table.read(f'{cont_dir}/{targ_name}.csv')
            for i in range(len(t_contam)):
                g = (t_contam["source_id"] == \
                     t_targets["source_id"][i])
                if len(g) >= 1:
                    t_targets["log_tot_bg"][i] = t_contam["log_tot_bg"][g][0]
                    t_targets["log_max_bg"][i] = t_contam["log_max_bg"][g][0]
                    t_targets["num_tot_bg"][i] = t_contam["num_tot_bg"][g][0]
    return t_targets, t_cont, cont_dir


def make_target_row(t_targets, r, scc):
    """Construct table row with the input target data.

    If tessilator fails to run on a target, that's all you get.

    Parameters
    ----------
    t_targets : `astropy.table.Row`
        One row of input data for the tessilator, with the following columns:

        * name: name of the target (`str`)

        * source_id: Gaia DR3 source identifier (`str`)

        * ra: right ascension

        * dec: declination

        * parallax: parallax

        * Gmag: Gaia DR3 apparent G-band magnitude

        * BPmag: Gaia DR3 apparent BP-band magnitude

        * RPmag: Gaia DR3 apparent RP-band magnitude

        * log_tot_bg: log-10 value of the flux ratio between contaminants
          and target (optional)

        * log_max_bg: log-10 value of the flux ratio between the largest
          contaminant and target (optional)

        * num_tot_bg: number of contaminant sources (optional)

    r : `float`
        The pixel size of the aperture radius
    scc : tuple, size=3
        A list containing the Sector, Camera and CCD.

    returns
    -------
    dr : dict
        A dictionary for the target star containing input data
    """
    copycols = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        "Gmag",
        "BPmag",
        "RPmag",
        "log_tot_bg",
        "log_max_bg",
        "num_tot_bg",
    ]
    dr = {col: t_targets[col] for col in copycols}
    dr["original_id"] = t_targets["name"]
    dr["Sector"] = scc[0]
    dr["Camera"] = scc[1]
    dr["CCD"] = scc[2]
    dr["ap_rad"] = r
    # All other lines we leave on default, so they will be maksed,
    # can can be filled if needed.
    return dr


def apply_noise_corr(targ_lc, sim_lc):
    '''Apply the correction to the lightcurve based on the noise simulation.
    
    parameters
    ----------
    targ_lc : `dict`
        A dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    sim_lc : `dict`
        A dictionary containing the simulated data.
        
    returns
    -------
    targ_lc : `dict`
        The same input lightcurve, corrected by the noise simulation.
    '''
    targ_lc['nflux_err'][np.where(targ_lc['nflux_err'] < 0)] = .01
    cln_cond = np.logical_and.reduce([
                   targ_lc["pass_clean_scatter"],
                   targ_lc["pass_clean_outlier"],
                   targ_lc["pass_full_outlier"]
                   ])
    targ_lc["nflux_noise_corr"] = 0
    targ_lc["sim_flux"] = 0

    nflux_noise_corr = np.array([])
    sim_flux = np.array([])
    for t, time in enumerate(targ_lc["time"]):
        sim_bit = np.interp(time, sim_lc["time"], sim_lc["nflux_dtr"])
        flux_bit = targ_lc["nflux_dtr"][t]/sim_bit
        sim_flux = np.append(sim_flux, sim_bit)
        nflux_noise_corr = np.append(nflux_noise_corr, flux_bit)
    targ_lc["nflux_noise_corr"][cln_cond] = nflux_noise_corr[cln_cond]
    targ_lc["sim_flux"][cln_cond] = sim_flux[cln_cond]
    return targ_lc


def fix_noise_lc_local(targ_lc, med_lc, scc, targ_name, ref_name,
                       make_plots=False):
    '''Apply a flux-corrected key to the target lightcurve dictionary, and make
    a plot of the corrections if required.
    
    This function uses the lightcurve data from neighbouring sources.
    
    parameters
    ----------
    targ_lc : `dict`
        a dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    med_lc : `dict`
        the "median" lightcurve data from the neigbourhing sources.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    targ_name : `str`
        The name of the target
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    make_plots : `bool`, optional, default=False
        Choose to make plots of the noise corrections.

    returns
    -------
    targ_lc : `dict`
        The updated target lightcurve dictionary, with an extra key containing
        the noise-corrected flux.
    '''

    targ_lc = apply_noise_corr(targ_lc, med_lc)
    if make_plots:
        plot_name = f"{targ_name}_{scc[0]:04d}_{scc[1]}_"\
                    f"{scc[2]}_corr_local_lc.png"

        make_lc_corr_plot(plot_name, ref_name, targ_lc["time"],
                          targ_lc["nflux_dtr"], targ_lc["sim_flux"])
    return targ_lc



# NOTE - I'm cancelling this procedure until we get new lightcurves!
# 23.01.2024
def fix_noise_lc_sim(targ_lc, targ_name, t_targets, scc, ref_name,
                     mag_extr_lim=3., make_plots=False):
    '''Divide the target lightcurve by the noise lightcurve
    
    For a given sector, camera and CCD configuration, this function will search
    for the simulated noisy lightcurve and divide the target lightcurve by the
    noise lightcurve, which should mitigate some of the systematic features.
    
    The noisy lightcurves are described in the tessimulation.py module.
    
    parameters
    ----------
    targ_lc : `dict`
        A dictionary containing the lightcurve data for a target
        star (see tessilator.make_lc for the required inputs).
    targ_name : `str`
        The name of the target
    t_targets : `astropy.table.row.Row`
        Details of the target star.   
    scc : `list`, size=3
        A list containing the sector number, camera and CCD.    
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    mag_extr_lim : `float`, optional, default=3.
        The tolerated difference in magnitude between the target
        and the range of calculated simulated lightcurves if the
        magnitude is out of range.
    make_plots : `bool`, optional, default=False
        Choose to make plots of the noise corrections.

    returns
    -------
    targ_lc : `dict`
        The updated target lightcurve dictionary, with an extra key containing
        the noise-corrected flux.
    '''
    
    mag_files = sorted(glob(f"./tesssim/lc/{scc[0]:02d}_{scc[1]}_"\
                            f"{scc[2]}/mag*"))
    if not mag_files:
        logger.warning(f"No simulated lightcurve for {targ_name}, Sector "
                       f"{scc[0]:02d}, Camera {scc[1]}, CCD {scc[2]}")
        return targ_lc

    mag_target = t_targets['Gmag'][0]
    mag0 = np.array([float(mag_file.split("_")[-2]) for mag_file in mag_files])
    mag1 = np.array([float(mag_file.split("_")[-1]) for mag_file in mag_files])

    g = np.where((mag0 <= mag_target) & (mag1 > mag_target))[0]
    g = []
    if g.size > 0:
        sim_tab = f'{mag_files[g[0]]}/flux_fin.csv'
    elif (mag_target < mag0[0]) & \
         (np.abs(mag0[0] - mag_target) < mag_extr_lim):
        logger.info(f"Target {targ_name} is brighter than the magnitude range of "
                    f"simulated files, but within the extrapolation threshold")
        sim_tab = f'{mag_files[0]}/flux_fin.csv'
    elif (mag_target > mag1[-1]) & \
         (np.abs(mag1[-1] - mag_target) < mag_extr_lim):
        logger.info(f"Target {targ_name} is fainter than the magnitude range "
                    f"of simulated files, but within the extrapolation "
                    f"threshold")
        sim_tab = f'{mag_files[-1]}/flux_fin.csv'
    else:
        logger.warning(f"Target {targ_name} is out of the magnitude range of "
                       f"simulated files, and out of the extrapolation "
                       f"threshold")
        return targ_lc

    sim_lc = Table.read(sim_tab)
    targ_lc = apply_noise_corr(targ_lc, sim_lc)

    if make_plots:
        plot_name = f"{targ_name}_{scc[0]:04d}_{scc[1]}_{scc[2]}_"\
                    f"corr_sim_lc.png"
        make_lc_corr_plot(plot_name, ref_name, targ_lc["time"],
                          targ_lc["nflux_dtr"], targ_lc["sim_flux"])
    return targ_lc





def make_lc_corr_plot(plot_name, ref_name, targ_time, targ_flux, sim_flux,
                      im_dir='plots'):
    '''Make a plot of the noise corrections to the lightcurve.
    
    parameters
    ----------
    plot_name : `str`
        The file name which will be used to save the plot.
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    targ_time : `Iterable`
        The time coordinate of the data.
    targ_flux : `Iterable`
        The flux coordinate of the target data.
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
    im_dir_tot = f'./{im_dir}/{ref_name}'
    path_exist = os.path.exists(im_dir_tot)
    if not path_exist:
        os.mkdir(im_dir_tot)
    plt.savefig(f'{im_dir_tot}/{plot_name}', bbox_inches='tight')
    plt.close('all')



def get_name_target(t_target):
    '''Quick function to capture a string containing the name of the target.
    
    parameters
    ----------
    t_target : `astropy.table.row.Row`
        The target data, which contains the target name.
        
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



def get_median_lc(tables, files_loc, scc, n_bin=10):
    '''A function that makes the final noisy lightcurve.
    
    For a given number of lightcurves, this function calculates the median flux
    at each time step if there are more than 2 measurements (or the mean if 2
    or less).
    
    parameters
    ----------
    tables : `list`
        A list of lightcurves in a given directory.
    files_loc : `str`
        The name of the directory containing the lightcurves.
    scc : `list`, size=3
        A list containing the sector number, camera and CCD.    
    n_bin : `int`, optional, default=10
        The maximum number of lightcurves to be used in the analysis.
        
    returns
    -------
    t_fin : `astropy.table.Table`
        A table containing the data for the final noisy lightcurve.
    '''
    f_name = files_loc.split("/")[3].split("_")[1]
    directory = ('/').join(files_loc.split("/")[:-1])+'/'
    num, time, flux, eflux = [], [], [], []
    if len(tables) > n_bin:
        chosen_indices = np.random.choice(len(tables), n_bin)
        tables_chosen = [tables[n] for n in chosen_indices]
    else:
        tables_chosen = tables
    for t, tab in enumerate(tables_chosen):
        tab['nflux_err'][np.where(tab['nflux_err'] < 0)] = .01
        cln_cond = np.logical_and.reduce([
                       tab["pass_clean_scatter"],
                       tab["pass_clean_outlier"],
                       tab["pass_full_outlier"]
                       ])

        tab = tab[cln_cond]
        for t_line in tab:
            num.append(t+1)
            time.append(t_line['time'])
            flux.append(t_line['nflux_dtr'])
            eflux.append(t_line['nflux_err'])
    t_uniq = np.unique(np.array(time))
    t_fin = Table(names=('time', 'nflux_dtr', 'nflux_err', 'n_lc'),
                  dtype=(float, float, float, int))
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
    t_fin.write(f'{directory}flux_med_{f_name}_{scc[0]:04d}_'
                f'{scc[1]}_{scc[2]}.csv', overwrite=True)
    return t_fin




def assess_lc(ls_results):
    '''Decide whether to use the periodogram results from the original, or from
    the CBV-corrected lightcurve.
    
    The function provides 5 different tests to find which analysis provides
    better results. For each test, the winning lightcurve scores a point. The
    one with the most points at the end of the tests is chosen as the
    periodogram results output.

    parameters
    ----------
    ls_results : `list`
        The list of dictionaries containing the periodogram scores for both
        lightcurves.
        
    returns
    -------
        lc_choice : `int`
            The chosen periodogram, where 0=original and 1=CBV-corrected.
    '''
    ori_sc, cbv_sc = 0, 0
    ori_ls, cbv_ls = ls_results[0], ls_results[1]
    lc_choice = 0

#1) Check the best fit sine vs best fit line scores...
    if (ori_ls["AIC_sine"]-ori_ls["AIC_line"]) < \
       (cbv_ls["AIC_sine"]-cbv_ls["AIC_line"]):
        ori_sc += 1
    else:
        cbv_sc += 1
#2) Check how jumpy the lightcurves are...
    if not ori_ls["jump_flag"]:
        ori_sc += 1
    if not cbv_ls["jump_flag"]:
        cbv_sc += 1
#3) Check the max_power/FAP_001
    if (ori_ls["power_1"]/ori_ls["FAPs"][2]) > \
       (cbv_ls["power_1"]/cbv_ls["FAPs"][2]):
        ori_sc += 1
    else:
        cbv_sc += 1
#4) Check the height of the amplitude
    if ori_ls["pops_vals"][1]/ori_ls["phase_scatter"] > \
       cbv_ls["pops_vals"][1]/cbv_ls["phase_scatter"]:
        ori_sc += 1
    else:
        cbv_sc += 1
#5) Check number of datapoints
    if ori_ls["Ndata"] > cbv_ls["Ndata"]:
        ori_sc += 1
    else:
        cbv_sc += 1


#6) Check number of outliers in the phase-folded curve
#    if ori_ls["frac_phase_outliers"] < cbv_ls["frac_phase_outliers"]:
#        ori_sc += 1
#    else:
#        cbv_sc += 1
#    print('test fdev: ', ori_sc, cbv_sc)
#7) Check the max_power/2nd_max_power...
#    if (ori_ls["power_1"]/ori_ls["power_2"]) > \
#        (cbv_ls["power_1"]/cbv_ls["power_2"]):
#        ori_sc += 1
#    else:
#        cbv_sc += 1

        
    test_fdev = cbv_ls["frac_phase_outliers"] < \
                ori_ls["frac_phase_outliers"]
    if (ori_sc < cbv_sc) & (test_fdev):
        lc_choice = 1
    return lc_choice



def full_run_lc(file_in, t_target, make_plots, scc, res_table, gaia_sys=True,
                xy_pos=(10.,10.), ap_rad=1., sky_ann=(6.,8.), fix_rad=False,
                n_cont=10, cont_rad=10., mag_lim=3., tot_attempts=3,
                ref_name='targets', cutout_size=20, save_phot=False,
                cbv_flag=False, store_lc=False, lc_dir='lc', pg_dir='pg',
                plot_ext='plots', keep_data=False, flux_con=False,
                lc_con=False, fix_noise=False, shuf_per=False,
                make_shuf_plot=False, shuf_dir='plot_shuf'):
    '''Aperture photometry, lightcurve cleaning and periodogram analysis.

    This function calls a set of functions in the lc_analysis.py module to
    perform aperture photometry, clean the lightcurves from spurious data and
    runs the Lomb-Scargle periodogram to measure rotation periods.

    parameters
    ----------
    file_in : `str`
        Name of the input TESS fits file.
    t_target : `astropy.table.Table`
        Details of the target star.
    make_plots: `bool`
        Choose to make plots. 
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    res_table : `astropy.table.Table`
        The table to store the final tessilator results.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.
    xy_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run the
        calc_rad algorithm.
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    ref_name : `str`, optional, default='targets'
        The reference name for each subdirectory which will connect all output
        files.
    cutout_size : `int`, optional, default=20
        The pixel length of the downloaded cutout.
    save_phot : `bool`, optional, default=False
        Decide whether to save the full results from the aperture photometry.
    cbv_flag : `bool`, optional, default=False
        Decide whether to run the lightcurve analysis with a CBV-correction
        applied.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file.
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True.
    pg_dir : `str`, optional, default='pg'
        The directory used to store the periodogram data.
    plot_ext : `str`, optional, default='plots'
        The directory used to store the plots if make_plots==True.
    keep_data : `bool`
        Choose to save the input data to file.
    flux_con : `bool`, optional, default=False
        Decides if the flux contribution from contaminants is to be calculated.
    lc_con : `bool`, optional, default=0
        Decides if a lightcurve analysis is to be performed for the n strongest
        contaminants.
    fix_noise : `bool`, optional, default=False
        Choose to apply corrections accounting for systematic noise.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period
        analysis. 

    returns
    -------
    * A data entry for the final period file.
    * A plot of the lightcurve (if requested).
    '''
    nc = 'nc'
    try:
        tpf, rad_calc = aper_run(file_in, t_target, xy_pos=xy_pos, ap_rad=ap_rad,
                                 sky_ann=sky_ann, fix_rad=fix_rad)
    except Exception as e:
        logger.error(f"aperture photometry: of {file_in} failed to run")
    if len(tpf) < 10:
        logger.error(f"aperture photometry: failed to produce enough data "
                     f"points for {t_target['source_id']}")
        for t in t_target:
            res_table.add_row(make_target_row(t, r=rad_calc, scc=scc))
        return None
    if cbv_flag:
        corrected_flux, weights = get_cbv_scc(scc, tpf)
        tpf["cbv_oflux"] = corrected_flux[1][:]
    else:
        tpf["cbv_oflux"] = tpf["reg_oflux"]
    keyorder = ['run_no','gaia_dr3_id','aperture_rad','time','xcenter','ycenter',
                'flux','flux_err','bkg','total_bkg','mag','mag_err',
                'reg_oflux','cbv_oflux']
    tab_format = [
        "%i",
        "%s",
        ".2f",
        ".6f",
        ".1f",
        ".1f",
        ".6f",
        ".6f",
        ".6f",
        ".6f",
        ".6f",
        ".4e",
        ".6f",
        r".6f",
    ]
    tpf = tpf[keyorder]
    for n, f in zip(keyorder, tab_format):
        tpf[n].info.format = f

    phot_targets = tpf.group_by('gaia_dr3_id')
    for key, group in zip(phot_targets.groups.keys, phot_targets.groups):
        g_c = group[group["flux"] > 0.0]
        if isinstance(t_target, Table):
            t_targets = t_target[t_target["source_id"] == key[0]]
        else:
            t_targets = Table(t_target)
            
        name_target = get_name_target(t_targets["name"][0])
        name_full = f'{name_target}_{scc[0]:04d}_{scc[1]}_{scc[2]}'
        if flux_con:
            t_targets, t_cont, cont_dir = \
            collect_contamination_data(t_targets, ref_name, name_full,
                                       gaia_sys=gaia_sys, ap_rad=rad_calc, n_cont=n_cont, cont_rad=cont_rad, mag_lim=mag_lim, tot_attempts=tot_attempts
                                       )
        else:
            t_targets["log_tot_bg"] = -999.
            t_targets["log_max_bg"] = -999.
            t_targets["num_tot_bg"] = -999.
        if save_phot:
            tpf.write(f'{lc_dir}/ap_{name_full}.csv', overwrite=True)
        if len(g_c) >= 50:
            lcs, norm_flags, smooth_flags = make_lc(g_c,
                                                    name_lc='lc_'+name_full,
                                                    store_lc=store_lc,
                                                    lc_dir=lc_dir,
                                                    cbv_flag=cbv_flag)
        else:
            logger.error(f"No photometry was recorded for this group.")
            res_table.add_row(make_target_row(t_targets, r=rad_calc, scc=scc))
            continue
        if len(lcs) == 0:
            logger.error(f"no datapoints to make lightcurve analysis for "
                         f"{t_targets['source_id']}")
            res_table.add_row(make_target_row(t_targets, r=rad_calc, scc=scc))
            continue
        if fix_noise and not lc_con:
            logger.info('fixing the noise!')
            for l in range(len(lcs)):
                lcs[l] = fix_noise_lc_sim(lcs[l], name_target, t_targets, scc,
                                          ref_name, make_plots=make_plots)
            nc = 'corr_sim'
        ls_results = []
        for lc in lcs:
            lc_type = lc.colnames[2][:3]
            ls = run_ls(lc, lc_type=lc_type, ref_name=ref_name,
                        name_pg='pg_'+name_full,
                        check_jump=True, pg_dir=pg_dir, shuf_per=shuf_per,
                        n_shuf_runs=5000, make_shuf_plot=make_shuf_plot,
                        shuf_dir=shuf_dir,
                        name_shuf_plot=f'{name_full}_shuf.png')
            ls_results.append(ls)
        if len(lcs) == 1:
            choose_lc = 0
            best_lc = 0
        else:
            choose_lc = assess_lc(ls_results)
            best_lc = 1 + choose_lc
        d_target = ls_results[choose_lc]
        lc = lcs[choose_lc]
        d_target["CBV_flag"] = best_lc
        d_target["norm_flag"] = int(norm_flags[choose_lc])
        d_target["smooth_flag"] = int(smooth_flags[choose_lc])
        if d_target['period_1'] == -999:
            logger.error(f"the periodogram did not return any results for "
                         f"{t_targets['source_id']}")
            res_table.add_row(make_target_row(t_targets, r=rad_calc, scc=scc))
            continue

        false_flag, reliable_flag = 4, 4        
        if lc_con:
            print('performing periodogram analysis of potential contaminants')
            lc_cont_files = []
            if flux_con != 1:
                logger.warning("Contaminants not identified! "
                               "Please toggle lc_con=1")
                logger.warning("Continuing program using only the target.")
            else:
                logger.info('calculating contaminant lightcurves')
                if t_cont is not None:
                    t_cont = t_cont[t_cont["source_id_target"] ==
                                          t_targets["source_id"]]
                    xy_con = find_xy_cont(file_in, t_cont, cutout_size)
                    false_flag, reliable_flag = '', ''
                    for z in range(len(xy_con)):
                        name_lc, false_lab, reliable_lab, lc_cont, d_lc = \
                        run_test_for_contaminant(xy_con[z], file_in, t_cont[z],
                                                 d_target, scc,
                                                 lc_cont_dir=cont_dir)
                        false_flag += str(false_lab)
                        reliable_flag += str(reliable_lab)
                        if d_lc is not None:
                            if d_lc["AIC_sine"] > d_lc["AIC_line"]+1:
                                path_exist = os.path.exists(cont_dir)
                                if not path_exist:
                                    os.mkdir(cont_dir)
                                lc_cont.write(f'{cont_dir}/{name_lc}',
                                              overwrite=True)
                                lc_cont_files.append(lc_cont)
                        false_flag = int(false_flag)
                        reliable_flag = int(reliable_flag)
                else:
                    false_flag, reliable_flag = 3, 3
                    xy_con = None
                if fix_noise:
                    print('fixing the noise')
                    logger.info('fixing the noise!')
                    if len(lc_cont_files) > 0:
                        nc = 'corr_local'
                        t_median_lc = \
                        get_median_lc(lc_cont_files, f'{cont_dir}/{name_lc}',
                                      scc)
                        clean_norm_lc = \
                        fix_noise_lc_local(lc, t_median_lc, scc, name_target,
                                           ref_name, make_plots=make_plots)
                        d_target = run_ls(clean_norm_lc, check_jump=True)
                        d_target["best_lc"] = best_lc
                    else:
                        nc = 'corr_sim'
                        lc = fix_noise_lc_sim(lc, name_target, t_targets, scc,
                                              ref_name, make_plots=make_plots)
                        d_target = run_ls(lc, check_jump=True)
                        d_target["best_lc"] = best_lc
                else:
                    nc = 'nc'
        else:
            xy_con = None
            
        d_target["false_flag"] = false_flag
        d_target["reliable_flag"] = reliable_flag
        if make_plots:
            plot_dir = make_dir(plot_ext, ref_name)
            im_plot, xy_ctr = make_2d_cutout(file_in, group, 
                                             im_size=(cutout_size+1,
                                                      cutout_size+1))
            make_plot(im_plot, lc, d_target, scc, t_targets, name_target,
                      plot_dir, xy_contam=xy_con, p_min_thresh=0.1,
                      p_max_thresh=50., ap_rad=rad_calc, sky_ann=sky_ann, nc=nc)

        target_row = make_target_row(t_targets, r=rad_calc, scc=scc) | d_target
        common_cols = {k: v for k, v in target_row.items() if k in res_table.colnames}
        res_table.add_row(common_cols)
        temp_dir = make_dir("temp_results", ref_name)
        res_table.write(f'{temp_dir}/{ref_name}_periods.csv', overwrite=True)
        if not keep_data:
            if len(file_in) == 1:
                os.remove(file_in)
    print('completed!')


def print_time_taken(start, finish):
    '''Calculate the time taken for a process.

    This function takes a start and finish point a calculates the time taken in
    hours, minutes and seconds.

    parameters
    ----------
    start : `datetime.datetime`
        The start point of the process.
    
    finish : `datetime.datetime`
        The end point of the process.

    returns
    -------
    time_taken : `str`
        The time taken for the process to complete.
    '''
    time_in_secs = (finish - start).seconds
    mins, secs = divmod(time_in_secs, 60)
    hrs, mins = divmod(mins, 60)
    time_taken = f"{hrs} hours, {mins} minutes, {secs} seconds"
    return time_taken


def find_xy_cont(f_file, t_cont, cutout_size):
    '''Identify the pixel x-y positions for contaminant sources.

    If the user requests a periodogram analysis of neighbouring potential
    contaminants (lc_con=1), this function returns their x-y positions, which
    are used as the centroids for aperture photometry.

    parameters
    ----------
    f_file : `str`
        The name of the fits file.
    t_cont : `astropy.table.Table`
        The table containing Gaia data of the contaminants.
    cutout_size : `int`
        The length size of the TESS cutout image.

    returns
    -------
    cont_positions : `numpy.array`
        A 2-column array of the X-Y positions of the contaminants.
    '''
    xy_ctr = (cutout_size/2., cutout_size/2.)
    with fits.open(f_file) as hdul:
        head = hdul[0].header
        ra_ctr, dec_ctr = head["RA_OBJ"], head["DEC_OBJ"]
        ra_con, dec_con = t_cont["RA"], t_cont["DEC"]
        x_abs_con, y_abs_con = [], []
        _, _, _, _, _, _, col_ctr, row_ctr, _ = \
            tess_stars2px_function_entry(1, ra_ctr, dec_ctr,
                                         trySector=head["SECTOR"])
        for i in range(len(ra_con)):
            _, _, _, _, _, _, col_con, row_con, _ = \
                tess_stars2px_function_entry(1, ra_con[i], dec_con[i],
                                             trySector=head["SECTOR"])
            x_abs_con.append(col_con[0])
            y_abs_con.append(row_con[0])
        x_con = np.array(x_abs_con - col_ctr[0]).flatten() + xy_ctr[0]
        y_con = np.array(y_abs_con - row_ctr[0]).flatten() + xy_ctr[1]
        cont_positions = np.array([x_con, y_con]).T
        return cont_positions
        

def run_test_for_contaminant(
    xy_arr,
    file_in,
    t_cont,
    d_target,
    scc,
    aper_rad=1.0,
    sky_ann=(6.0, 8.0),
    store_lc=True,
    lc_cont_dir="lc_cont",
):
    """Run the periodogram analyses for neighbouring contaminants if required.

    parameters
    ----------
    xy_arr : `list`, size=2
        The X and Y positions of the contaminant (the output form the
        "find_XY_cont" function).
    file_in : `str`
        The name of the fits file containing the contaminant.
    t_cont : `astropy.table.Table`
        A single row from the contamination table which has
        details of the flux contribution.
    d_target : `dict`
        The dictionary returned from the periodogram analysis of
        the target star.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    aper_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file
    lc_cont_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files for contaminants if
        store_lc==True.

    returns
    -------
    name_lc : `str`
        The name of the file that the contaminant lightcurve will be saved to.
    labels_cont : `str` (a, b, c or d)
        A single character which assess if the calculated period for the target
        could actually come from the contaminant.

        a. At least 1 contaminant has a similar period to the target.

        b. No contaminants with similar periods

        c. The aperture photometry extraction failed for the contaminant

        d. Something went wrong with the routine.

    clean_norm_lc_cont : `astropy.table.Table`
    d_cont : `dict`
        The dictionary returned from the periodogram analysis of the
        contaminant star.
    """

    clean_norm_lc_cont, name_lc, d_cont = None, None, None
    try:
        xy_con = tuple((xy_arr[0], xy_arr[1]))
        phot_cont, _ = aper_run(file_in, t_cont, xy_pos=xy_con, aper_rad=aper_rad, sky_ann=sky_ann, fix_rad=False)
        if phot_cont is not None:
            name_lc = f'lc_{t_cont["source_id_target"]}_{t_cont["source_id"]}'\
                      f'_{scc[0]:04d}_{scc[1]}_{scc[2]}.csv'
            clean_norm_lc_cont, _, _ = make_lc(
                phot_cont, store_lc=store_lc, lc_dir=lc_cont_dir
            )[0]
            if len(clean_norm_lc_cont) != 0:
                d_cont = run_ls(clean_norm_lc_cont, name_pg=None)
                false_flag, reliable_flag = is_period_cont(d_target, d_cont, t_cont)
        else:
            false_flag, reliable_flag = 2, 2
    except:
        logger.error(f"something went wrong with measuring the period for"
                     f"{name_lc}")
        false_flag, reliable_flag = 2, 2
    logger.info(f"label for this contaminant: {name_lc}")
    return name_lc, false_flag, reliable_flag, clean_norm_lc_cont, d_cont


def get_tess_pixel_xy(t_targets):
    '''Get the pixel x-y positions for all targets in a Sector/Camera/CCD mode.

    For a given pair of celestial sky coordinates, this function returns table
    rows containing the sector, camera, CCD, and x/y position of the full-frame
    image fits file, so that all stars located in a given (large) fits file can
    be processed simultaneously. After the table is returned, the input table
    is joined to the input table on the source_id, to ensure this function only
    needs to be called once.

    This function is only used if the tessilator program runs over the
    calibrated full-frame images - i.e., when the "all_cc" function is called.
    
    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table created by the function get_gaia_data.py.

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



def make_2d_cutout(file_in, phot_table, im_size=(20,20)):
    '''Makes a 2D cutout object of a target using the median time-stacked
    image.

    parameters
    ----------
    file_in : `astropy.table.Table`
        The astropy table containing the output from the aperture photometry
        for a given target.
    phot_table : `list`
        The list of fits files used to make the aperture photometry.
    im_size : `tuple`, optional, default=(20,20)
        The required size of the 2D-cutout object.

    returns
    -------
    cutout : `astropy.nddata.Cutout2D`
        A 2D-cutout object.
    ctr_pt : `tuple`
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
        logger.error(f"Fits file {file_in} has the invalid type: "
                     f"{type(file_in)}")
        return None
    return cutout, ctr_pt


def get_cutouts(
    coord,
    cutout_size,
    name_target,
    choose_sec=None,
    tot_attempts=3,
    cap_files=None,
    fits_dir="fits",
):
    """Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.

    parameters
    ----------
    coord : `astropy.coordinates.SkyCoord`
        A set of coordinates in the SkyCoord format.
    cutout_size : `float`
        The pixel length of the downloaded cutout.
    name_target : `str`
        Name of the target.
    choose_sec : `None`, `int` or `Iterable`, optional, default=None
        The sector, or sectors required for download.

        * If `None`, TESScut will download all sectors available for the
          target.

        * If `int`, TESScut will attempt to download this sector number.

        * If `Iterable`, TESScut will attempt to download a list of sectors.

    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.

    Returns
    -------
    manifest : `list`
        A list of the fits files for lightcurve analysis.
    """
    if choose_sec is None:
        choose_sec = Tesscut.get_sectors(coordinates=coord)["sector"].data
        logger.info(f"There are {len(choose_sec)} in total: {choose_sec}")
        logger.info(f"There are {len(choose_sec)} in total: {choose_sec}")
        if len(choose_sec) == 0:
            logger.error(f"Sorry, no TESS data available for {name_target}")
            return []
        np.random.shuffle(choose_sec)
    if isinstance(choose_sec, int):
        choose_sec = [choose_sec]

    choose_sec = choose_sec[:cap_files]
    if not np.all(np.isin(choose_sec, np.arange(1, sec_max + 1))):
        raise ValueError(
            f"Invalid sector numbers: {choose_sec}, sectors are 1-{sec_max}."
        )

    manifest = []
    for c in choose_sec:
        num_attempts = 0

        while num_attempts < tot_attempts:
            filename = glob(f"{fits_dir}/{name_target}_{c:04d}*.fits")
            if len(filename) == 1:
                manifest.append(filename[0])
                break
            else:
                try:
                    dl = Tesscut.download_cutouts(
                        coordinates=coord, size=cutout_size, sector=c, path=fits_dir
                    )
                    manifest.append(dl["Local Path"][0])
                    break
                except:
                    print(
                        f"Didn't get Sector {c} data for {name_target}, "
                        f"attempt {num_attempts+1} of {tot_attempts}"
                    )
                    logger.error(
                        f"Didn't get Sector {c} data for "
                        f"{name_target}, attempt {num_attempts+1} of "
                        f"{tot_attempts}"
                    )
            num_attempts += 1

            if num_attempts == tot_attempts:
                print(f"No data for {name_target} in Sector {c}")
                logger.error(f"No data for {name_target} in Sector {c}")
    return manifest


def one_source_cutout(
    target,
    lc_con,
    flux_con,
    make_plots,
    res_table,
    ref_name,
    gaia_sys=True,
    xy_pos=(10.0, 10.0),
    ap_rad=1.0,
    sky_ann=(6.0, 8.0),
    fix_rad=False,
    n_cont=10,
    cont_rad=10.0,
    mag_lim=3.0,
    keep_data=False,
    save_phot=False,
    cbv_flag=False,
    choose_sec=None,
    store_lc=False,
    cutout_size=20,
    tot_attempts=3,
    cap_files=None,
    fits_dir="fits",
    lc_dir="lc",
    pg_dir="pg",
    fix_noise=False,
    shuf_per=False,
    make_shuf_plot=False,
    shuf_dir="shuf_plots",
):
    """Download cutouts and run lightcurve/periodogram analysis for one target.

    Called by the function "all_sources_cutout".

    parameters
    ----------
    target : `astropy.table.row.Row`
        A row of data from the astropy table.
    lc_con : `bool`
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    res_table : `astropy.table.Table`
        The table to store the final tessilator results.
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.
    xy_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run the
        calc_rad algorithm.
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    keep_data : `bool`
        Choose to save the input data to file.
    save_phot : `bool`, optional, default=False
        Decide whether to save the full results from the aperture photometry.
    cbv_flag : `bool`, optional, default=False
        Decide whether to run the lightcurve analysis with a CBV-correction
        applied.
    choose_sec : `None`, `int` or `Iterable`, optional, default=None
        The sector, or sectors required for download.

        * If `None`, TESScut will download all sectors available for the
          target.

        * If `int`, TESScut will attempt to download this sector number.

        * If `Iterable`, TESScut will attempt to download a list of sectors.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file.
    cutout_size : `float`, optional, default=20.
        The pixel length of the downloaded cutout.
    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    cap_files : `None`, `int`, optional, default=None
        The maximum number of sectors for each target.
    fits_dir : `str`, optional, default='fits'
        The name of the directory to store the fits files.
    lc_dir : `str`, optional, default='lc'
        The directory used to store the lightcurve files if store_lc==True.
    pg_dir : `str`, optional, default='pg'
        The directory used to store the periodogram data.
    fix_noise : `bool`, optional, default=False
        Choose to apply corrections accounting for systematic noise.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period analysis.

    returns
    -------
    Nothing returned. Results are saved to table and plots are generated (if
    specified).
    """

    # Set the contaminant parameters to the default values in case
    # they have not been added
#    if 'log_tot_bg' not in target.colnames:
#        target.add_column(-999, name='log_tot_bg')
#        target.add_column(-999, name='log_max_bg')
#        target.add_column(0,    name='n_contaminants')
    name_target = target['name'].replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]

    coo = SkyCoord(target["ra"], target["dec"], unit="deg")

    # use Tesscut to get the cutout fits files for the target star
    # there may be more than 1 fits file if the target lands in
    # multiple sectors!
    fits_files = get_cutouts(coo, cutout_size, name_target,
                             tot_attempts=tot_attempts, choose_sec=choose_sec,
                             cap_files=cap_files, fits_dir=fits_dir)
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
                    f_new = f'{fits_dir}/'+'_'.join([name_target,
                                                     f_sp[1][1:],
                                                     f_sp[2],
                                                     f_sp[3][0]])+'.fits'
                    os.rename(f'./{file_in}', f_new)
                    logger.info(f"target: {target['source_id']}, "
                                f"sector: {f_sp[1][1:]}, "
                                f"{m+1}/{len(fits_files)}")
    # run the lightcurve analysis for the given target/fits file
                elif len(f_sp) == 1:
                    f_new = f'{fits_dir}/{f_sp[0]}'
                else:
                    f_new = file_in
                t_sp = f_new.split('_')
    # simply extract the sector, ccd and camera numbers from the fits file.
                scc = [int(t_sp[-3][1:]), int(t_sp[-2]), int(t_sp[-1][0])]
                print(f_new, scc)
                
                full_run_lc(f_new, target, make_plots, scc, res_table, gaia_sys=gaia_sys, 
                            xy_pos=xy_pos, ap_rad=ap_rad, sky_ann=sky_ann, fix_rad=fix_rad,
                            n_cont=n_cont, cont_rad=cont_rad, mag_lim=mag_lim, tot_attempts=tot_attempts,
                            ref_name=ref_name, cutout_size=cutout_size, save_phot=save_phot,
                            cbv_flag=cbv_flag, flux_con=flux_con,
                            store_lc=store_lc, lc_con=lc_con, lc_dir=lc_dir,
                            pg_dir=pg_dir,
                            fix_noise=fix_noise, shuf_per=shuf_per, make_shuf_plot=make_shuf_plot,
                            shuf_dir=shuf_dir)
            except Exception as e:
                logger.error(f"Error occurred when processing {file_in}. "
                             f"Trying next target.")

  
    

def all_sources_cutout(t_targets, period_file, lc_con, flux_con, make_plots, 
                       ref_name, gaia_sys=True, xy_pos=(10.,10.), ap_rad=1., sky_ann=(6.,8.), fix_rad=False,
                       n_cont=10, cont_rad=10., mag_lim=3.,
                       choose_sec=None, save_phot=False,
                       cbv_flag=False, store_lc=False, tot_attempts=3,
                       cap_files=None, res_ext='results', lc_ext='lc',
                       pg_ext='pg', fits_ext='fits', keep_data=False,
                       fix_noise=False, shuf_per=False, shuf_ext='shuf_plots', make_shuf_plot=False):
    '''Run the tessilator for all targets.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        Table of input data for the tessilator, with the following columns:

        * name: name of the target (`str`)

        * source_id: Gaia DR3 source identifier (`str`)

        * ra: right ascension

        * dec: declination

        * parallax: parallax

        * Gmag: Gaia DR3 apparent G-band magnitude

        * BPmag: Gaia DR3 apparent BP-band magnitude

        * RPmag: Gaia DR3 apparent RP-band magnitude

        * log_tot_bg: log-10 value of the flux ratio between contaminants
          and target (optional)

        * log_max_bg: log-10 value of the flux ratio between the largest
          contaminant and target (optional)

        * num_tot_bg: number of contaminant sources (optional)

    period_file : `str` 
        Name of the file to store periodogram results.
    lc_con : `bool`
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : `bool`
        Decides if the flux contribution from contaminants is to be calculated.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.
    xy_pos : `tuple`, size=2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    ap_rad : `float`, optional, default=1.
        The size of the aperture radius in pixels.
    sky_ann : `tuple`, optional, default=(6.,8.)
        A 2-element tuple defining the inner and outer annulus to calculate
        the background flux.
    fix_rad : `bool`, optional, default=False
        If True, then set the aperture radius equal to ap_rad, otherwise run
        the calc_rad algorithm.
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    choose_sec : `None`, `int`, or `Iterable`, optional, default=None
        The sector, or sectors required for download.
        
        * If `None`, TESScut will download all sectors available for the
          target.

        * If `int`, TESScut will attempt to download this sector number.

        * If `Iterable`, TESScut will attempt to download a list of sectors.

    save_phot : `bool`, optional, default=False
        Decide whether to save the full results from the aperture photometry.
    cbv_flag : `bool`, optional, default=False
        Decide whether to run the lightcurve analysis with a CBV-correction
        applied.
    store_lc : `bool`, optional, default=False
        Choose to save the cleaned lightcurve to file.
    tot_attempts : `int`, optional, default=3
        The number of sql queries in case of request or server errors.
    cap_files : `int`, optional, default=None
        The maximum number of sectors for each target.
    res_ext : `str`, optional, default='results'
        The directory to store the final results file.
    lc_ext : `str`, optional, default='lc'
        The directory used to store the lightcurve files if lc_dir==True
    pg_ext : `str`, optional, default='pg'
        The directory used to store the periodogram data.
    fits_ext : `str`, optional, default='fits'
        The name of the directory to store the fits files.
    keep_data : `bool`
        Choose to save the input data to file.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.
    shuf_per : `bool`, optional, default=False
        Choose to run the shuffled period analysis (True=yes, False=no)
    make_shuf_plot : `bool`, optional, default=False
        Choose to make a plot for the shuffled period analysis
    shuf_dir : `str`, optional, default='plot_shuf'
        The name of the directory to save the plots of the shuffled period
        analysis. 

    returns
    -------
    Nothing returned. The final table is saved to file and the program
    terminates.
    '''

    start = datetime.now()
    logger.info(f"Starting Time: {start}")
    print("Start time: ", start.strftime("%d/%m/%Y %H:%M:%S"))

    fits_dir = make_dir(fits_ext, ref_name)
    lc_dir = make_dir(lc_ext, ref_name)
    pg_dir = make_dir(pg_ext, ref_name)
    shuf_dir = make_dir(shuf_ext, ref_name)
    res_dir = make_dir(res_ext, ref_name)

    res_table = create_table_template()
#    if 'log_tot_bg' not in t_targets.colnames:
#        t_targets.add_column(-999, name='log_tot_bg')
#        t_targets.add_column(-999, name='log_max_bg')
#        t_targets.add_column(0,    name='num_tot_bg')
    for i, target in enumerate(t_targets):
        logger.info(f"{target['name']} (Gaia DR3 {target['source_id']}), star #{i+1}"
                    f" of {len(t_targets)}")
        one_source_cutout(
            target,
            lc_con,
            flux_con,
            make_plots,
            res_table,
            ref_name,
            gaia_sys=gaia_sys,
            xy_pos=xy_pos,
            ap_rad=ap_rad,
            sky_ann=sky_ann,
            fix_rad=fix_rad,
            keep_data=keep_data,
            n_cont=n_cont,
            cont_rad=cont_rad,
            mag_lim=mag_lim,
            save_phot=save_phot,
            cbv_flag=cbv_flag,
            store_lc=store_lc,
            choose_sec=choose_sec,
            tot_attempts=tot_attempts,
            cap_files=cap_files,
            fits_dir=fits_dir,
            lc_dir=lc_dir,
            pg_dir=pg_dir,
            fix_noise=fix_noise,
            shuf_per=shuf_per,
            shuf_dir=shuf_dir,
            make_shuf_plot=make_shuf_plot,
        )
    finish = datetime.now()
    dt_string = finish.strftime("%b-%d-%Y_%H:%M:%S")

    res_table.write(f'{res_dir}/{period_file}_{dt_string}.csv')

    hrs_mins_secs = print_time_taken(start, finish)
    print(f"Finished {len(t_targets)} targets in {hrs_mins_secs}")
    logger.info(f"Total time taken: {hrs_mins_secs}")


def get_fits(scc):
    '''Function which returns a list of fits files corresponding to a
    given Sector, Camera and CCD configuration.

    parameters
    ----------
    sector_num : `int`
        The required sector number.

    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    
    file_dir : `str`
        The name of the base directory containin the fits files.

    returns
    -------
    fits_files : `list`
        A list of the fits files to be used for aperture photometry.
    '''
    
    list_fits = sorted(glob(f"../tess_fits_files/sector{scc[0]:02d}/*.fits"))
    l_cam = np.array([int(j.split('-')[2]) for j in list_fits])
    l_ccd = np.array([int(j.split('-')[3]) for j in list_fits])
    fits_indices = (l_cam == scc[1]) & (l_ccd == scc[2])
    fits_files = np.array(list_fits)[fits_indices]
    print(fits_files)
    return fits_files




def one_cc(t_targets, scc, make_plots, res_table, file_ref, ap_rad=1.0,
                      sky_ann=[6.,8.], keep_data=False, fix_noise=False):
    '''Run the tessilator for targets in a given Sector/Camera/CCD
    configuration.

    This routine finds the full-frame calibrated fits files and targets which
    land in a given Sector/Camera/CCD configuration (SCC). Aperture photometry
    is carried out simultaneously for all stars in a given SCC for each fits
    file in chronological order. This makes the method run much faster than
    doing it star-by-star (i.e. vectorisation). The output is a table for each
    SCC and plots for each target (if required).

    parameters
    ----------
    t_targets : `astropy.table.Table`
        Table containing the targets to be analysed.
    scc : `list`, size=3
        List containing the sector number, camera and CCD.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    res_table : `astropy.table.Table`
        The table to store tessilator results.
    file_ref : `str`
        A common string to give all output files the same naming convention.
    ap_rad : `float`, optional, default=1.0
        The pixel radius of the flux collecting area for aperture photometry.
    sky_ann : `Iterable`, size=2, optional, default=[6.,8.]
        The inner and outer background annuli used for aperture photometry.
    keep_data : `bool`
        Choose to save the input data to file.
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
    suffix_file = f'scc_{scc[0]:02d}_{scc[1]}_{scc[2]}'
    lc_dir = make_dir(f'lc_{suffix_file}', file_ref)
    pg_dir = make_dir(f'pg_{suffix_file}', file_ref)
    full_run_lc(
        fits_files,
        t_targets[ind],
        make_plots,
        scc,
        res_table,
        ap_rad=ap_rad,
        sky_ann=sky_ann,
        keep_data=keep_data,
        fix_noise=fix_noise,
        lc_dir=lc_dir,
        pg_dir=pg_dir,
    )


def all_sources_sector(t_targets, scc, make_plots, period_file, file_ref,
                       keep_data=False, fix_noise=False):
    """Iterate over all cameras and CCDs for a given sector.

    This routine iterates over all cameras and CCDs for a given sector and
    performs the tessilator analysis for each camera/CCD configuration.

    Parameters
    ----------
    t_targets : `astropy.table.Table`
        Input data for the targets to be analysed.
    scc : tuple or int
        This can be a single integer or a tuple of size 3. If a single integer
        is given, then all cameras and CCDs for that sector are analysed. If a
        tuple is given, is has the format `(sector, camera, CCD)`.
    make_plots : `bool`
        Decides is plots are made from the lightcurve analysis.
    period_file : `str`
        Name of file for recording parameters measured by the periodogram
        analysis.
    file_ref : `str`
        A common string to give all output files the same naming convention.
    keep_data : `bool`
        Choose to save the input data to file.
    fix_noise : `bool`, optional, default=False
        Choose to apply the noise correction to the cleaned lightcurve.

    Returns
    -------
    Nothing returned. The Tessilator data for each camera/CCD configuration
    is saved to file.
    """
    if len(scc) == 3:
        sector = scc[0]
        cameras = [scc[1]]
        ccds = [scc[2]]
    elif isinstance(scc, int):
        sector = scc
        cameras, ccds = np.mgrid[1:5, 1:5]
        cameras = cameras.flatten()
        ccds = ccds.flatten()
    for cam, ccd in zip(cameras, ccds):
        start = datetime.now()
        logger.info(f"Starting Time: {start}")
        print("Start time: ", start.strftime("%d/%m/%Y %H:%M:%S"))

        res_table = create_table_template()
        one_cc(
            t_targets,
            [sector, cam, ccd],
            make_plots,
            res_table,
            file_ref=file_ref,
            keep_data=keep_data,
            fix_noise=fix_noise,
        )
        res_table.write(f"{period_file}_{cam}_{ccd}.csv", overwrite=True)
        finish = datetime.now()
        hrs_mins_secs = print_time_taken(start, finish)
        print(
            f"Finished {len(res_table)} targets for Sector {scc[0]},"
            f" Camera {scc[1]}, CCD {scc[2]} in {hrs_mins_secs}"
        )


__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__],
           predicate = lambda f: inspect.isfunction(f)
           and f.__module__ == __name__)]
