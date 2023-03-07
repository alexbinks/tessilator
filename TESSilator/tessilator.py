'''
Alexander Binks and Moritz Guenther, 2023

The TESSilator

Licence: MIT

This is a python3 program designed to provide an all-in-one module to measure
lightcurves and stellar rotation periods from the Transiting Exoplanet Survey
Satellite (TESS). Whilst there are many useful (and powerful) software tools
available for working with TESS data, they are mostly provided as various steps
in the data reduction process --- to our knowledge there are no programs that
automate the full process from downloading the data (start) to obtaining
rotation period measurements (finish). The software provided here fills this
gap. The user provides a table of targets with basic Gaia DR3 information
(source ID, sky positions, parallax and Gaia G magnitude) and simply allows the
TESSilator to do the rest! The steps are:

(1) download photometric time-series data from TESS.

(2) scan the Gaia DR3 catalogue to quantify the level of background
    contamination from nearby sources.

(3) clean the lightcurves for poor quality data caused by systematic and
    instrumental effects.

(4) normalize and detrend lightcurves over the whole sector of observations.

(5) measure stellar rotation periods using two processes: the Lomb-Scargle
    periodogram and the Auto-Correlation Function.

(6) quantify various data quality metrics from photometric time-series data
    which can be used by the user to assess data reliability

In this module, the data is downloaded from TESSCut (Brasseur et al. 2019) -- a
service which allows the user to acquire a stack of n 20x20 pixel
"postage-stamp" image frames ordered in time sequence and centered using the
celestial coordinates provided by the user, where n represents the number of
TESS sectors available for download. It uses modules from the TesscutClass
(astroquery.readthedocs.io/en/latest/api/astroquery.mast.TesscutClass.html) - 
part of the astroquery.mast package) to download the data, then applies steps
2-6, sector-by-sector.

Since there are no requirements to download the full image frames (0.5-0.7TB
and 1.8-2.0TB per sector for 30 and 10 min cadence observations, respectively)
this software is recommended for users who require a relatively fast extraction
for a manageable number of targets (i.e., < 5000). With the correct
pre-requisite Python modules and an uninterrupted internet connection, for a
target with 5 sectors of TESS data, the processing time is approximately 1-2
minutes (depending whether or not the user wants to measure contamination
and/or generate plots). If the user is interested in formulating a much larger
survey, we recommend they obtain the bulk downloads of the TESS calibrated
full-frame images
(archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html) and
run the "tess_large_sectors.py" module, which has been used to measure rotation
periods for catalogues of >1 million targets.

The module "tess_functions.py" contains the functions called to run both
tess_cutouts.py and tess_large_sectors.py

Should there be any problems in using this software please contact Alex Binks
(lead author) at abinks@mit.edu

If this package is useful for research leading to publication we would
appreciate the following acknowledgement:

"The data from the Transiting Exoplanet Survey Satellite (TESS) was acquired
using the TESSilator software package (Binks et al. 2023)."
'''

from .lc_analysis import *
from .contaminants import *
from .maketable import *
from .makeplots import *

from datetime import datetime
import pyinputplus as pyip
import sys, os
from astropy.nddata.utils import Cutout2D


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
print("acquired using the TESSilator software package (Binks et al. 2023).'")
print("\n")
print("Start time: ", start.strftime("%d/%m/%Y %H:%M:%S"))

# Create custom logger
logger = logging.getLogger(__name__)



def create_table_template():
    '''Create a template astropy table to store TESSilator results. 
    '''
    final_table = Table(names=['name', 'source_id', 'ra', 'dec', 'parallax',\
                           'Gmag', 'Sector', 'Camera', 'CCD',\
                           'log_tot_bg_star', 'log_max_bg_star',\
                           'n_contaminants', 'Period_Max', 'Period_Gauss',\
                           'e_Period', 'Period_2', 'power1', 'power1_power2',\
                           'FAP_001', 'amp', 'scatter', 'fdev', 'Ndata',\
                           'cont_flags'],\
                    dtype=(str, str, float, float, float, float, int, int,\
                           int, float, float, int, float, float, float,\
                           float, float, float, float, float, float, float,\
                           int, str))
    return final_table


def setupInputParameters():
    '''Retrieve the input parameters to run the TESSilator program.

    The input parameters are:
    1) "flux_con": the toggle for applying the contamination calculation
       (yes=1, no=0).
    2) either "LC_con", if using the tess_cutouts functions or "sector_num" if
       the tess_sectors function is needed.
       -> "LC_con" determines if lightcurve/periodogram analyses should be
          carried out for neighbouring contaminants (yes=1, no=0).

       -> "sector_num" prompts the user to enter the sector number needed. If
          command line arguments are not used, the program will ask if a
          specific Camera and CCD is needed (1=yes, 0=no). If not required,
          the whole sector is analysed. If this is a command line argument, if
          the user enters just the sector number (maximum 2 digits) the whole
          sector will be analysed, and if the Camera and CCD number are given
          right after the sector number with no spaces, then a specific Camera
          and CCD configuration will be used. E.G: if "sector_num = 8", the
          entire sector 8 is analysed, whereas if "sector_num = 814" then the
          program will analyse only Camera 1 and CCD 4 in sector 8.

    3) "make_plots" gives the user the option to make plots (yes=1, no=0)
    4) "file_ref" is a string expression used to reference the files produced.
    5) "t_filename" is the name of the input file required for analysis.

    If the function is called without all five command line parameters, a set
    of prompts are initiated to receive input. If just one target is needed,
    then the user can simply supply either target name, or the sky coordinates
    as the final input.
    Otherwise, if the full set of command line parameters are supplied, the
    function will use these as the inputs, however, if they have the wrong
    format the program will return a warning message and exit.

    '''
    if len(sys.argv) != 6:
        flux_con = pyip.inputInt("Do you want to search for contaminants? "
                   "1=yes, 0=no : ", min=0, max=1)
        if 'cutout' in sys.argv[0]:
            LC_con = pyip.inputInt("Do you want to calculate period data for "
                     "the contaminants? 1=yes, 0=no : ", min=0, max=1)
        elif 'sector' in sys.argv[0]:
            sector_num = pyip.inputInt("Which sector of data do you require? "
                         "(1-55) : ", min=1, max=55)
            cc_request = pyip.inputBool("Do you want a specific Camera/CCD? "
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
                         "table or object.\nIf this is an object please use "
                         "double quotations around the target identifier : ")
            if t_filename.startswith('"') & t_filename.endswith('"'):
                t_name = t_filename[1:-1]
                t_name_joined = t_name.replace(' ', '_')+'.dat'
                if os.path.exists(t_name_joined):
                    os.remove(t_name_joined)
                with open(t_name_joined, 'a') as single_target:
                    single_target.write(t_name)
                t_filename = t_name_joined
                break
            if os.path.exists(t_filename) == False:
                print(f'The file "{t_filename}" does not exist.')
            else:
                break
    else:
        flux_con = int(sys.argv[1])
        if 'cutout' in sys.argv[0]:
            LC_con = int(sys.argv[2])
        elif 'sector' in sys.argv[0]:
            scc_in = str(sys.argv[2])
            if len(scc_in) > 4:
                print("Incorrect format for sector/camera/ccd values")
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
        sec_vals = np.arange(1,56)
        cam_ccd_vals = np.arange(1,5)
        while True:
            if flux_con not in true_vals:
                print(f"flux_con value {flux_con} not a valid input. "
                       "Exiting program.")
                sys.exit()
            if 'cutout' in sys.argv[0]:
                if LC_con not in true_vals:
                    print(f"LC_con value {LC_con} not a valid input. "
                           "Exiting program.")
                    sys.exit()
            elif 'sector' in sys.argv[0]:
                if scc[0] not in sec_vals:
                    print(f"sector_num value {scc[0]} not a valid input. "
                           "Exiting program.")
                    sys.exit()
                if len(scc) == 3:
                    if scc[1] not in cam_ccd_vals:
                        print(f"Camera value {scc[1]} out of range.")
                        sys.exit()
                    if scc[2] not in cam_ccd_vals:
                        print(f"CCD value {scc[2]} out of range.")
                        sys.exit()                        
            if make_plots not in true_vals:
                print("make_plots not a valid input. Exiting program.")
                sys.exit()
            if os.path.exists(t_filename) == False:
                print(f'File "{t_filename}" does not exist. Exiting program.')
                sys.exit()
            break
    if 'cutout' in sys.argv[0]:
        return flux_con, LC_con, make_plots, file_ref, t_filename
    elif 'sector' in sys.argv[0]:
        return flux_con, scc, make_plots, file_ref, t_filename


def setupFilenames(file_ref, scc=None):
    '''Set up the file names to store data
    
    parameters
    ----------
    
    file_ref : str
    
    returns
    -------
    con_file : `str'
        Name of file to store contamination values.
    period_file : `str'
        Name of file for recording parameters measured by the periodogram
        analysis.
    '''    
    if scc is None:
        con_file = '_'.join(['contamination', file_ref, 'tesscut'])
        period_file = '_'.join(['periods', file_ref, 'tesscut'])
    else:
        sn = 'sector'+f"{scc[0]:02d}"
        con_file = '_'.join(['contamination', file_ref, sn])
        period_file = '_'.join(['periods', file_ref, sn])
    return con_file, period_file


def test_table_large_sectors(t_filename):
    '''Check if the input file needs modifying at all.
    
    If running the TESSilator for a whole sector, read the input file
    and if the format is ready for analysis, make a couple of adjustments,
    then simply pass the file.
    
    For a straight pass, the columns must be ordered in two ways. Either:
    (1) exactly as set out in the list "cnc".
    (2) the same as (1), but with a preceding column entitled "name",
        which refers to a target identifier name. This can be any string.
    In any other case, None is returned and other functions are used to
    get the table into the correct format.

    parameters
    ----------
    table_in : `astropy.table.Table`
        The table input which will be checked for formatting.

    returns
    -------
    t or `None`
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
        # return nothing if neither of the above two conditions are met.
        return None


def readData(t_filename, name_is_source_id=0):
    '''Read input data and convert to an astropy table ready for analysis
    
    The input data must be in the form of a comma-separated variable and may
    take 3 forms:
    (a) a 1-column table of source identifiers
    (b) a 2-column table of decimal sky coordinates (celestial or galactic)
    (c) a pre-prepared table of 5 columns consisting of
       source_id, ra, dec, parallax, Gmag (without column headers)
       
    parameters
    ----------
    t_filename : `astropy.table.Table`
        name of the file containing the input data
    name_is_source_id : int, optional, Default=0
        when running option (c), this toggle if == 1 will automatically set
        the "name" column as the Gaia DR3 identifiers. This avoids long sql
        queries for very large input tables.  

    returns
    -------
    t_targets : `astropy.table.Table`
        a formatted astropy table ready for further analysis
    '''

    logger.info(f"Starting Time: {start}")
    t_input = ascii.read(t_filename, delimiter=',', format='no_header')
    t_targets = getGaiaData(t_input, name_is_source_id)

    return t_targets


def collectContaminationData(t_targets, flux_con, LC_con, con_file, **kwargs):
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
    flux_con : bool
        Decides if the flux contribution from contaminants is to be calculated.
    LC_con : bool
        Decides if a lightcurve analysis is to be performed for the n strongest
        contaminants, where n is a keyword in the "contamination" function from
        tess_functions2.py.
    con_file : str
        The name of the file to store data from the total flux contribution
        from contaminants.
        
    Returns
    -------
    t_targets : `astropy.table.Table`
        Input target table with 3 columns added containing details of the
        contamination: "log_tot_bg", "log_max_bg", "num_tot_bg"
    '''
    if flux_con:
        t_targets, t_con_table = contamination(t_targets, LC_con,
                                                         **kwargs)
        t_contam = t_targets[['source_id', 'log_tot_bg', 'log_max_bg',\
                              'num_tot_bg']]
        t_contam.write(con_file+'.ecsv', overwrite=True)
        if LC_con:
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
    '''Once the TESSilator has analysed a target, the results are printed line
    by line to a table.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The Gaia and contamination details (see 'getGaiaData' and
        'contamination' functions in TESSilator.tess_functions.)

    d_target : `dict`
        The dictionary containing details of the periodogram analysis, which is
        returned by 'run_LS' in TESSilator.tess_functions.)
    
    scc : `list` size=3
        A list containing the Sector, Camera and CCD.

    labels_cont : `str`
        A string of labels listing details of any contaminant sources.
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
          d_target['pops_vals'][1],
          d_target['phase_scatter'],
          d_target['frac_phase_outliers'],
          d_target['Ndata'],
          labels_cont
          ]
    return dr

def make_failrow(t_target, scc):
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
    for i in range(10):
        dr.append(np.nan)
    dr.append(0)
    dr.append('z')
    return dr


def full_run_LC(file_in, t_target, make_plots, scc, final_table,\
                cutout_size=20, flux_con=0, LC_con=0, con_file=0,\
                XY_pos=(10.,10.)):
    '''Aperture photometry, lightcurve cleaning and periodogram analysis.
    
    This function calls a set of functions in the tessilator_functions.py to
    perform aperture photometry, clean the lightcurves from spurious data and
    runs the Lomb-Scargle periodogram to measure rotation periods.

    parameters
    ----------
    file_in : str
        name of the input TESS fits file
    t_target : `astropy.table.Table`
        details of the target star
    make_plots: `bool'
        decides if plots are made 
    scc : list (3)
        sector, camera, ccd
    cutout_size : int
        the pixel length of the downloaded cutout
    flux_con : bool, optional, default=0
        Decides if the flux contribution from contaminants is to be calculated.
    LC_con : bool, optional, default=0
        Decides if a lightcurve analysis is to be performed for the n strongest
        contaminants, where n is a keyword in the "contamination" function from
        tess_functions2.py.
    con_file : str
        The name of the file to store data from the total flux contribution
        from contaminants.
    XY_pos : tuple 2x2, optional, default=(10.,10.)
        The centroid of the target in pixels.
    returns
    -------
    the final period file
    a plot of the lightcurve (if requested)
    '''


    try:
        table_phot_final = aper_run(file_in, t_target, XY_pos=XY_pos)
    except Exception as e:
        logger.error(f"aperture photometry: of {file_in} failed to run")

    if len(table_phot_final) < 10:
        logger.error(f"aperture photometry: failed to produce enough data "
                     f"points for {t_target['source_id']}")
        for t in t_target:
            final_table.add_row(make_failrow(t, scc))
        return None
    phot_targets = table_phot_final.group_by('id')
    for key, group in zip(phot_targets.groups.keys, phot_targets.groups):
        
        if isinstance(t_target, Table):
            t_targets = t_target[t_target["source_id"] == key[0]]
        else:
            t_targets = Table(t_target)
            t_targets["source_id"] = t_targets["source_id"].astype(str)
        clean_norm_lc, original_norm_lc = make_lc(group)
        if len(clean_norm_lc) == 0:
            logger.error(f"no datapoints to make lightcurve analysis for "
                         f"{t_targets['source_id']}")
            for t in t_targets:
                final_table.add_row(make_failrow(t, scc))
            return None
        d_target = run_LS(clean_norm_lc)
        if LC_con:
            if flux_con != 1:
                print("Contaminants not identified! Please toggle LC_con=1")
                print("Continuing program using only the target.")
            else:
                print('calculating contaminant lightcurves')
                con_table = ascii.read(con_file+
                                       '_individiual.ecsv')
                con_table = con_table[con_table["source_id_target"] == \
                                      t_target["source_id"].astype(str)]
                XY_con = find_XY_cont(file_in, con_table, cutout_size)
                labels_cont = ''
                for z in range(len(XY_con)):
                    labels_cont += run_test_for_contaminant(XY_con[z],\
                                                            file_in,\
                                                            con_table[z],\
                                                            d_target,\
                                                            cutout_size)
                if not labels_cont:
                    labels_cont = '0'
        else:
            labels_cont = 'z'

        if make_plots:
            im_plot, XY_ctr = make_2D_Cutout(file_in, group, \
                                             im_size=(cutout_size+1,\
                                                      cutout_size+1))
            make_LC_plots(im_plot, clean_norm_lc, original_norm_lc,\
                          d_target, scc, t_targets, XY_contam=XY_con)

        final_table.add_row(make_datarow(t_targets, scc, d_target,\
                                         labels_cont))
        

def print_time_taken(start, finish):
    '''Calculate the time taken for a process.
    
    This function takes a start and finish point a calculates the time taken in
    hours, minutes and seconds

    parameters
    ----------
    start : `datetime.datetime'
        The start point of the process
    
    finish : `datetime.datetime'
        The end point of the process

    returns
    -------
    hrs : `int'
        number of hours
    mins : `int'
        number of minutes
    secs : `int'
        number of seconds
    '''
    time_in_secs = (finish - start).seconds
    mins, secs = divmod(time_in_secs, 60)
    hrs, mins = divmod(mins, 60)
    return hrs, mins, secs



#'''TESS CUTOUTS

#THE FOLLOWING FUNCTIONS ARE TO RUN THE TESSILATOR USING TESS CUTOUTS'''

def find_XY_cont(f_file, con_table, cutout_size):
    '''Identify the pixel X-Y positions for contaminant sources.

    If the user requests a periodogram analysis of neighbouring potential
    contaminants, this function returns their X-Y positions, which are
    used as the centroids for aperture photometry.

    parameters
    ----------
    f_file : `str'
        The name of the fits file.
    con_table : `astropy.table.Table'
        The table containing Gaia data of the contaminants.
    cutout_size : `int'
        The length size of the TESS cutout image.

    returns
    -------
    cont_positions : `np.array'
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
        

def run_test_for_contaminant(XY_arr, file_in, con_table, d_target, cutout_size):
    '''Run the periodogram analyses for neighbouring contaminants if required.

    parameters
    ----------
    XY_arr : `list` size=2
        The X and Y positions of the contaminant (calculated in
        TESSilator.tess_functions.find_XY_cont)
    file_in : `str`
        The name of the fits file containing the contaminant.
    con_table : `astropy.table.Table`
        A single row from the contamination table which has
        details of the flux contribution.
    d_target : `dict`
        The dictionary returned from the periodogram analysis of
        the target star (returned from TESSilator.tess_functions.run_LS)

    returns
    -------
    labels_cont : `str` (a, b, c or d)
        A single character which assess if the calculated period for the target
        could actually come from the contaminant. 
    '''

    XY_con = tuple((XY_arr[0], XY_arr[1]))
    phot_cont = aper_run(file_in, con_table, Rad=1.,
                         SkyRad=(6.,8.), XY_pos=(10.,10.))
    if phot_cont is None:
        labels_cont = 'd'
    else:
        clean_norm_lc_cont, original_norm_lc_cont = make_lc(phot_cont)
        if len(clean_norm_lc_cont) != 0:
            d_cont = run_LS(clean_norm_lc_cont)
            labels_cont = isPeriodCont(d_target, d_cont, con_table)
        else:
            labels_cont = 'd'
    return labels_cont



def get_cutouts(coord, cutout_size, choose_sec, target_name):
    '''Download TESS cutouts and store to a list for lightcurve analysis.

    The TESScut function will save fits files to the working directory.
    
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    cutout_size : float
        The pixel length of the downloaded cutout
    choose_sec : `None', `int' or `Iterable'
        The sector, or sectors required for download.
        If `None', TESScut will download all sectors available for the target.
        If `int', TESScut will attempt to download this sector number.
        If `Iterable', TESScut will attempt to download a list of sectors.
    target_name : `str'
        Name of the target
    returns
    -------
    manifest : `list'
        A list of the fits files for lightcurve analysis.
    '''
    manifest = []
    if choose_sec is None:
        dl = Tesscut.download_cutouts(coordinates=coord, size=cutout_size,\
                                      sector=None)
        for d in dl["Local Path"]:
            manifest.append(d)
    elif isinstance(choose_sec, int):
        if (choose_sec < 1 or choose_sec > 70):
            print(f"Sector {choose_sec} is out of range.")
            logger.error(f"Sector {choose_sec} is out of range.")
        else:
            try:
                dl = Tesscut.download_cutouts(coordinates=coord,\
                                              size=cutout_size,\
                                              sector=choose_sec)
                manifest.append(dl["Local Path"][0])
            except:
                print(f"Sector {choose_sec} unavailable for {target_name}")
                logger.error(f"Sector {choose_sec} unavailable for "
                             f"{target_name}")
    elif isinstance(choose_sec, Iterable):
        cs = np.array(list(set(choose_sec)))
        if all(isinstance(x, np.int64) for x in cs):
            cs_g = cs[np.where((cs > 0) & (cs < 70))[0]]
            if len(cs) != len(cs_g):
                logger.warning(f"Sectors {np.setdiff1d(cs, cs_g)} are out of "
                               f"range.")
            for c in cs_g:
                try:
                    dl = Tesscut.download_cutouts(coordinates=coord,\
                                                  size=cutout_size,\
                                                  sector=c)
                    manifest.append(dl["Local Path"][0])
                except:
                    print(f"Sector {c} unavailable for {target_name}")
                    logger.error(f"Sector {c} unavailable for {target_name}")
        else:
            print("Some sectors not of type `int'. Fix and try again.")
            logger.error("Some sectors not of type `int'. Fix and try again.")
    else:
        print(f"The choose_sec parameter has an invalid format type: "
              f"{type(choose_sec)}")
        logger.error(f"The choose_sec parameter has an invalid format type: "
                     f"{type(choose_sec)}")
    return manifest


def one_source_cutout(coord, target, LC_con, flux_con, con_file, make_plots,\
                      final_table, choose_sec, cutout_size=20):
    '''Download cutouts and run lightcurve/periodogram analysis for one target.

    Called by the function "iterate_sources".
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    target : `astropy.table.row.Row`
        A row of data from the astropy table
    LC_con : bool
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : bool
        Decides if the flux contribution from contaminants is to be calculated.
    con_file : str
        The name of the file to store data from the total flux contribution
        from contaminants.
    make_plots : bool
        Decides is plots are made from the lightcurve analysis.
    final_table : `astropy.astropy.Table'
        The table to store the final TESSilator results.
    choose_sec : `None', `int' or `Iterable'
        The sector, or sectors required for download.
        If `None', TESScut will download all sectors available for the target.
        If `int', TESScut will attempt to download this sector number.
        If `Iterable', TESScut will attempt to download a list of sectors.
    cutout_size : float, Default=20
        the pixel length of the downloaded cutout
        
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

    # use Tesscut to get the cutout fits files for the target star
    # there may be more than 1 fits file if the target lands in
    # multiple sectors!

    manifest = get_cutouts(coord, cutout_size, choose_sec, target['name'])

    if manifest is None:
        logger.error(f"could not download any data for {target['name']}. "
                     f"Trying next target.")
    else:
        for m, file_in in enumerate(manifest):
            try:
                print(f"{m+1} of {len(manifest)} sectors")
    # rename the fits file to something more legible for users
                name_underscore = str(target['name']).replace(" ", "_")
                f_sp = file_in.split('-')
                file_new = '_'.join([name_underscore, f_sp[1][1:], f_sp[2],\
                                     f_sp[3][0]])+'.fits'
                os.rename(file_in, file_new)
                logger.info(f"target: {target['source_id']}, "
                            f"{m+1}/{len(manifest)}")
    # run the lightcurve analysis for the given target/fits file
                t_sp = file_new.split('_')
    # simply extract the sector, ccd and camera numbers from the fits file.
                scc = [int(t_sp[-3][1:]), int(t_sp[-2]), int(t_sp[-1][0])]
                full_run_LC(file_new, target, make_plots, scc, final_table,
                                   flux_con=flux_con, LC_con=LC_con,
                                   con_file=con_file, XY_pos=(10.,10.))
            except Exception as e:
                logger.error(f"Error occurred when processing {file_new}. "
                             f"Trying next target.")

        
def all_sources_cutout(t_targets, period_file, LC_con, flux_con, con_file,\
                       make_plots, choose_sec=None):
    '''Run the TESSilator for all targets.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        Table of input data for the TESSilator, with the columns
        name --> name of the target (str)
        source_id --> Gaia DR3 source identifier (str)
        ra --> right ascension
        dec --> declination
        parallax --> parallax
        Gmag --> Gaia DR3 apparent G-band magnitude
        log_tot_bg_star --> log-10 value of the flux ratio between contaminants
                            and the target (optional)
        log_max_bg_star --> log-10 value of the flux ratio between the largest
                            contaminant and the target (optional)
        n_contaminants --> number of contaminant sources (optional)
    period_file : str
        Name of the file to store periodogram results.
    LC_con : bool
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
    flux_con : bool
        Decides if the flux contribution from contaminants is to be calculated.
    con_file : str
        The name of the file to store data from the total flux contribution
        from contaminants.
    make_plots : bool
        Decides is plots are made from the lightcurve analysis.
    choose_sec : `None', `int' or `Iterable', optional, default=`None'
        The sector, or sectors required for download.
        If `None', TESScut will download all sectors available for the target.
        If `int', TESScut will attempt to download this sector number.
        If `Iterable', TESScut will attempt to download a list of sectors.

    returns
    -------
    Nothing returned. The final table is saved to file and the program
    terminates.
    '''
    final_table = create_table_template()
    isdir = os.path.isdir('fits')
    if not isdir:
        os.mkdir('fits')
    if t_targets['log_tot_bg'] is None:
        t_targets.add_column(-999, name='log_tot_bg')
        t_targets.add_column(-999, name='log_max_bg')
        t_targets.add_column(0,    name='num_tot_bg')
    for i, target in enumerate(t_targets):
        print(f"{target['name']}, star # {i+1} of {len(t_targets)}")
        coo = SkyCoord(target["ra"], target["dec"], unit="deg")
        one_source_cutout(coo, target, LC_con, flux_con, con_file,
                          make_plots, final_table, choose_sec)
    finish = datetime.now()
    dt_string = finish.strftime("%b-%d-%Y_%H:%M:%S")
    final_table.write(period_file+'_'+dt_string+'.ecsv')

    hrs, mins, secs = print_time_taken(start, finish)
    print(f"Finished {len(t_targets)} targets in {hrs} hours, "
          f"{mins} minutes, {secs} seconds")








#'''TESS SECTORS

#THE FOLLOWING FUNCTIONS ARE TO RUN THE TESSILATOR USING TESS SECTORS'''

def getTESSPixelXY(t_targets):
    '''Get the pixel X-Y positions for all targets in a Sector/Camera/CCD mode.

    For a given pair of celestial sky coordinates, this function returns table
    rows containing the sector, camera, CCD, and X/Y position of the full-frame
    image fits file, so that all stars located in a given (large) fits file can
    be processed simultaneously. This function is only used when the
    "tess_large_sectors.py" method is called. After the table is returned, the
    input table is joined to the input table on the source_id, to ensure this
    function only needs to be called once.

    parameters
    ----------
    t_targets : `astropy.table.Table'
        The input table created by the function getGaiaData.py
    '''
    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
           outColPix, outRowPix, scinfo = tess_stars2px_function_entry(
           t_targets['source_id'], t_targets['ra'], t_targets['dec'])
    return Table([outID, outSec, outCam, outCcd, outColPix, outRowPix],
                names=('source_id', 'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos'))


def get_fits(scc):
    '''Function which returns a list of fits files corresponding to a
    given Sector, Camera and CCD configuration

    parameters
    ----------
    sector_num : `int`
        Sector number required

    cc : `list` size=2
        List of [a, b], where a is the Camera number (1-4)
        and b is the CCD number (1-4)
    
    returns
    -------
    fits_files : `list`
        A list of the fits files to be used for aperture photometry
    '''
    list_fits = sorted([os.path.join(f"sector{scc[0]:02d}", f) \
                        for f in os.listdir(f"sector{scc[0]:02d}") \
                        if f.endswith('.fits')])
    l_cam = np.array([int(j.split('-')[2]) for j in list_fits])
    l_ccd = np.array([int(j.split('-')[3]) for j in list_fits])
    fits_indices = (l_cam == scc[1]) & (l_ccd == scc[2])
    fits_files = np.array(list_fits)[fits_indices]
    return fits_files


def make_2D_Cutout(file_in, phot_table, im_size=(21,21)):
    '''Makes a 2D cutout object (from astropy.nddata.2Dcutout) of a given
    target from the median time-stacked image

    parameters
    ----------
    target : `astropy.table.Table`
        The astropy table containing the output from the aperture photometry
        for a given target.
    fits_files : `list`
        The list of fits files used to make the aperture photometry
    im_size : `tuple`, optional, default=(21,21)
        The required size of the 2D-cutout object

    returns
    -------
    cutout : `astropy.nddata.Cutout2D`
        A 2D-cutout object

    XY_ctr : `tuple`
        A tuple containing the X, Y position of the median time-stacked image.
    '''

    if isinstance(file_in, np.ndarray):
        image_index = math.floor((len(phot_table)-1)/2)
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


def iterate_single_cc(t_targets, scc, make_plots, final_table, Rad=1.0,
                      SkyRad=np.array([6.0,8.0])):
    '''Run the TESSilator for targets in a given Sector/Camera/CCD configuration

    This routine finds the full-frame calibrated fits files and targets which
    land in a given Sector/Camera/CCD configuration (SCC). Aperture photometry
    is carried out simultaneously for all stars in a given SCC for each fits
    file in chronological order. This makes the method run much faster than
    doing it star-by-star (i.e. vectorisation). The aperture photometry is
    carried out using the TESSilator.tess_functions.aper_run_sectors module,
    and all other steps in the TESSilator share the same functions as the TESS
    cutouts. The output is a set of periodogram analyses tables (for each SCC)
    and plots (if required).

    parameters
    ----------

    cc : `list` size=2
        List of [a, b], where a is the Camera number (1-4)
        and b is the CCD number (1-4)

    t_targets : `astropy.table.Table`
        Table containing the targets to be analysed

    sector_num : `int`
        Sector number required

    make_plots : bool
        Decides is plots are made from the lightcurve analysis.
    '''
    fits_files = get_fits(scc)
    ind = (t_targets['Sector'] == scc[0]) & \
          (t_targets['Camera'] == scc[1]) & \
          (t_targets['CCD'] == scc[2])
    if ind.any() == False:
        return

    full_run_LC(fits_files, t_targets[ind], make_plots, scc, final_table)




def iterate_all_cc(t_targets, scc, make_plots, period_file):
    start = datetime.now()
    if len(scc) == 3:
        final_table = create_table_template()
        iterate_single_cc(t_targets, scc, make_plots, final_table)
        final_table.write(f"{period_file}_{scc[1]}_{scc[2]}.ecsv",
                          overwrite=True)
        finish = datetime.now()
        hrs, mins, secs = print_time_taken(start, finish)
        print(f"Finished {len(final_table)} targets for Sector {scc[0]}, "
              f"Camera {scc[1]}, CCD {scc[2]} in "
              f"{hrs} hours, {mins} minutes, {secs} seconds")
    else:
        cam_ccd = np.array([[i,j] for i in range(1,5) for j in range(1,5)])
        for cc in cam_ccd:
            final_table = create_table_template()
            start_ccd = datetime.now()
            iterate_single_cc(t_targets, [scc[0], cc[0], cc[1]], make_plots,
                              final_table)
            final_table.write(f"{period_file}_{cc[0]}_{cc[1]}.ecsv",
                              overwrite=True)
            finish_ccd = datetime.now()
            hrs, mins, secs = print_time_taken(start_ccd, finish_ccd)
            print(f"Finished {len(final_table)} targets for Sector {scc[0]}, "
                  f"Camera {cc[0]}, CCD {cc[1]} in "
                  f"{hrs} hours, {mins} minutes, {secs} seconds")
        finish = datetime.now()
        hrs, mins, secs = print_time_taken(start, finish)
        print(f"Finished the whole of Sector {scc[0]} in "
              f"{hrs} hours, {mins} minutes, {secs} seconds")