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

The calling sequence is:

python tess_cutouts.py flux_con LC_con make_plots file_ref t_filename

where
      "flux_con" is the toggle for applying the contamination calculation
       (yes=1, no=0).
      "LC_con" determines if lightcurve/periodogram analyses should be carried
       out for neighbouring contaminants (yes=1, no=0).
      "make_plots" gives the user the option to make plots (yes=1, no=0)
      "file_ref" is a string expression used to reference the files produced.
      "t_filename" is the name of the input file required for analysis.

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

from .tess_functions2 import *

start_date = time.asctime(time.localtime(time.time()))
start = time.time()
start0 = time.time()
print(start_date)

# Produce an astropy table with the relevant column names and data types. These will be filled in list comprehensions later in the program.
def create_table_template():
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
    '''Retrieve the input parameters to run the program

       If the function is called without the full set of command line
       parameters, then a set of prompts are initiated to receive input. In
       this case it is possible to use the name of a single target if just one
       source is needed for analysis, either as a target name, or a pair of sky
       coordinates.
       Otherwise, if the full set of command line parameters are supplied, the
       function will use these as the inputs, however, if they have the wrong
       format the program will exit with a message.
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
            sector_num = int(sys.argv[2])
        make_plots = int(sys.argv[3])
        file_ref = sys.argv[4]
        t_filename = sys.argv[5]
        true_vals = [0, 1]
        sec_vals = np.arange(1,56)
        while True:
            if flux_con not in true_vals:
                print("flux_con not a valid input. Exiting program.")
                sys.exit()
            if 'cutout' in sys.argv[1]:
                if LC_con not in true_vals:
                    print("LC_con not a valid input, Exiting program.")
                    sys.exit()
            elif 'sector' in sys.argv[1]:
                if sector_num not in sec_vals:
                    print("sector_num not a valid input, Exiting program.")
                    sys.exit()
            if make_plots not in true_vals:
                print("make_plots not a valid input. Exiting program.")
                sys.exit()
            if os.path.exists(t_filename) == False:
                print(f'File "{t_filename}" does not exist. Exiting program.')
                sys.exit()
    if 'cutout' in sys.argv[0]:
        return flux_con, LC_con, make_plots, file_ref, t_filename
    elif 'sector' in sys.argv[0]:
        return flux_con, sector_num, make_plots, file_ref, t_filename


def setupFilenames(file_ref, sector_num=None):
    '''Set up the file names to store data
    
    parameters
    ----------
    
    file_ref : str
    
    returns
    -------
    con_file : to store contamination values.
    period_file : for recording parameters measured by the periodogram
                  analysis.
    store_file : to record any logging information.
    '''    
    if sector_num is None:
        con_file = '_'.join(['contamination', file_ref, 'tesscut'])+'.ecsv'
        period_file = '_'.join(['periods', file_ref, 'tesscut'])+'.ecsv'
        store_file = '_'.join(["cur", file_ref, 'tesscut'])+".txt"
    else:
        sn = 'sector'+f"{sector_num:02d}"
        con_file = '_'.join(['contamination', file_ref, sn])+'.ecsv'
        period_file = '_'.join(['periods', file_ref, sn])+'.ecsv'
        store_file = '_'.join(["cur", file_ref, sn])+".txt"
    return con_file, period_file, store_file


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


def readData(t_filename, storeFile=None, name_is_source_id=0):
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
    store_file : str, optional, Default=None
        Name of the file to store logging details.
    name_is_source_id : int, optional, Default=0
        when running option (c), this toggle if == 1 will automatically set
        the "name" column as the Gaia DR3 identifiers. This avoids long sql
        queries for very large input tables.  

    returns
    -------
    t_targets : `astropy.table.Table`
        a formatted astropy table ready for further analysis
    '''

    if storeFile is not None:
        with open(storeFile, 'w') as file1:
            file1.write('Starting Time: '+str(start_date)+'\n')

    t_input = readascii(t_filename)
    t_targets = getGaiaData(t_input, name_is_source_id)

    return t_targets

# If the user selects flux_con = 1, then the contamination method
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
        Decides if a lightcurve analysis is to be performed for the 5 strongest
        contaminants. Here, the data required for further analysis are
        stored in a table.
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
        t_targets, t_contam, t_con_table = contamination(t_targets, LC_con,
                                                         **kwargs)
        t_contam.write(con_file, overwrite=True)
        if LC_con:
            t_con_table.write(con_file[:-5]+
                              '_individiual'+
                              con_file[-5:], overwrite=True)
    else:
        t_targets["log_tot_bg"] = -999.
        t_targets["log_max_bg"] = -999.
        t_targets["num_tot_bg"] = -999.
    
        if os.path.exists(con_file):
            t_contam = Table.read(con_file)
            for i in range(len(t_contam)):
                g = (t_contam["source_id"] == t_targets["source_id"].astype(str)[i])
                if len(g) >= 1:
                    t_targets["log_tot_bg"][i] = t_contam["log_tot_bg"][g][0]
                    t_targets["log_max_bg"][i] = t_contam["log_max_bg"][g][0]
                    t_targets["num_tot_bg"][i] = t_contam["num_tot_bg"][g][0]
    return t_targets


def make_datarow(t_target, scc, d_target, labels_cont):
    '''Once the TESSilator has analysed a target, the results are printed line by
    line to a table.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The Gaia and contamination details (see 'getGaiaData' and 'contamination'
        functions in TESSilator.tess_functions.)

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


def run_test_for_contaminant(XY_arr, file_in, store_file, con_table, d_target):
    '''Run the periodogram analyses for neighbouring contaminants if required.

    parameters
    ----------
    XY_arr : `list` size=2
        The X and Y positions of the contaminant (calculated in
        TESSilator.tess_functions.find_XY_cont)
    file_in : `str`
        The name of the fits file containing the contaminant.
    store_file : `str`
        Name of the file to store logging details.
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
    phot_cont = np.array(aper_run_cutouts(file_in, XY_con, store_file=store_file))
    phot_cont_table = make_phot_table(phot_cont)
    if phot_cont_table is None:
        labels_cont = 'd'
    else:
        clean_norm_lc_cont, original_norm_lc_cont = make_lc(phot_cont_table)
        if len(clean_norm_lc_cont) != 0:
            d_cont = run_LS(clean_norm_lc_cont, store_file)
            labels_cont = isPeriodCont(d_target, d_cont, con_table)
        else:
            labels_cont = 'd'
    return labels_cont






#'''TESS CUTOUTS

#THE FOLLOWING FUNCTIONS ARE TO RUN THE TESSILATOR USING TESS CUTOUTS'''

def single_LC(file_in, t_target, cutout_size, store_file, LC_con, flux_con,
                 con_file, make_plots):
    '''Run the aperture photometry, lightcurve cleaning and periodogram
    analysis
    
    Called by the function "iterate_one_source".

    parameters
    ----------
    file_in : str
        name of the input TESS cutout fits file
    t_target : `astropy.table.Table`
        details of the target star
    cutout_size : int
        the pixel length of the downloaded cutout
    store_file : str, optional, Default=None
        Name of the file to store logging details.
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

    returns
    -------
    the final period file
    a plot of the lightcurve (if requested)
    '''
    t_sp = file_in.split('_')
    # simply extract the sector, ccd and camera numbers from the fits file.
    scc = [int(t_sp[-3][1:]), int(t_sp[-2]), int(t_sp[-1][0])]
    # create a numpy array which holds the photometric measurements for each
    # time-step from the aper_run_cutouts routine.
    XY_ctr = (cutout_size/2., cutout_size/2.)
    phot_targ = np.array(aper_run_cutouts(file_in, XY_ctr, store_file, make_slices=0))
    phot_targ_table = make_phot_table(phot_targ)
    if phot_targ_table is None:
        ### MAKE A PRINT STATEMENT
        ### WE EVENTUALLY MIGHT WANT TO LOG THIS
        ### CHECK OUT PYTHON LOGGING
        return None
    clean_norm_lc, original_norm_lc = make_lc(phot_targ_table)
    if len(clean_norm_lc) != 0:
        d_target = run_LS(clean_norm_lc, store_file)
        labels_cont = 'z'
    else:
        return

    if LC_con:
        if flux_con != 1:
            print("Contaminants not identified! Please toggle LC_con=1\n\
                   Continuing program using only the target.")
        else:
            labels_cont = ''
            con_table = ascii.read(con_file[:-5]+
                                   '_individiual'+
                                   con_file[-5:])
            con_table = con_table[con_table["source_id_target"] == t_target["source_id"]]
            XY_arr = find_XY_cont(file_in, con_table, XY_ctr)
            for z in range(len(XY_arr)):
                labels_cont += run_test_for_contaminant(XY_arr[z], file_in, store_file, con_table[z], d_target)

    # make an astropy table out of the array
    # produce plots if required
    if make_plots:
        if LC_con:
            make_LC_plots(file_in, clean_norm_lc, original_norm_lc, d_target, scc, t_target, XY_contam=phot_cont['XY'])
        else:
            make_LC_plots(file_in, clean_norm_lc, original_norm_lc, d_target, scc, t_target)

    return make_datarow(t_target, scc, d_target, labels_cont)


def one_source_cutout(coord, target, store_file, LC_con, flux_con, con_file, make_plots, cutout_size=20):
    '''Download the cutout and run the lightcurve/periodogram analysis for one target.

    Called by the function "iterate_sources".
    parameters
    ----------
    coord : `astropy.astropy.SkyCoord`
        A set of coordinates in the SkyCoord format
    target : `astropy.astropy.Table`
        A row of data from the astropy table
    store_file : str, optional, Default=None
        Name of the file to store logging details.
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
    cutout_size : float, Default=20
        the pixel length of the downloaded cutout
        
    returns
    -------

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
    manifest = Tesscut.download_cutouts(coordinates=coord, size=cutout_size, sector=None)
    for j in range(len(manifest)):
        print(f"{j+1} of {len(manifest)} sectors")
        file_in = manifest["Local Path"][j]
        # rename the fits file to something more useful (target name, sector, camera, ccd)
        name_underscore = str(target['name']).replace(" ", "_")
        f_sp = file_in.split('-')
        file_new = '_'.join([name_underscore, f_sp[1][1:], f_sp[2], f_sp[3][0]])+'.fits'
        os.rename(file_in, file_new)
        with open(store_file, 'a') as file1:
            file1.write(f'{target["source_id"]}, {j+1}/{len(manifest)}\n')
        # run the lightcurve analysis for the given target/fits file
        datarow = single_LC(file_new, target, cutout_size, store_file, LC_con, flux_con, con_file, make_plots)
        final_table.add_row(datarow)

def all_sources_cutout(t_targets, period_file, store_file, LC_con, flux_con, con_file, make_plots):
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
    store_file : str, optional, Default=None
        Name of the file to store logging details.
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
    '''
    if t_targets['log_tot_bg'] is None:
        t_targets.add_column(-999, name='log_tot_bg')
        t_targets.add_column(-999, name='log_max_bg')
        t_targets.add_column(0,    name='num_tot_bg')
    for i, target in enumerate(t_targets):
        print(f"{target['name']}, star # {i+1} of {len(t_targets)}")
        coo = SkyCoord(target["ra"], target["dec"], unit="deg")
        one_source_cutout(coo, target, store_file, LC_con, flux_con, con_file, make_plots)
    final_table.write(period_file)
    print(f"Finished {len(t_targets)} in {(time.time() - start)/60.:.2f} minutes")













#'''TESS SECTORS

#THE FOLLOWING FUNCTIONS ARE TO RUN THE TESSILATOR USING TESS SECTORS'''

def get_fits(sector_num, cc):
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
    list_fits = sorted([f for f in os.listdir(f"sector{sector_num:02d}") if f.endswith('.fits')])
    l_cam = [int(j.split('-')[2]) for j in list_fits]
    l_ccd = [int(j.split('-')[3]) for j in list_fits]
    fits_indices = (l_cam == cc[0]) & (l_ccd == cc[1])
    fits_files = np.array(list_fits)[fits_indices]
    return fits_files


def clean_phot_table(table_phot):
    '''Filters out missing entries from the aperture table and reformats.

    parameters
    ----------
    phot_table : `list`
        A list of table entries generated from running the aperture photometry

    returns
    -------
    An astropy table formatted for further lightcurve analysis.
    '''
    table_phot_filtered = filter_out_none(table_phot)
    table_phot_fixed = np.reshape(np.array(table_phot_filtered, dtype=object),
                                  (-1,14))
    return Table(table_phot_fixed, names=('id', 'xcenter', 'ycenter', 'flux',
                 'flux_err', 'source_id', 'flux_aper', 'bkg',  'flux_corr',
                 'mag', 'mag_err', 'time', 'qual', 'run_no'),
                                   dtype=(str, float, float, float, float, str,
                 float, float, float, float, float, float, int, int))


def make_image_from_sector(sector_num, target, fits_files, im_size=(21,21)):
    '''Makes a 2D cutout object (from astropy.nddata.2Dcutout) of a given
    target from the median time-stacked image

    parameters
    ----------
    sector_num : `int`
        Sector number required
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
    image_index = math.floor((len(target)-1)/2)
    image_fits  = fits_files[image_index]
    fits_slice = f"./sector{sector_num:02d}/{image_fits}"
    table_slice = target[image_index]
    with fits.open(fits_slice) as hdul:
        data = hdul[1].data
        head = hdul[1].header
        error = hdul[2].data
    cutout = Cutout2D(data, (table_slice["xcenter"], table_slice["ycenter"]), im_size)
    return cutout, (table_slice["xcenter"], table_slice["ycenter"])


def iterate_single_cc(cc, t_targets, store_file, sector_num, make_plots, Rad=1.0, SkyRad=np.array([6.0,8.0])):
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

    store_file : `str`
        Name of the file to store logging details.

    sector_num : `int`
        Sector number required

    make_plots : bool
        Decides is plots are made from the lightcurve analysis.
    '''
    fits_files = get_fits(sector_num, cc)
    table_indices = (t_targets['Sector'] == sector_num) & \
                    (t_targets['Camera'] == cc[0]) & \
                    (t_targets['CCD'] == cc[1])
    if table_indices.any() == False:
        return
    table_phot = []
    for f, fits_file in enumerate(fits_files):
        file_in = f"./sector{sector_num:02d}/{fits_file}"
        table_phot.append(aper_run_sectors(file_in, store_file,
                          t_targets[table_indices], f, len(fits_files),
                          Rad, SkyRad))
    table_phot_final = clean_phot_table(table_phot)

    if len(table_phot_final) == 0:
        return
    phot_targets = table_phot_final.group_by('id')
    for target in phot_targets.groups:
        t_target = t_targets[t_targets["source_id"] == target["source_id"][0]]
        clean_norm_lc, original_norm_lc = make_lc(target)
        if len(clean_norm_lc) == 0:
            return
        d_target = run_LS(clean_norm_lc, store_file)
        scc = [sector_num, cc[0], cc[1]]
        if make_plots:
            im_plot, XY_ctr = make_image_from_sector(sector_num, target, fits_files)
            make_LC_plots(im_plot, clean_norm_lc, original_norm_lc, d_target,
                          scc, t_target, XY_ctr=XY_ctr)
        labels_cont = 'z'
        datarow = make_datarow(t_target, scc, d_target, labels_cont)
        return datarow


def iterate_all_cc(t_targets, store_file, sector_num, make_plots, period_file):
    cam_ccd = np.array([[i,j] for i in range(1,5) for j in range(1,5)])
    final_table = create_table_template()
    for cc in cam_ccd:
        start = time.time()
        final_table.add_row(iterate_single_cc(cc, t_targets, store_file, sector_num, make_plots))
        final_table.write(f"{period_file[:-5]}_{cc[0]}_{cc[1]}.ecsv", overwrite=True)
        print(f"Finished {len(final_table)} targets for Sector {sector_num}, "
              f"Camera {cc[0]}, CCD {cc[1]} in {(time.time() - start)/60.:.2f}"
              " minutes")
    print(f"Finished the whole of Sector {sector_num} in "
          f"{(time.time() - start0)/60.:.2f} minutes")