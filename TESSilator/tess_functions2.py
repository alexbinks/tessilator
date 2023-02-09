'''

Alexander Binks & Moritz Guenther, January 2023

Licence: MIT 2023

This module contains all the functions called upon by either the tess_cutouts.py
or tess_large_sectors.py modules. These are:

1)  get_TESS_XY - a function to return the X, Y positions on the Calibrated Full
    Frame Images. This function is needed only for tess_large_sectors.py.
   
2)  contamination - returns three measured quantities to assess the amount of
    incident flux in the TESS aperture that is from contaminating background
    sources. This is calculated by querying the Gaia DR3 catalogue within a 5-pixel
    distance (~105") from the sky position of the target.
   
3a) aper_run_cutouts - returns a numpy array containing the times, magnitudes and
    fluxes from aperture photometry. This is the function used for tess_cutouts.py.

3b) aper_run_sectors - similarly returns a numpy array, but for all stars that are
    located within a given sector/ccd/camera. The resulting array has approximately,
    but no more than n * m rows, where n is the number of stars and m is the number
    of observations in a given sector (some data are excluded from the aperture
    photometry because of quality screening). This function is designed to be used
    by tess_large_sectors.
    
4)  gtd/make_detrends/make_lc - these 3 functions are made to remove spurious data
    points, ensure that only "strings" of contiguous data are being processed, and
    each string is detrended using either a linear (1st) or quadratic (2nd) polynomial
    (chosen by Aikake Inference Criterion) using only data contained in each string.
    Finally, the lightcurve is pieced together and make_lc returns a table containing
    the data ready for periodogram analysis.
    
5)  run_LS - function to conduct Lomb-Scargle periodogram analysis. Returns a table
    with period measurements, plus several data quality flags. If required a plot of
    the lightcurve, periodogram and phase-folded lightcurve is provided.

'''

import sys
import os
import math
import time
import traceback
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning)



# Third party imports
import numpy as np
import pyinputplus as pyip


from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy.io import ascii, fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astroquery.mast import Tesscut
from astropy.nddata import Cutout2D
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Local application imports
from TESSilator.tess_stars2px import tess_stars2px_function_entry


# determine fixed constants (e.g., pixel size)
pixel_size=21.0
'''This is the pixel size for TESS (in arcseconds)

This is held constant. Do not change.
'''


exprf=0.65
'''This is the full-width half maximum of a TESS pixel

This is held constant. Do not change.
'''


Zpt, eZpt = 20.44, 0.05
'''The zero-point TESS magnitude

calculated in Vanderspek et al. 2018. Do not change.
'''

# Read the data from the input file as an astropy table (ascii format)
# Ensure the source_id has a string type (not long integer).

def readascii(inputFile):
    return ascii.read(inputFile, delimiter=',')
    
def tableFromSimbad(simbad_names):
    '''Generate the formatted astropy table from a list of target names

    parameters
    ----------
    simbad_names : `astropy.table.Table'
        a list of target names

    returns
    -------
    gaia_table : `astropy.table.Table'
        the astropy table ready for further analysis
    '''
    # Part 1: Use the SIMBAD database to retrieve the Gaia source identifier
    #         from the target names. 
    simbad_names.rename_column(simbad_names.colnames[0], 'ID')
    simbad_names["ID"] = simbad_names["ID"].astype(str) # set the column header = "ID"
    print(simbad_names) # print the column of targets
    n_arr, is_Gaia = [], [0 for i in simbad_names]
    for i, simbad_name in enumerate(simbad_names["ID"]):
    # if the target name is the numeric part of the Gaia DR3 source identifier
    # prefix the name with "Gaia DR3 "
        if simbad_name.isnumeric() and len(simbad_name) > 10:
            simbad_name = "Gaia DR3 " + simbad_name
            is_Gaia[i] = 1
        n_arr.append(simbad_name)
        # Get a list object identifiers from Simbad
        result_table = [Simbad.query_objectids(i) for i in n_arr]
        NameList = []
        GaiaList = []
        for i, r in enumerate(result_table):
            if r is None: # if no targets were resolved by SIMBAD (could be a typo)
                print(f"Simbad did not resolve {n_arr[i]} - checking Gaia")
                if is_Gaia[i] == 1:
                    NameList.append("Gaia DR3 " + simbad_names["ID"][i])
                    GaiaList.append(simbad_names["ID"][i])
            else:
                r_list = [z for z in r["ID"]]
                m = [s for s in r_list if "Gaia DR3" in s]
                if len(m) == 0: # if the Gaia identifier is not in the Simbad list
                    print(f"There are no corresponding Gaia DR3 identifiers for {n_arr[i]}.")
                    NameList.append("Gaia DR3 " + simbad_names["ID"][i])
                    GaiaList.append(simbad_names["ID"][i])
                else:
                    NameList.append(n_arr[i])
                    GaiaList.append(m[0].split(' ')[2])

    # Part 2: Query the Gaia database using the Gaia source identifiers retrieved in part 1.
    ID_string = ""
    for gi, g in enumerate(GaiaList):
        if gi < len(GaiaList)-1:
            ID_string += g+','
        else:
            ID_string += g
    qry = f"SELECT source_id,ra,dec,parallax,phot_g_mean_mag FROM gaiadr3.gaia_source gs WHERE gs.source_id in ({ID_string});"
    job = Gaia.launch_job_async( qry )
    gaia_table = job.get_results() # Astropy table
    # convert the source_id column to string (astroquery returns the column as np.int64)
    gaia_table["source_id"] = gaia_table["source_id"].astype(str)
    list_ind = []
    # astroquery returns the table sorted numerically by the source identifier
    # the rows are rearranged to match with the input list.
    for row in GaiaList:
        list_ind.append(np.where(np.array(gaia_table["source_id"] == str(row)))[0][0])
    gaia_table = gaia_table[list_ind]
    gaia_table['name'] = NameList
    gaia_table.rename_column('phot_g_mean_mag', 'Gmag')
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def tableFromCoords(coord_table, ang_max=10.0, type_coord='icrs'):
    '''Generate the formatted astropy table from a list of celestial coordinates

    parameters
    ----------
    coord_table : `astropy.table.Table'
        a table consisting of right ascension and declination coordinates (in degrees)

    ang_max : float
        

    returns
    -------
    gaia_table : `astropy.table.Table'
        the astropy table ready for further analysis
    '''
    gaia_table = Table(names=('source_id', 'ra', 'dec', 'parallax', 'Gmag'), dtype=(int,float,float,float,float))
    if type_coord == 'galactic':
        gal = SkyCoord(l=coord_table['col1']*u.degree, b=coord_table['col2']*u.degree, frame='galactic')
        c = gal.transform_to(ICRS)
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    elif type_coord == 'ecliptic':
        ecl = SkyCoord(l=coord_table['col1']*u.degree, b=coord_table['col2']*u.degree, frame='ecliptic')
        c = ecl.transform_to(ICRS)
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    names = ['ra', 'dec'] # set the headers of the 2 columns to "ra" and "dec" 
    coord_table.rename_column(coord_table.colnames[0], 'ra')
    coord_table.rename_column(coord_table.colnames[1], 'dec')
    for i in range(len(coord_table)):
    # Generate an SQL query for each target, where the nearest source is returned within
    # a maximum radius of 10 arcseconds.
        qry = f"SELECT source_id, ra, dec, parallax, phot_g_mean_mag, \
                DISTANCE(\
                POINT({coord_table['ra'][i]}, {coord_table['dec'][i]}),\
                POINT(ra, dec)) AS ang_sep\
                FROM gaiadr3.gaia_source \
                WHERE 1 = CONTAINS(\
                POINT({coord_table['ra'][i]}, {coord_table['dec'][i]}),\
                CIRCLE(ra, dec, {ang_max}/3600.)) \
                ORDER BY ang_sep ASC"
        job = Gaia.launch_job_async( qry )
        x = job.get_results() # Astropy table
        # Fill the empty table with results from astroquery
        print(x, 'len of x ', len(x))
        if len(x) == 0:
            continue
        else:
            y = x[0]['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag']
            gaia_table.add_row((y))
    # For each source, query the identifiers resolved by SIMBAD and return the
    # target with the shortest number of characters (which is more likely to be
    # the most common reference name for the target).
    GDR3_Names = ["Gaia DR3 " + i for i in gaia_table['source_id'].astype(str)]
    result_table =  [Simbad.query_objectids(i) for i in GDR3_Names]
    NameList = []
    for i, r in enumerate(result_table):
        if r is None:
            NameList.append(gaia_table['source_id'][i].astype(str))
        else:
            NameList.append(sorted(r["ID"], key=len)[0])
    gaia_table["name"] = NameList
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag']
    gaia_table = gaia_table[new_order]
    return gaia_table

def tableFromTable(input_table, name_is_source_id):
            # if the Gaia data is supplied by the user (and Gaia source
            # identifiers are used)
    print(input_table)
    gaia_table = Table(data=input_table, dtype=(str, float, float, float, float),
                       names=('source_id', 'ra', 'dec', 'parallax', 'Gmag'))
#    new_cols = ['source_id', 'ra', 'dec', 'parallax', 'Gmag']
#    for n, nc in enumerate(new_cols):
#        gaia_table[nc] = input_table[f'col{str(n+1)}', nc)
    print(gaia_table['source_id'])
    gaia_table['source_id'] = gaia_table['source_id'].astype(int)
#    print(gaia_table['source_id'].data)
    if name_is_source_id:
        gaia_table['name'] = gaia_table['source_id'].data
#        print(gaia_table["name"])
    else:
        GDR3_Names = ["Gaia DR3 " + i for i in gaia_table['source_id'].astype(str)]
        NameList = []
        result_table =  [Simbad.query_objectids(i) for i in GDR3_Names]
        for i, r in enumerate(result_table):
            if r is None:
                NameList.append(gaia_table['source_id'][i].astype(str))
            else:
                NameList.append(sorted(r["ID"], key=len)[0])
        gaia_table["name"] = NameList
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def getGaiaData(gaia_table, name_is_source_id=0):

    '''
    Reads the input table and returns Gaia data.
    The returned table has the columns:
        name --> the preferred choice of source identifier
        source_id --> the Gaia DR3 source identifier
        ra --> right ascension in the ICRS (J2000) system
        dec --> declination in the ICRS (J2000) system
        parallax --> parallax from Gaia DR3 (in mas)
        Gmag --> the apparent G-band magnitude from Gaia DR3
        
    THE USER MUST SUPPLLY A TABLE WITH COMMA-SEPARATED VARIABLES!
    This table can be formatted in either of these 3 ways:
    1) A table with a single column containing the source identifier
       *note that this is the preferred method since the target identified
       in the Gaia query is unambiguously the same as the input value.
    2) A table with RA and DEC columns in ICRS co-ordinates.
       *note this is slower because of the time required to run the Vizier
       query.
    3) A table with all 5 columns already made.
       *note in this case, only the column headers are checked, everything
       else is passed.
    '''

    print(gaia_table)
    if len(gaia_table.colnames) == 1:
        tbl = tableFromSimbad(gaia_table)
    elif len(gaia_table.colnames) == 2:
        tbl = tableFromCoords(gaia_table)
    elif len(gaia_table.colnames) == 5:
        tbl = tableFromTable(gaia_table, name_is_source_id)
    else:
        raise Exception('Input table has invalid format. Please use one of the following formats: \n [1] source_id \n [2] ra and dec \n [3] source_id, ra, dec, parallax and Gmag')
    return tbl

#select len(Gaiai_Input.colnames):
#    match 1:
#        tlb = from Simbas(...)
#    match 2:
#        tbl = from Coords(...)
#    _:
#        raise Exceptin()










def getTESSPixelXY(t_targets):
    '''
    For a given pair of celestial sky coordinates, this function returns table rows containing the sector,
    camera, CCD, and X/Y position of the full-frame image fits file, so that all stars located in a given
    (large) fits file can be processed simultaneously. This function is only used when the
    "tess_large_sectors.py" method is called. After the table is returned, the input table is joined to the
    input table on the source_id, to ensure this function only needs to be called once. The time to process
    approximately 500k targets is ~4 hours.
    '''
    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
           outColPix, outRowPix, scinfo = tess_stars2px_function_entry(
           t_targets['source_id'], t_targets['ra'], t_targets['dec'])
    return Table([outID, outSec, outCam, outCcd, outColPix, outRowPix], names=('source_id', 'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos'))



def runSQLQueryContaminants(t_target, pix_radius=5.0):
    # Generate an SQL query for each target.
    query = f"SELECT source_id, ra, dec, phot_g_mean_mag,\
    DISTANCE(\
    POINT({t_target['ra']}, {t_target['dec']}),\
    POINT(ra, dec)) AS ang_sep\
    FROM gaiadr3.gaia_source\
    WHERE 1 = CONTAINS(\
    POINT({t_target['ra']}, {t_target['dec']}),\
    CIRCLE(ra, dec, {pix_radius*pixel_size/3600.})) \
    AND phot_g_mean_mag < {t_target['Gmag']+3.0} \
    ORDER BY phot_g_mean_mag ASC"

    # Attempt a synchronous SQL job, otherwise try the asyncronous method.
    try:
        job = Gaia.launch_job(query)
    except Exception:
        traceback.print_exc(file=log)
        job = Gaia.launch_job_async(query)
    return job.get_results()  


def flux_fraction_contaminant(ang_sep, s, d_th=0.000005):
    '''calculate the fraction of flux incident in the aperture from every neighbouring contaminating source. The equation is a converging sum (i.e., infinite indices) so a threshold is made that if the nth contribution to the sum is less than this, the loop breaks.
    '''
    n, n_z = 0, 0
    t = (ang_sep/pixel_size)**2/(2.0*exprf**(2)) # measured in pixels
    while True:
        sk = np.sum([(s**(k)/np.math.factorial(k)) for k in range(0,n+1)])
        sx = 1.0 - (np.exp(-s)*sk)
        n_0 = ((t**n)/np.math.factorial(n))*sx
        n_z += n_0
        if np.abs(n_0) > d_th:
            n = n+1
        if np.abs(n_0) < d_th:
            break
    return n_z*np.exp(-t)




def contamination(t_targets, LC_con, Rad=1.0):
    '''
    The purpose of this function is to estimate the amount of flux incident in the TESS aperture that originates
    from neighbouring, contaminating sources. Given that the passbands from TESS (T-band, 600-1000nm) are similar
    to Gaia G magnitude, and that Gaia is sensitive to G~21, the Gaia DR3 catalogue is used to quantify contamination.
    
    For each target in the input file, an SQL query of the Gaia DR3 catalogue is performed for all neighbouring
    sources that are within 5 pixels (5*21=105 arcseconds) of the sky position, and are brighter than 3 magnitudes
    fainter than the target (in Gaia G-band).
    
    The Rayleigh formula is used to calculate the fraction of flux incident in the aperture from the target, and an
    analytical formula (Biser & Millman 1965, https://books.google.co.uk/books?id=5XBGAAAAYAAJ -- equation 3b-10) is
    used to calculate the contribution of flux from all neighbouring sources.
    
    The output from the function gives two table. First table contains 3 columns:
    
    1) the base-10 logarithmic flux ratio between the total background (all neighbouring sources) and the target.
    2) the base-10 logarithmic flux ratio between the most contaminating source and the target.
    3) the number of background sources used in the calculation.
    
    The second table contains the top 5 contaminating sources in terms of their flux contribution to the aperture,
    from largest contaminant to 5th largest. These are used if the user wants to measure lightcurves and periodograms
    of potential nearby contaminants to flag spurious data.
    '''

    con1, con2, con3 = [], [], []
    # Create an empty table to fill with results from the contamination analysis.
    t_contam = Table(names=['source_id', 'log_tot_bg', 'log_max_bg', 'num_tot_bg'],
                  dtype=(str, float, float, int))

    con_table_full = Table(names=('source_id_target', 'source_id_cont', 'RA', 'DEC', 'Gmag', 'd_as', 'log_flux_frac'), dtype=(str, str, float, float, float, float, float))
    for i in range(len(t_targets)):
        r = runSQLQueryContaminants(t_targets[i])
        r["ang_sep"] = r["ang_sep"]*3600. # convert the angular separation from degrees to arcseconds
        if len(r) > 1:
            rx = Table(r[r["source_id"].astype(str) != t_targets["source_id"][i].astype(str)])

            # calculate the fraction of flux from the source object that falls into the aperture
            # using the Rayleigh formula P(x) = 1 - exp(-[R^2]/[2*sig^2])
            s = Rad**(2)/(2.0*exprf**(2)) # measured in pixels
            fg_star = (1.0-np.exp(-s))*10**(-0.4*t_targets["Gmag"][i])


            fg_cont = []                 
            # calculate the fractional flux incident in the aperture from
            # each contaminant.       
            for G_cont, ang_sep in zip(rx["phot_g_mean_mag"], rx["ang_sep"]):
                f_frac = flux_fraction_contaminant(ang_sep, s)
                fg_cont.append(f_frac*10**(-0.4*G_cont))


            if LC_con:
                rx['log_flux_frac'] = np.log10(fg_cont/fg_star)
                rx['source_id_target'] = t_targets["source_id"][i]
                new_order = ['source_id_target', 'source_id', 'ra', 'dec', 'phot_g_mean_mag', 'ang_sep', 'log_flux_frac']
                rx.sort(['log_flux_frac'], reverse=True)
                rx = rx[new_order]
                rx['source_id_target'] = rx['source_id_target'].astype(str)
                rx['source_id'] = rx['source_id'].astype(str)
                # get the five highest flux contributors and save them to the con_table_full file
                for rx_row in rx[0:5][:]:
                    con_table_full.add_row(rx_row)
                    
                    
            # if the sum is very small, make the results = -999
#            if np.sum(fg_cont) < 10**(-8):
#                con1.append(-999)
#                con2.append(-999)
            # otherwise sum up the fluxes from each neighbour and divide by the target flux.
#            else:
            con1.append(np.log10(np.sum(fg_cont)/fg_star))
            con2.append(np.log10(max(fg_cont)/fg_star))
            con3.append(len(fg_cont))
        else:
            con1.append(-999)
            con2.append(-999)
            con3.append(0)
        # store each entry to file
#        with open(store_file, 'a') as file1:
#            file1.write("Gmag = "+str(t_targets["Gmag"][i])+', log_tot_bg = '+
#                        str(con1[i])+', log_max_bg = '+
#                        str(con2[i])+', num_tot_bg = '+
#                        str(con3[i])+', '+
#                        str(i+1)+'/'+str(len(t_targets["Gmag"]))+
#                        '\n')
        # add the entry to the input target table
        t_contam.add_row([str(t_targets["source_id"][i]), con1[i], con2[i], con3[i]])
    t_targets["log_tot_bg"] = con1
    t_targets["log_max_bg"] = con2
    t_targets["num_tot_bg"] = con3
    if LC_con == False:
        con_table_full = None
    return t_targets, t_contam, con_table_full
    
    


def find_XY_cont(f_file, con_table, XY_ctr):
    '''
    If the user requests a periodogram analysis of neighbouring potential contaminants,
    this function returns the pixel X-Y centroid of the contaminants on the TESS cutout.
    '''
    with fits.open(f_file) as hdul:
        head = hdul[0].header
        RA_ctr, DEC_ctr = head["RA_OBJ"], head["DEC_OBJ"]
        RA_con, DEC_con = con_table["RA"], con_table["DEC"]
        X_abs_con, Y_abs_con = [], []
        _, _, _, _, _, _, Col_ctr, Row_ctr, _ = tess_stars2px_function_entry(1, RA_ctr, DEC_ctr, trySector=head["SECTOR"])
        for i in range(len(RA_con)):
            _, _, _, _, _, _, Col_con, Row_con, _ = tess_stars2px_function_entry(1, RA_con[i], DEC_con[i], trySector=head["SECTOR"])
            X_abs_con.append(Col_con)
            Y_abs_con.append(Row_con)
        X_con = np.array(X_abs_con - Col_ctr).flatten() + XY_ctr[0]
        Y_con = np.array(Y_abs_con - Row_ctr).flatten() + XY_ctr[1]
        return np.array([X_con, Y_con]).T


def aper_run_cutouts(f_file, XY_pos=(10.,10.), store_file=None, Rad=1, SkyRad=(6,8), Zpt=20.44, eZpt=0.05, make_slices=0):
    '''
    This function reads in each "postage-stamp" fits file of stacked data from TESScut, and
    performs aperture photometry on each image slice across the entire sector. This returns
    a table of results from the aperture photometry at each time step, which forms the raw
    lightcurve to be processed in subsequent functions.
    '''
    # first need to test that the fits file can be read and isn't corrupted
    try:
        with fits.open(f_file) as hdul:
            with open(store_file, "a") as log:
                # if the file can be opened, check the data structures are correctly recorded
                try:
                    data = hdul[1].data
                    head = hdul[0].header
                    error = hdul[2].data
                except Exception:
                    traceback.print_exc(file=log)
                    return
            f_out = []
            if data.size == int(XY_pos[0]*2):
                file_iter = len(data.shape)
            else:
                file_iter = data.shape[0]
            for i in range(file_iter): # make a for loop over each 2D image in the fits stack
                # ensure the quality flag returned by TESS is 0 (no quality issues) across all
                # X,Y pixels in the image slice - otherwise move to next image.
                if data["QUALITY"][:][:][i] == 0:
                    if data.shape[0] == int(XY_pos[0]*2):
                        flux_vals = data["FLUX"][:][:].reshape((int(XY_pos[0]*2), int(XY_pos[1]*2)))
                    else:
                        flux_vals = data["FLUX"][:][:][i]
                    if (make_slices == 1) & (XY_pos[0] == flux_vals.shape[0]/2.) & ("slice" not in f_file) & (i in np.linspace(0, file_iter-1, 10).astype(int)):
                        primary_hdu = fits.PrimaryHDU(header=head)
                        col1 = fits.Column(name='QUALITY', format='J', array=np.array([data["QUALITY"][:][:][i]], dtype=np.object_))
                        col2 = fits.Column(name='FLUX', format=f'{int(XY_pos[0]*2)}E', array=data["FLUX"][:][:][i])
                        col3 = fits.Column(name='TIME', format='E', array=np.array([data["TIME"][:][:][i]], dtype=np.object_))
                        data_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
                        error_hdu = fits.ImageHDU(data=error)
                        hdul = fits.HDUList([primary_hdu, data_hdu, error_hdu])
                        hdul.writeto(f"{f_file[:-5]}_slice{i:04d}.fits", overwrite=True)
                    aperture = CircularAperture(XY_pos, Rad) #define a circular aperture around all objects
                    annulus_aperture = CircularAnnulus(XY_pos, SkyRad[0], SkyRad[1]) #select a background annulus
                    aperstats = ApertureStats(flux_vals, annulus_aperture) #get the image statistics for the background annulus
                    phot_table = aperture_photometry(flux_vals, aperture, error=error) #obtain the raw (source+background) flux
                    aperture_area = aperture.area_overlap(flux_vals) #calculate the background contribution to the aperture
                    #print out the data to "phot_table"
                    phot_table['id'] = i
                    phot_table['bkg'] = aperstats.median # select the mode (3median - 2mean) to represent the background (per pixel)
                    phot_table['total_bkg'] = phot_table['bkg'] * aperture_area
                    phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['total_bkg']
                    phot_table['mag'] = -2.5*np.log10(phot_table['aperture_sum_bkgsub'])+Zpt
                    phot_table['mag_err'] = np.abs((-2.5/np.log(10))*phot_table['aperture_sum_err']/phot_table['aperture_sum'])
                    if data.shape[0] == int(XY_pos[0]*2):
                        phot_table['time'] = data["TIME"][:][:][0]
                        phot_table['qual'] = data["QUALITY"][:][:][0]
                    else:
                        phot_table['time'] = data["TIME"][:][:][i]
                        phot_table['qual'] = data["QUALITY"][:][:][i]
                    for col in phot_table.colnames:
                        phot_table[col].info.format = '%.6f'
                    for j in range(len(phot_table)):
                        f_out.append(phot_table.as_array()[j])
            if (XY_pos[0] == flux_vals.shape[0]/2.) & (XY_pos[1] == flux_vals.shape[1]/2.) & ("slice" not in f_file):
                ascii.write(np.array(f_out), f_file[:-5] + '_phot_out.tab', overwrite=True)
            return f_out
    except OSError as e:
        return




    
def aper_run_sectors(f_file, store_file, objects, zr, lenzr, Rad=1.0, SkyRad=np.array([6.0,8.0])):
    try:
        with fits.open(f_file) as hdul:
            with open(store_file, "a") as log:
                try:
                    data = hdul[1].data
                    head = hdul[1].header
                    error = hdul[2].data
                except Exception:
                    traceback.print_exc(file=log)
                    return
            if (head["DQUALITY"] != 0) or (head["NAXIS"] != 2):
                return
            print(objects["Sector"][0], objects["Camera"][0], objects["CCD"][0], zr+1, lenzr)
            with open(store_file, 'a') as file1:
                file1.write(str(f_file)+', '+
                            str(zr+1)+'/'+str(lenzr)+
                            '\n')
            w = WCS(head)
            c = SkyCoord(objects['ra'], objects['dec'], unit=u.deg, frame='icrs')
            with open(store_file, "a") as log:
                try:
#                    x_obj, y_obj = w.world_to_pixel(c)
                    y_obj, x_obj = w.world_to_array_index(c)
                except Exception:
                    traceback.print_exc(file=log)
                    return
            positions = tuple(zip(x_obj, y_obj))
#            positions = tuple(zip(objects["Xpos"].data, objects["Ypos"].data))
      #define a circular aperture around all objects
            aperture = CircularAperture(positions, Rad)
      #select a background annulus
            annulus_aperture = CircularAnnulus(positions, SkyRad[0], SkyRad[1])
      #fit the background using the median flux in the annulus
            aperstats = ApertureStats(data, annulus_aperture)
      #obtain the raw (source+background) flux
            phot_table = aperture_photometry(data, aperture, error=error)
            aperture_area = aperture.area_overlap(data)
      #print out the data to "phot_table"
            phot_table['source_id'] = objects['source_id']
            phot_table['bkg'] = aperstats.median
            phot_table['total_bkg'] = phot_table['bkg'] * aperture_area
            phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['total_bkg']
            phot_table['mag'] = -2.5*np.log10(phot_table['aperture_sum_bkgsub'])+Zpt
            phot_table['mag_err'] = np.abs((-2.5/np.log(10))*phot_table['aperture_sum_err']/phot_table['aperture_sum'])
            phot_table['time'] = (head['TSTART'] + head['TSTOP'])/2.
            phot_table['qual'] = head['DQUALITY']
            g_match = (phot_table["total_bkg"] > 0) & (phot_table["aperture_sum_bkgsub"] > 0)
            phot_table['qual'][~g_match] = 999
            phot_table['run_no'] = zr
            for col in phot_table.colnames[:5]:
                phot_table[col].info.format = '%.6f'
            for col in phot_table.colnames[6:-2]:
                phot_table[col].info.format = '%.6f'
            phot_table['source_id'].info.format = '%s'
            phot_table['qual'].info.format = '%i'
            phot_table['run_no'].info.format = '%i'
            f_out = []
            for i in range(len(phot_table)):
                f_out.append(list(phot_table.as_array()[i]))
            return f_out
    except OSError as e:
        return




def make_phot_table(tab):
    if tab.size == 1:
        return
    data = np.lib.recfunctions.structured_to_unstructured(np.array(tab))
    has_nan = np.any(np.isnan(data), axis=1)
    tab_no_nan = tab[~has_nan]
    tab_astro = Table(tab_no_nan, names=('num', 'xcenter', 'ycenter', 'flux', 'flux_err',
                                  'bkg', 'total_bkg', 'flux_corr', 'mag',
                                  'mag_err', 'time', 'qual'),
                                  dtype=(int, float, float, float, float, float, float, float,
                                  float, float, float, int))
    tab_fin = tab_astro[tab_astro["flux"] > 10.]
    return tab_fin



def make_image_from_sector(fits_slice, table_slice):
    with fits.open(fits_slice) as hdul:
        data = hdul[1].data
        head = hdul[1].header
        error = hdul[2].data
    cutout = Cutout2D(data, (table_slice["xcenter"], table_slice["ycenter"]), (21,21))
    return cutout






def gtd(t, f, MAD_fac=2., time_fac=10., min_num_per_group=50):
    '''
    This function is used to remove data points from the lightcurve
    that are likely to be spurious. When data is being downloaded
    from the satellite there is typically a 1 or 2 day gap in each
    TESS sector. To avoid systematic offsets and ensure the data is
    efficiently normalized, the lightcurve is split into sections of
    data such that neighbouring data points must have been observed
    within 10x the median absolute deviation of the time difference
    between each observation.
    
    The start and end point of each data section must have a flux
    value within 3 MAD of the median flux in the sector. This is done
    because often after large time gaps the temperature of the sensors
    changes, and including these components is likely to just result
    in a signal from instrumental noise.
    
    The function returns the start and end points for each data section
    in the sector, which must contain at least 50 data points. This is
    to ensure there are enough datapoints to construct a periodogram
    analysis.
    '''
    
    tm, fm = np.median(t), np.median(f)
    f_MAD  = median_abs_deviation(f, scale='normal')
    td     = np.zeros(len(t))
    td[1:] = np.diff(t)

    A0 = (np.abs(f-fm) <= MAD_fac*f_MAD).astype(int)
    A1 = (td <= time_fac*np.median(td)).astype(int)
    B  = (A0+A1).astype(int)
    gs, gf = [], []
    i = 0    
    l = 0
    while i < len(f)-1:
        if B[i] == 2:
            gs.append(i)
            j = i+1
            while (A1[j]) == 1 and (j < len(f)-1):
                j = j+1
            if j == len(f)-1:
                l = l+1
                break
            else:
                k = j
                j = j-1
                while A0[j] == 0:
                    j = j-1
                gf.append(j)
                i = k + 1
        else:
            i = i+1
    if l == 1:
        k = len(f)-1
        while B[k] != 2:
            k = k-1
        gf.append(k)
    
    gs, gf = np.array(gs), np.array(gf)
    return gs[(gf-gs)>min_num_per_group], gf[(gf-gs)>min_num_per_group]



def make_detrends(ds,df,t,f,err):
    '''
    This function operates on each section of data returned from
    the "gtd" function and performs a detrending routine so that
    data from separate sections can be connected. Five lists are
    outputted which form the normalized, detrended lightcurve.
    
    The choice of polynomial fit for the detrending function is
    linear or quadratic, and depends on which component best
    satisfies the Aikake Information Criterion. 
    '''
    t_fin, f_no, f_fin, e_fin, s_fin = [], [], [], [], []
    for i in range(len(ds)):
        p1, r1, _,_,_ = np.polyfit(t[ds[i]:df[i]+1], f[ds[i]:df[i]+1], 1, full=True)
        p2, r2, _,_,_ = np.polyfit(t[ds[i]:df[i]+1], f[ds[i]:df[i]+1], 2, full=True)
        chi1 = np.sum((np.polyval(p1, t[ds[i]:df[i]+1])-f[ds[i]:df[i]+1])**2/(err[ds[i]:df[i]+1])**2)
        chi2 = np.sum((np.polyval(p2, t[ds[i]:df[i]+1])-f[ds[i]:df[i]+1])**2/(err[ds[i]:df[i]+1])**2)
        AIC1, AIC2 = 2.*(2. - np.log(chi1/2.)), 2.*(3. - np.log(chi2/2.))
        if AIC1 < AIC2:
            fn = f[ds[i]:df[i]+1]/np.polyval(p2, t[ds[i]:df[i]+1])
            s_fit = 1
        else:
            fn = f[ds[i]:df[i]+1]/np.polyval(p2, t[ds[i]:df[i]+1])
            s_fit = 2
        t_fin.append(t[ds[i]:df[i]+1])
        f_no.append(f[ds[i]:df[i]+1])
        f_fin.append(fn)
        e_fin.append(err[ds[i]:df[i]+1])
        s_fin.append(s_fit)
    return t_fin, f_no, f_fin, e_fin, s_fin


def make_lc(phot_table):
    print(phot_table)
    g = np.abs(phot_table["flux"][:] - np.median(phot_table["flux"][:])) < 20.0*median_abs_deviation(phot_table["flux"][:], scale='normal')
    time = np.array(phot_table["time"][:][g])
    mag = np.array(phot_table["mag"][:][g])
    flux = np.array(phot_table["flux_corr"][:][g])
    eflux = np.array(phot_table["flux_err"][:][g])
    ds, df = gtd(time, flux)
    if (len(ds) == 0) or (len(df) == 0):
        return [], []
# 1st: normalise the flux by dividing by the median value
    nflux  = flux/np.median(flux)

    neflux = eflux/flux
# 2nd: detrend each lightcurve sections by either a straight-line fit or a parabola. The choice is selected using AIC.
    t_detrend, f_orig, f_detrend, e_detrend, s_detrend = make_detrends(ds, df, time, nflux, neflux)

    original_data = dict()
    original_data["time"] = np.array(time)
    original_data["nflux"] = np.array(nflux)
    original_data["mag"] = np.array(mag)
    if len(t_detrend) > 0:
        if len(t_detrend[0][:]) > 50:
            clean_data = Table()
            clean_data['time']  = np.concatenate(t_detrend)
            clean_data['time0'] = np.concatenate(t_detrend)-time[0]
            clean_data['oflux'] = np.concatenate(f_orig)
            clean_data['nflux'] = np.concatenate(f_detrend)
            clean_data['enflux'] = np.concatenate(e_detrend)
            return clean_data, original_data
        else:
            return [], []
    else:
        return [], []


def get_second_peak(power):
### Get the left side of the peak
    p_m = np.argmax(power)
    x = p_m
    while (power[x-1] < power[x]) and (x > 0):
        x = x-1
    p_l = x
    p_lx = 0
    while (power[p_l] > 0.85*power[p_m]) and (p_l > 1):
        p_lx = 1
        p_l = p_l - 1
    if p_lx == 1:
        while (power[p_l] > power[p_l-1]) and (p_l > 0):
            p_l = p_l - 1
    if p_l < 0:
        p_l = 0

### Get the right side of the peak
    x = p_m
    if x < len(power)-1:
        while (power[x+1] < power[x]) and (x < len(power)-2):
            x = x+1
        p_r = x
        p_rx = 0
        while (power[p_r] > 0.85*power[p_m]) and (p_r < len(power)-2):
            p_rx = 1
            p_r = p_r + 1
        if p_rx == 1:
           while (power[p_r] > power[p_r+1]) and (p_r < len(power)-2):
                p_r = p_r + 1
        if p_r > len(power)-1:
            p_r = len(power)-1
        a = np.arange(len(power))
        a_g = a[p_l:p_r+1]
        a_o = a[np.setdiff1d(np.arange(a.shape[0]), a_g)] 
    return a_g, a_o


def filter_out_none(files):
    '''
    Simple function to remove list items if they are empty.
    >>> filter_out_none([1, 4, None, 6, 9])
    [1, 4, 6, 9]
    '''
    return list(filter(lambda item: item is not None, files))


def gauss_fit(x, a0, x_mean, sigma):
    '''
    Returns the best-fit Gaussian parameters: amplitude (A),
    mean (x_mean) and uncertainty (sigma) for a distribution
    of x values
    '''
    return a0*np.exp(-(x-x_mean)**2/(2*sigma**2))

def sine_fit(x, y0, A, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase)
    to a regular sinusoidal function.
    '''
    return y0 + A*np.sin(2.*np.pi*x + phi)


def run_LS(clean, store_file, p_min_thresh=0.1, p_max_thresh=50., samples_per_peak=10):
    '''
    Runs a Lomb-Scargle periodogram on the cleaned lightcurve
    and returns a dictionary of results.
    '''
    LS_dict = dict()
    med_f, MAD_f = np.median(clean["nflux"]), median_abs_deviation(clean["nflux"], scale='normal')
    ls = LombScargle(clean["time0"], clean["nflux"])
    frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                    maximum_frequency=1./p_min_thresh,
                                    samples_per_peak=samples_per_peak)
    FAP = ls.false_alarm_probability(power.max())
    probabilities = [0.1, 0.05, 0.01]
    FAP_test = ls.false_alarm_level(probabilities)
    p_m = np.argmax(power)
    y_fit = ls.model(clean["time0"], frequency[p_m])
    period_best = 1.0/frequency[p_m]
    power_best = power[p_m]

    period = 1./frequency[::-1]
    power = power[::-1]
    
    a_g, a_o = get_second_peak(power) # a_g returns the datapoints that form the Gaussian
                                           # around the highest power
                                           # a_o returns the array for all other values 
    with open(store_file, "a") as log:
        try:
            popt, _ = curve_fit(gauss_fit, period[a_g], power[a_g], bounds=(0, [1., p_max_thresh, p_max_thresh]))
        except Exception:
            traceback.print_exc(file=log)
            popt = np.array([1, p_max_thresh/2, p_max_thresh/2])
            pass

    ym = gauss_fit(period[a_g], *popt)

    per_a_o, power_a_o = period[a_o], power[a_o]
    per_2 = per_a_o[np.argmax(power[a_o])]
    pow_2 = power_a_o[np.argmax(power[a_o])]
    pow_pow2 = 1.0*power_best/pow_2
    phase, cycle_num = np.modf((clean["time0"]-min(clean["time0"]))/period_best)
    phase, cycle_num = np.array(phase), np.array(cycle_num)
    ind4fit = np.argsort(phase)
    ind4plt = np.argsort((clean["time0"]-min(clean["time0"]))/period_best)
    phase_fit, nflux_fit, cycle_num_fit = phase[ind4fit], clean["nflux"][ind4fit], cycle_num[ind4fit].astype(int)
    phase_plt, nflux_plt, cycle_num_plt = phase[ind4plt], clean["nflux"][ind4plt], cycle_num[ind4plt].astype(int)

    with open(store_file, "a") as log:
        try:
#            pops, popsc = curve_fit(sine_fit, phase, clean["nflux"].data, bounds=(0, [2., 2., 2.*np.pi]))
            pops, popsc = curve_fit(sine_fit, phase_fit, nflux_fit, bounds=(0, [2., 2., 2.*np.pi]))
        except Exception:
            traceback.print_exc(file=log)
            pops = np.array([1., 0.001, 0.5])
            pass
            
    Ndata = len(clean)
    yp = sine_fit(phase_fit, *pops)
    phase_scatter = median_abs_deviation(yp - clean["nflux"], scale='normal')
    fdev = 1.*np.sum(np.abs(clean["nflux"] - yp) > 3.0*phase_scatter)/Ndata

    LS_dict['median_MAD_nLC'] = [med_f, MAD_f]
    LS_dict['period'] = period
    LS_dict['power'] = power
    LS_dict['period_best'] = period_best
    LS_dict['power_best'] = power_best 
    LS_dict['y_fit_LS'] = y_fit 
    LS_dict['FAPs'] = FAP_test
    LS_dict['Gauss_fit_peak_parameters'] = popt
    LS_dict['Gauss_fit_peak_y_values'] = ym
    LS_dict['period_around_peak'] = period[a_g]
    LS_dict['power_around_peak'] = power[a_g]
    LS_dict['period_not_peak'] = period[a_o] 
    LS_dict['power_not_peak'] = power[a_o] 
    LS_dict['period_second'] = per_2
    LS_dict['power_second'] = pow_2
    LS_dict['phase_fit_x'] = phase_fit
    LS_dict['phase_fit_y'] = yp
    LS_dict['phase_x'] = phase_plt
    LS_dict['phase_y'] = nflux_plt
    LS_dict['phase_col'] = cycle_num_plt
    LS_dict['pops_vals'] = pops    
    LS_dict['pops_cov'] = popsc
    LS_dict['phase_scatter'] = phase_scatter
    LS_dict['frac_phase_outliers'] = fdev
    LS_dict['Ndata'] = Ndata
    return LS_dict
    
    
def isPeriodCont(d_target, d_cont, t_cont, frac_amp_cont=0.5):
    '''
    If the user selects to measure periods for the neighbouring contaminants
    this function returns a flag to assess if a contaminant may actually be
    the source causing the observed periodicity.
    >>> d_target = {"period_best": 4.2,
                    "Gauss_fit_peak_parameters": [0.8,4.15,0.5],
                    "pops_vals": [0.1,0.1,0.1]}
    >>> d_cont = {"period_best": 4.2,
                  "Gauss_fit_peak_parameters": [0.8,4.15,0.5],
                  "pops_vals": [0.1,0.1,0.1]}
    >>> t_cont = {"flux_cont": 0.01}
    >>> isPeriodCont(d_target, d_cont, t_cont)
    >>> 'a'
    '''
    if abs(d_target["period_best"] - d_cont["period_best"] <
          (d_target["Gauss_fit_peak_parameters"][2] + d_cont["Gauss_fit_peak_parameters"][2])):
        if d_cont["pops_vals"][1]/d_target["pops_vals"][1] > (frac_amp_cont*10**(t_cont["log_flux_frac"]))**(-1):
            return 'a'
        else:
            return 'b'
    else:
        return 'c'



def make_LC_plots(f_file, clean, orig, LS_dict, scc, t_table, XY_ctr=(10,10), XY_contam=None, p_min_thresh=0.1, p_max_thresh=50., Rad=1.0, SkyRad = np.array([6.0,8.0])):
    mpl.rcParams.update({'font.size': 14})
    fsize = 22.
    lsize = 0.9*fsize
    fig, axs = plt.subplots(2,2, figsize=(20,15))

    axs[0,0].set_position([0.05,0.55,0.40,0.40])
    axs[0,1].set_position([0.55,0.55,0.40,0.40])
    axs[1,0].set_position([0.05,0.3,0.90,0.2])
    axs[1,1].set_position([0.05,0.05,0.90,0.2])


    t_orig0 = orig["time"]-orig["time"][0]
    if type(f_file) == str:
        with fits.open(f_file) as hdul:
            data = hdul[1].data
            flux_vals = np.log10(data["FLUX"][:][:][int(data.shape[0]/2)])
            circ_aper = Circle((flux_vals.shape[0]/2., flux_vals.shape[1]/2.),
                                Rad, linewidth=1.2, fill=False, color='r')
            circ_ann1 = Circle((flux_vals.shape[0]/2., flux_vals.shape[1]/2.),SkyRad[0], linewidth=1.2, fill=False, color='b')
            circ_ann2 = Circle((flux_vals.shape[0]/2., flux_vals.shape[1]/2.),SkyRad[1], linewidth=1.2, fill=False, color='b')
        fig.text(0.5,0.96, f"{t_table['name'][0]}, Sector {str(scc[0])}, Camera {str(scc[1])}, CCD {str(scc[2])}", fontsize=lsize*2.0, horizontalalignment='center')
        f = axs[0,0].imshow(flux_vals, cmap='binary')
        name_underscore = t_table['name'].replace(" ", "_")
    else:
        flux_vals = np.log10(f_file.data)
        XY_ctr = (f_file.xmin_original + f_file.center_cutout[0], f_file.ymin_original + f_file.center_cutout[1])
        circ_aper = Circle(XY_ctr, Rad, linewidth=1.2, fill=False, color='r')
        circ_ann1 = Circle(XY_ctr, SkyRad[0], linewidth=1.2, fill=False, color='b')
        circ_ann2 = Circle(XY_ctr, SkyRad[1], linewidth=1.2, fill=False, color='b')
        fig.text(0.5,0.96, f"Gaia DR3 {t_table['source_id'][0]}, Sector {str(scc[0])}, Camera {str(scc[1])}, CCD {str(scc[2])}", fontsize=lsize*2.0, horizontalalignment='center')
        axs[0,0].set_xlim(f_file.xmin_original, f_file.xmax_original)
        axs[0,0].set_ylim(f_file.ymin_original, f_file.ymax_original)
        f = axs[0,0].imshow(flux_vals, cmap='binary', extent=[f_file.xmin_original, f_file.xmax_original, f_file.ymin_original, f_file.ymax_original])
        name_underscore = t_table['source_id'][0]


    axs[0,0].set_xlabel("X pixel", fontsize=fsize)
    axs[0,0].set_ylabel("Y pixel", fontsize=fsize)
    axs[0,0].add_patch(circ_aper)
    axs[0,0].add_patch(circ_ann1)
    axs[0,0].add_patch(circ_ann2)
    if XY_contam is not None:
        axs[0,0].scatter(XY_contam[:, 0], XY_contam[:, 1], marker='X', s=150, color='y')
    divider = make_axes_locatable(axs[0,0])
    cax = divider.new_horizontal(size='5%', pad=0.4)
    fig.add_axes(cax)
    cbar = fig.colorbar(f, cax=cax)
    cbar.set_label('log$_{10}$ counts (e$^-$/s)', rotation=270, labelpad=+15)


    axs[0,1].set_xlim([p_min_thresh, p_max_thresh])
    axs[0,1].grid(True)
    axs[0,1].set_xlabel("Period (days)", fontsize=fsize)
    axs[0,1].set_ylabel("Power", fontsize=fsize)
    axs[0,1].semilogx(LS_dict['period'], LS_dict['power'])
    [axs[0,1].axhline(y=i, linestyle='--', color='grey', alpha=0.8) for i in LS_dict['FAPs']]
    axs[0,1].text(0.99,0.94, "$P_{\\rm rot}^{\\rm (max)}$ = " + f"{LS_dict['period_best']:.3f} days, power = {LS_dict['power_best']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.82, "$P_{\\rm rot}^{\\rm (2nd)}$ = " + f"{LS_dict['period_second']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.76, f"power ratio = {LS_dict['power_best']/LS_dict['power_second']:.3f}",
                  fontsize=lsize,horizontalalignment='right', 
                  transform=axs[0,1].transAxes)

    if LS_dict['Gauss_fit_peak_parameters'][1] != 15:
        axs[0,1].plot(LS_dict['period_around_peak'], LS_dict['Gauss_fit_peak_y_values'], c='r', label='Best fit')
        axs[0,1].text(0.99,0.88, "$P_{\\rm rot}^{\\rm (Gauss)}$ = " + f"{LS_dict['Gauss_fit_peak_parameters'][1]:.3f}" + "$\\pm$" + f"{LS_dict['Gauss_fit_peak_parameters'][2]:.3f}",
                      fontsize=lsize, horizontalalignment='right',
                      transform=axs[0,1].transAxes)        

    axs[1,0].set_xlim([0, 30])
    axs[1,0].set_xlabel("Time (days)", fontsize=fsize)
    axs[1,0].set_ylim([LS_dict['median_MAD_nLC'][0]-(8.*LS_dict['median_MAD_nLC'][1]),
                       LS_dict['median_MAD_nLC'][0]+(8.*LS_dict['median_MAD_nLC'][1])])
    axs[1,0].set_ylabel("normalised flux", c='g', fontsize=fsize)
    axs[1,0].plot(clean["time0"], LS_dict['y_fit_LS'], c='orange', linewidth=0.5, label='LS best fit')
    axs[1,0].scatter(t_orig0, orig["nflux"], s=0.5, alpha=0.3, label='raw, normalized')
    axs[1,0].scatter(clean["time0"], clean["oflux"],s=0.5, c='r', alpha=0.5, label='cleaned, normalized')
    axs[1,0].scatter(clean["time0"], clean["nflux"],s=1.2, c='g', alpha=0.7, label='cleaned, normalized, detrended')
    print(t_table['source_id'])
    axs[1,0].text(0.99,0.90, f"Gaia DR3 {t_table['source_id'][0]}", fontsize=lsize, horizontalalignment='right', transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.80, f"Gmag = {float(t_table['Gmag']):.3f}", fontsize=lsize, horizontalalignment='right', transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.70, "$\log (f_{\\rm bg}/f_{*})$ = " + f"{float(t_table['log_tot_bg']):.3f}", fontsize=lsize, horizontalalignment='right', transform=axs[1,0].transAxes)
    axs[1,0].legend(loc='lower right')
    ax2=axs[1,0].twinx()
    ax2.set_position([0.05,0.3,0.90,0.2])
    ax2.invert_yaxis()
    ax2.scatter(t_orig0, orig["mag"], s=0.3, alpha=0.3, color="b", marker="x")
    ax2.set_ylabel("TESS magnitude", c="b",fontsize=fsize)


    axs[1,1].set_xlim([0,1])
    axs[1,1].set_xlabel("phase", fontsize=fsize)
    axs[1,1].set_ylabel("normalised flux", fontsize=fsize)
    axs[1,1].plot(LS_dict['phase_fit_x'], LS_dict['phase_fit_y'], c='b')
    LS_dict["phase_col"] += 1
    N_cyc = int(max(LS_dict["phase_col"]))
    cmap_use = plt.get_cmap('rainbow', N_cyc)
    s = axs[1,1].scatter(LS_dict['phase_x'], LS_dict["phase_y"], c=LS_dict['phase_col'], cmap=cmap_use, vmin=0.5, vmax=N_cyc+0.5)
    axs[1,1].text(0.01, 0.90, f"Amplitude = {LS_dict['pops_vals'][1]:.3f}, Scatter = {LS_dict['phase_scatter']:.3f}", fontsize=lsize, horizontalalignment='left', transform=axs[1,1].transAxes)

    cbaxes = inset_axes(axs[1,1], width="100%", height="100%", bbox_to_anchor=(0.79, 0.92, 0.20, 0.05), bbox_transform=axs[1,1].transAxes)
    cbar = plt.colorbar(s, cax=cbaxes, orientation='horizontal', label='cycle number')


    plot_name = '_'.join([name_underscore, f"{scc[0]:04d}", str(scc[1]), str(scc[2])])+'.png'
    print(name_underscore, plot_name)
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close('all')

