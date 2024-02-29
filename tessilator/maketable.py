'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

Make tabular data for tessilator

This module contains the functions required to convert the input data
into correct formatted astropy tables to be used for further analysis
in the tessilator.
'''

###############################################################################
####################################IMPORTS####################################
###############################################################################
#Internal
import warnings
import sys
import inspect

#Third party
from astropy.table import Table
from astropy.io import ascii
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, ICRS
import astropy.units as u
import numpy as np

# Local application
from .logger import logger_tessilator
###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__) 




def table_from_simbad(input_names):
    '''Generate the formatted astropy table from a list of target names.

    All characters can be parsed except commas, since the table is in comma
    separated variable (.csv) format.

    parameters
    ----------
    input_names : `astropy.table.Table`
        an input list of target names

    returns
    -------
    gaia_table : `astropy.table.Table`
        the output table ready for further analysis
    '''
    # Part 1: Use the SIMBAD database to retrieve the Gaia source identifier
    #         from the target names. 
    # set the column header = "ID"
    input_names.rename_column(input_names.colnames[0], 'ID')
    input_names["ID"] = input_names["ID"].astype(str)
    # create arrays to store naming variables
    name_arr, is_Gaia = [], [0 for i in input_names]
    for i, input_name in enumerate(input_names["ID"]):
    # if the target name is the numeric part of the Gaia DR3 source identifier
    # prefix the name with "Gaia DR3 "
        if input_name.isnumeric() and len(input_name) > 10:
            input_name = "Gaia DR3 " + input_name
            is_Gaia[i] = 1
        name_arr.append(input_name)

    # Get a list object identifiers from Simbad
    # suppress the Simbad.query_objectids warnings if there are no matches for
    # the input name
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        try:
            result_table = [Simbad.query_objectids(name) for name in name_arr]
        except:
            result_table = [None for name in name_arr]
    NameList = []
    GaiaList = []
    for r, res in enumerate(result_table):
        input_name = input_names["ID"][r]
        if res is None: # no targets resolved by SIMBAD
            logger.warning(f"Simbad did not resolve {input_name} - checking Gaia")
            if is_Gaia[i] == 1:
                NameList.append("Gaia DR3 " + input_name)
                GaiaList.append(input_name)
            else:
                logger.error(f"Could not find any match for '{input_name}'")
        else: # Simbad returns at least one identifier
            r_list = [z for z in res["ID"]]
            m = [s for s in r_list if "Gaia DR3" in s]
            if len(m) == 0: # if Gaia identifier is not in the Simbad list
                if is_Gaia[i] == 1:
                    logger.warning("Simbad didn't resolve Gaia DR3 identifiers for "
                          f"{input_name}, but we'll check anyway!")
                    NameList.append("Gaia DR3 " + input_name)
                    GaiaList.append(input_name)
                else:
                    logger.error(f"Could not find any match for '{input_name}'")
            else:
                NameList.append(input_name)
                GaiaList.append(m[0].split(' ')[2])
    if len(NameList) == 0:
        logger.error(f"No targets have been resolved, either by Simbad or Gaia DR3. Please check the target names are resolvable.")
        sys.exit()
    # Part 2: Query Gaia database using Gaia identifiers retrieved in part 1.
    ID_string = ""
    for g_i, gaia_name in enumerate(GaiaList):
        if g_i == len(GaiaList)-1:
            ID_string += gaia_name
        else:
            ID_string += gaia_name+','
    qry = "SELECT source_id,ra,dec,parallax,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag "\
          "FROM gaiadr3.gaia_source "\
          f"WHERE source_id in ({ID_string});"
    job = Gaia.launch_job_async( qry )
    gaia_table = job.get_results() # Astropy table
    logger.info('query completed!')
    # convert source_id column to str (astroquery returns type np.int64)
    gaia_table["source_id"] = gaia_table["source_id"].astype(str)
    list_ind = []
    # astroquery returns the table sorted numerically by the source identifier
    # the rows are rearranged to match with the input list.
    for row in GaiaList:
        list_ind.append(np.where(np.array(gaia_table["source_id"] == \
                        str(row)))[0][0])
    gaia_table = gaia_table[list_ind]
    gaia_table['name'] = NameList
    gaia_table.rename_column('phot_g_mean_mag', 'Gmag')
    gaia_table.rename_column('phot_bp_mean_mag', 'BPmag')
    gaia_table.rename_column('phot_rp_mean_mag', 'RPmag')
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag']
    gaia_table = gaia_table[new_order]
    return gaia_table



def get_twomass_like_name(coords):
    '''If the Gaia DR3 system is not chosen, this function returns a string
    which has the same format as the 2MASS identifiers.
    
    parameters
    ----------
    coords : `astropy.coordinates.SkyCoord`
         The SkyCoord tuple of right ascencion and declination values
         
    returns
    -------
    radec_fin : `list`
        A list of 2MASS-like identifiers
    '''
    ra_hms = coords.ra.to_string(u.h, sep="", precision=2, alwayssign=False, pad=True)
    ra_hms_fin = [ra.replace(".","") for ra in ra_hms]
    dec_hms = coords.dec.to_string(sep="", precision=1, alwayssign=True, pad=True)
    dec_hms_fin = [dec.replace(".","") for dec in dec_hms]
    radec_fin = []
    for r,d in zip(ra_hms_fin, dec_hms_fin):
        radec_fin.append(f'{r}{d}')
    return radec_fin



def table_from_coords(coord_table, ang_max=10.0, type_coord='icrs', gaia_sys=True):
    '''Generate the formatted astropy table from a list of coordinates.

    Each entry needs to be in comma separated variable(.csv) format.

    parameters
    ----------
    coord_table : `astropy.table.Table`
        a table consisting of right ascension and declination coordinates (in
        degrees) in comma separated variable (.csv) format.
    ang_max : `float`, optional, default=10.0
        the maximum angular distance in arcseconds from the input coordinates
        provided in the table.
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input positions. These can be 'icrs'
        (default), 'galactic' or 'ecliptic'.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination
        can be calculated if this is False.

    returns
    -------
    gaia_table : `astropy.table.Table`
        The output table ready for further analysis
    '''
    gaia_table = Table(names=('source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag'), \
                       dtype=(int,float,float,float,float,float,float))
    if type_coord == 'galactic':
        gal = SkyCoord(l=coord_table['col1'],\
                       b=coord_table['col2'],\
                       unit=u.deg, frame='galactic')
        c = gal.transform_to(ICRS)
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    elif type_coord == 'ecliptic':
        ecl = SkyCoord(lon=coord_table['col1'],\
                       lat=coord_table['col2'],\
                       unit=u.deg, frame='barycentricmeanecliptic')
        c = ecl.transform_to(ICRS)
    elif type_coord == 'icrs':
        c = SkyCoord(ra=coord_table['col1'], dec=coord_table['col2'], unit=u.deg, frame='icrs')
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    coord_table.rename_column(coord_table.colnames[0], 'ra')
    coord_table.rename_column(coord_table.colnames[1], 'dec')

    if gaia_sys:
        for i in range(len(coord_table)):
        # Generate an SQL query for each target, where the nearest source is
        # returned within a maximum radius set by ang_max.
            qry = f"SELECT source_id,ra,dec,parallax,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag, \
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
            print(f'astroquery completed for target {i+1} of {len(coord_table)}')
            # Fill the empty table with results from astroquery
            if len(x) == 0:
                continue
            else:
                y = x[0]['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']
                gaia_table.add_row((y))
        # For each source, query the identifiers resolved by SIMBAD and return the
        # target with the shortest number of characters (which is more likely to be
        # the most common reference name for the target).
        GDR3_Names = ["Gaia DR3 " + i for i in gaia_table['source_id'].astype(str)]
        try:
            result_table = [Simbad.query_objectids(i) for i in GDR3_Names]
        except:
            result_table = [None for i in GDR3_Names]
        NameList = []
        for i, r in enumerate(result_table):
            if r is None:
                NameList.append(gaia_table['source_id'][i].astype(str))
            else:
                NameList.append(sorted(r["ID"], key=len)[0])
        gaia_table["name"] = NameList
    else:
        twomass_name = get_twomass_like_name(c)
        for i in range(len(twomass_name)):
            source_id = f'{i+1:0{len(str(len(twomass_name)))}d}'
            row = [source_id, c[i].ra.deg, c[i].dec.deg, -999, -999, -999, -999]
            gaia_table.add_row(row)
        gaia_table['name'] = twomass_name

    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag']
    gaia_table = gaia_table[new_order]
    return gaia_table
    
    

def table_from_table(input_table, name_is_source_id=False):
    '''Generate the formatted astropy table from a pre-formatted astropy
    table.

    Each entry needs to be in comma separated variable(.csv) format. This
    is the quickest way to produce the table ready for analysis, but it is
    important the input data is properly formatted.
    
    parameters
    ----------
    input_table : `astropy.table.Table`
        | The columns of table must be in the following order:
        * source_id (data type: `str`)
        * ra (data type: `float`)
        * dec (data type: `float`)
        * parallax (data type: `float`)
        * Gmag (data type: `float`)
        * BPmag (data type: `float`)
        * RPmag (data type: `float`)

        The column headers must not be included!
    name_is_source_id : `bool`, optional, default=False
        Choose if the name is to be the same as the Gaia DR3 source identifier.

    returns
    -------
    gaia_table : `astropy.table.Table`
        the output table ready for further analysis
    '''

    gaia_table = Table(data=input_table, dtype=(str, float, float, float, float, float, float),
                       names=('source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag'))
    gaia_table['source_id'] = gaia_table['source_id'].astype(int)
    if name_is_source_id:
        gaia_table['name'] = gaia_table['source_id'].data
    else:
        GDR3_Names = ["Gaia DR3 " + i for i in gaia_table['source_id'].astype(str)]
        NameList = []
        try:
            result_table =  [Simbad.query_objectids(i) for i in GDR3_Names]
        except:
            result_table = [None for i in GDR3_Names]
        for i, r in enumerate(result_table):
            if r is None:
                NameList.append(gaia_table['source_id'][i].astype(str))
            else:
                NameList.append(sorted(r["ID"], key=len)[0])
        gaia_table["name"] = NameList
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag', 'BPmag', 'RPmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def get_gaia_data(gaia_table, name_is_source_id=False, type_coord='icrs', gaia_sys=True):
    '''Reads the input table and returns a table in the correct format for
    TESSilator.

    | The table must be in comma-separated variable format, in either of these
      3 ways:
    | 1) A table with a single column containing the source identifier
    |     * note that this is the preferred method since the target identified
            in the Gaia query is unambiguously the same as the input value.
            Also, the name match runs faster than the coordinate match using
            astroquery.
    | 2) A table with sky-coordinates in either the 'icrs' (default),
         'galactic', or 'ecliptic' system.
    |     * note this is slower because of the time required to run the Vizier
            query.
    | 3) A table with all 7 columns already made.

    parameters
    ----------
    gaia_table : `astropy.table.Table`
        The input table
    name_is_source_id : `bool`, optional, default=False
        If the input table has 7 columns, this provides the choice to set the
        name column equal to "source_id" (True), or to find a common target
        identifier (False)
    type_coord : `str`, optional, default='icrs'
        The coordinate system of the input data. Choose from 'icrs', 'galactic' or
        'barycentricmeanecliptic', where the latter is the conventional coordinate
        system used by TESS.
    gaia_sys : `bool`, optional, default=True
        Choose to format the data based on Gaia DR3. Note that no contamination can
        be calculated if this is False.
    
    results
    -------
    tbl : `astropy.table.Table`
        | The table ready for TESSilator analysis, with the columns:
        * name: the preferred choice of source identifier
        * source_id: the Gaia DR3 source identifier
        * ra: right ascension (icrs) or longditude (galactic,
          barycentricmeanecliptic)
        * dec: declination (icrs) or latitude (galactic,
          barycentricmeanecliptic)
        * parallax: parallax from Gaia DR3 (in mas)
        * Gmag: the apparent G-band magnitude from Gaia DR3
        * BPmag: the apparent BP-band magnitude from Gaia DR3
        * RPmag: the apparent RP-band magnitude from Gaia DR3
    '''
    if len(gaia_table.colnames) == 1:
        tbl = table_from_simbad(gaia_table)
    elif len(gaia_table.colnames) == 2:
        tbl = table_from_coords(gaia_table, type_coord=type_coord, gaia_sys=gaia_sys)
    elif len(gaia_table.colnames) == 7:
        tbl = table_from_table(gaia_table, name_is_source_id=name_is_source_id)
    else:
        raise Exception('Input table has invalid format. Please use one of the \
                        following formats: \n [1] source_id \n [2] ra and dec \
                        \n [3] source_id, ra, dec, parallax, Gmag, BPmag and RPmag')
    return tbl


__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__], predicate = lambda f: inspect.isfunction(f) and f.__module__ == __name__)]
