'''Make tabular data for tessilator

This module contains the functions required to convert the input data
into correct formatted astropy tables to be used for further analysis
in the tessilator.
'''

__all__ = ['logger', 'table_from_simbad', 'table_from_coords',
           'table_from_table', 'get_gaia_data']

from astropy.table import Table
from astropy.io import ascii
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord, ICRS

import numpy as np
import warnings
import logging
           
logger = logging.getLogger(__name__)



# Read the data from the input file as an astropy table (ascii format)
# Ensure the source_id has a string type (not long integer).

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
        result_table = [Simbad.query_objectids(name) for name in name_arr]
    NameList = []
    GaiaList = []
    for r, res in enumerate(result_table):
        input_name = input_names["ID"][r]
        if res is None: # no targets resolved by SIMBAD
            print(f"Simbad did not resolve {input_name} - checking Gaia")
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
                    print("Simbad didn't resolve Gaia DR3 identifiers for "
                          f"{input_name}, but we'll check anyway!")
                    NameList.append("Gaia DR3 " + input_name)
                    GaiaList.append(input_name)
                else:
                    logger.error(f"Could not find any match for '{input_name}'")
            else:
                NameList.append(input_name)
                GaiaList.append(m[0].split(' ')[2])

    # Part 2: Query Gaia database using Gaia identifiers retrieved in part 1.
    ID_string = ""
    for g_i, gaia_name in enumerate(GaiaList):
        if g_i < len(GaiaList)-1:
            ID_string += gaia_name+','
        else:
            ID_string += gaia_name
    qry = "SELECT source_id,ra,dec,parallax,phot_g_mean_mag "\
          "FROM gaiadr3.gaia_source "\
          f"WHERE source_id in ({ID_string});"
    print(qry)
    job = Gaia.launch_job_async( qry )
    print(job)
    gaia_table = job.get_results() # Astropy table
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
    new_order = ['name', 'source_id', 'ra', 'dec', 'parallax', 'Gmag']
    gaia_table = gaia_table[new_order]
    return gaia_table


def table_from_coords(coord_table, ang_max=10.0, type_coord='icrs'):
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
        
    returns
    -------
    gaia_table : `astropy.table.Table`
        The output table ready for further analysis
    '''
    gaia_table = Table(names=('source_id', 'ra', 'dec', 'parallax', 'Gmag'), \
                       dtype=(int,float,float,float,float))
    if type_coord == 'galactic':
        gal = SkyCoord(l=coord_table['col1']*u.degree,\
                       b=coord_table['col2']*u.degree,\
                       frame='galactic')
        c = gal.transform_to(ICRS)
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg
    elif type_coord == 'ecliptic':
        ecl = SkyCoord(l=coord_table['col1']*u.degree,\
                       b=coord_table['col2']*u.degree,\
                        frame='ecliptic')
        c = ecl.transform_to(ICRS)
        coord_table['col1'], coord_table['col2'] = c.ra.deg, c.dec.deg

    coord_table.rename_column(coord_table.colnames[0], 'ra')
    coord_table.rename_column(coord_table.colnames[1], 'dec')

    for i in range(len(coord_table)):
    # Generate an SQL query for each target, where the nearest source is
    # returned within a maximum radius set by ang_max.
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
        if len(x) == 0:
            continue
        else:
            y = x[0]['source_id', 'ra', 'dec', 'parallax', 'phot_g_mean_mag']
            gaia_table.add_row((y))
    # For each source, query the identifiers resolved by SIMBAD and return the
    # target with the shortest number of characters (which is more likely to be
    # the most common reference name for the target).
    GDR3_Names = ["Gaia DR3 " + i for i in gaia_table['source_id'].astype(str)]
    result_table = [Simbad.query_objectids(i) for i in GDR3_Names]
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

def table_from_table(input_table, name_is_source_id=0):
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

        The column headers are not necessary.

    returns
    -------
    gaia_table : `astropy.table.Table`
        the output table ready for further analysis
    '''

    gaia_table = Table(data=input_table, dtype=(str, float, float, float, float),
                       names=('source_id', 'ra', 'dec', 'parallax', 'Gmag'))
    gaia_table['source_id'] = gaia_table['source_id'].astype(int)
    if name_is_source_id:
        gaia_table['name'] = gaia_table['source_id'].data
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


def get_gaia_data(gaia_table, name_is_source_id=0):
    '''Reads the input table and returns a table in the correct format for
    TESSilator.

    | The table must be in comma-separated variable format, in either of these 3 ways:
    | 1) A table with a single column containing the source identifier
    |     * note that this is the preferred method since the target identified in the Gaia query is unambiguously the same as the input value.
    | 2) A table with sky-coordinates in either the 'icrs' (default),'galactic', or 'ecliptic' system.
    |     * note this is slower because of the time required to run the Vizier query.
    | 3) A table with all 5 columns already made.

    parameters
    ----------
    gaia_table : `astropy.table.Table`
        The input table

    name_is_source_id : `bool`, optional, default=0
        If the input table has 5 columns, this provides the choice to set the
        name column equal to "source_id" (=1), or to find a common target
        identifier (=0)
    
    results
    -------
    tbl : `astropy.table.Table`
        | The table ready for TESSilator analysis, with the columns:
        * name: the preferred choice of source identifier
        * source_id: the Gaia DR3 source identifier
        * ra: right ascension in the ICRS (J2000) system
        * dec: declination in the ICRS (J2000) system
        * parallax: parallax from Gaia DR3 (in mas)
        * Gmag: the apparent G-band magnitude from Gaia DR3
    '''

    if len(gaia_table.colnames) == 1:
        tbl = table_from_simbad(gaia_table)
    elif len(gaia_table.colnames) == 2:
        tbl = table_from_coords(gaia_table)
    elif len(gaia_table.colnames) == 5:
        tbl = table_from_table(gaia_table, name_is_source_id)
    else:
        raise Exception('Input table has invalid format. Please use one of the \
                        following formats: \n [1] source_id \n [2] ra and dec \
                        \n [3] source_id, ra, dec, parallax and Gmag')
    return tbl
