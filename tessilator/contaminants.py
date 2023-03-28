import traceback
import logging
import warnings


# Third party imports
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table

from .fixedconstants import *

__all__ = ['logger', 'run_sql_query_contaminants', 'flux_fraction_contaminant', 'contamination']

logger = logging.getLogger(__name__)


def run_sql_query_contaminants(t_target, pix_radius=5.0):
    '''Perform an SQL Query to identify neighbouring contaminants.

    If an analysis of flux contribution from neighbouring contaminants is
    specified, this function generates the SQL query to identify targets
    within a specified pixel radius. The function returns a table of Gaia
    information on the neighbouring sources which is used to quantify the
    flux contamination within the target aperture.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The input table

    pix_radius : `float`, optional, default=5.0
        The maximum angular distance (in arcsecs) to search for contaminants

    returns
    -------
    t_gaia : `astropy.table.Table`
        The Gaia results table from the SQL query.    
    '''
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
        logger.warning(f"Couldn't run the sync query for "
                       f"{t_target['source_id']}")
        job = Gaia.launch_job_async(query)
    t_gaia = job.get_results()
    return t_gaia


def flux_fraction_contaminant(ang_sep, s, d_th=0.000005):
    '''Quantify the flux contamination from a neighbouring source.

    Calculates the fraction of flux from a neighbouring contaminating source
    that gets scattered into the aperture. The analytic function uses equation
    3b-10 from `Biser & Millman (1965) <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_, which is a double converging sum with infinite limits, given by
    
    .. math::

       f_{\\rm bg} = e^{-t} \sum_{n=0}^{n\\to{\infty}} {\Bigg\{\\frac{t^{n}}{n!}\\bigg[1-e^{-s}\sum_{k=0}^{n}{\\frac{s^{k}}{k!}}} \\bigg]\Bigg\}

    To solve the equation computationally, the summation terminates once the
    difference from the nth iteration is less than a given threshold value.

    parameters
    ----------
    ang_sep : `float`
        The angular distance (in arcseconds) between a contaminant and the
        aperture centre. 

    s : `float`
        For a given aperture size, Rad (in pixels)
        and an FWHM of the TESS PSF, exprf (set at 0.65 pixels), :math:`s = {\\rm Rad}^2/(2.0*{\\rm exprf}^2)`

    returns
    -------
    frac_flux_in_aperture : `float`
        Fraction of contaminant flux that gets scattered into the aperture.
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
    frac_flux_in_aperture = n_z*np.exp(-t)
    return frac_flux_in_aperture



def contamination(t_targets, LC_con, Rad=1.0, n_cont=5):
    '''Estimate flux from neighbouring contaminant sources.

    The purpose of this function is to estimate the amount of flux incident in
    the TESS aperture that originates from neighbouring, contaminating sources.
    Given that the passbands from TESS (T-band, 600-1000nm) are similar to Gaia
    G magnitude, and that Gaia is sensitive to G~21, the Gaia DR3 catalogue is
    used to quantify contamination.
    
    For each target in the input file, the function "runSQLQueryContaminants"
    returns a catalogue of Gaia DR3 objects of all neighbouring sources that
    are within a chosen pixel radius and are brighter than $G_{source} + 3$.
    
    The Rayleigh formula is used to calculate the fraction of flux incident in
    the aperture from the target, and the function "flux_fraction_contaminant"
    uses an analytical formula `(Biser & Millman 1965, equation 3b-10) <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_ to
    calculate the flux contribution from all neighbouring sources incident in
    the aperture.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table for all the targets.
    
    LC_con : `bool`
        If true, a table of Gaia DR3 information on the contaminants is
        returned, else None.
    
    Rad : `float`, optional, default=1.0
        The size of the radius aperture (in pixels)

    n_cont : `int`, optional, default=5
        The maximum number of neighbouring contaminants to store to table if
        LC_con is True.

    returns
    -------
    t_targets : `astropy.table.Table`
        The input table for all the targets with 3 extra columns to quantify
        the flux contamination.

    t_cont : `astropy.table.Table` or `None`
        If LC_con is true, a table of Gaia DR3 information on the contaminants
        is returned, else None.
    '''

    con1, con2, con3 = [], [], []
    # Create empty table to fill with results from the contamination analysis.
    if LC_con:
        t_cont = Table(names=('source_id_target', 'source_id', 'RA',\
                              'DEC', 'Gmag', 'd_as', 'log_flux_frac'),\
                       dtype=(str, str, float, float, float, float, float))
    for i in range(len(t_targets)):
        r = run_sql_query_contaminants(t_targets[i])
        # convert the angular separation from degrees to arcseconds
        r["ang_sep"] = r["ang_sep"]*3600.
        if len(r) > 1:
            # make a table of all objects from the SQL except the target itself
            rx = Table(r[r["source_id"].astype(str) != \
                         t_targets["source_id"][i].astype(str)])

            # calculate the fraction of flux from the source object that falls
            # into the aperture using the Rayleigh formula
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
                new_order = ['source_id_target', 'source_id', 'ra', 'dec', \
                             'phot_g_mean_mag', 'ang_sep', 'log_flux_frac']
                rx.sort(['log_flux_frac'], reverse=True)
                rx = rx[new_order]
                rx['source_id_target'] = rx['source_id_target'].astype(str)
                rx['source_id'] = rx['source_id'].astype(str)
                # store the n_cont highest flux contributors to table
                for rx_row in rx[0:n_cont][:]:
                    t_cont.add_row(rx_row)
            else:
                t_cont = None
            con1.append(np.log10(np.sum(fg_cont)/fg_star))
            con2.append(np.log10(max(fg_cont)/fg_star))
            con3.append(len(fg_cont))
        else:
            con1.append(-999)
            con2.append(-999)
            con3.append(0)
            t_cont = None

    t_targets["log_tot_bg"] = con1
    t_targets["log_max_bg"] = con2
    t_targets["num_tot_bg"] = con3

    return t_targets, t_cont

