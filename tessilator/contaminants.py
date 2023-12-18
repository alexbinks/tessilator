'''

Alexander Binks & Moritz Guenther, December 2023

Licence: MIT 2023

This module contains functions to quantify the background flux contamination from
neighbouring sources in the TESS full-frame images. Each TESS pixel has a length of
21", therefore targets are highly susceptible to background contamination when they
(1) are in crowded fields, (2) are faint sources and/or (3) have high contributing
sky-counts.

The background is quanitified by acquistioning RP-band magnitudes in the Gaia DR3
catalogue for potential contaminants in the surrounding image apertures, and
then calculating their flux contribution which is incident within the target
aperture using an analytical formula provided as equation 3b-10 in Biser & Millman
(1965).

An additional function can be called which stores the potential neighbouring
contaminants, and periodogram analyses for these neighbouring sources are perform to
assess whether the observed periodicity is that of the target, or a neighbour.
'''

 
import traceback
import logging
import warnings
import sys

# Third party imports
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table, Row

from .fixedconstants import *

__all__ = ['logger', 'run_sql_query_contaminants', 'flux_fraction_contaminant', 'contamination']

logger = logging.getLogger(__name__)


def run_sql_query_contaminants(t_target, pix_radius=10., mag_lim=3., tot_attempts=3):
    '''Perform an SQL Query to identify neighbouring contaminants.

    If an analysis of flux contribution from neighbouring contaminants is
    required, this function generates the SQL query to identify targets
    within a specified pixel radius. The function returns a table of Gaia
    information on the neighbouring sources which is used to quantify the
    flux contamination within the target aperture.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The input table
    pix_radius : `float`, optional, default=10.
        The maximum angular distance (in arcsecs) to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faint magnitude limit to search for contaminants, where this value
        is relative to the target. E.G., a value of 3. is 3. magnitudes
        fainter than the target.
    tot_attempts : `int`, optional, default=3
        The total number of SQL query attempts to be made, in case of http
        response issues.

    returns
    -------
    t_gaia : `astropy.table.Table`
        The Gaia results table from the SQL query.    
    '''
    # Generate an SQL query for each target.
    query = f"SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, \
    DISTANCE(\
    POINT({t_target['ra']}, {t_target['dec']}),\
    POINT(ra, dec)) AS ang_sep\
    FROM gaiadr3.gaia_source\
    WHERE 1 = CONTAINS(\
    POINT({t_target['ra']}, {t_target['dec']}),\
    CIRCLE(ra, dec, {pix_radius*pixel_size/3600.})) \
    AND phot_g_mean_mag < {t_target['RPmag']+mag_lim} \
    ORDER BY phot_g_mean_mag ASC"

    # Attempt a synchronous SQL job, otherwise try the asyncronous method.

    num_attempts = 0
    while num_attempts < tot_attempts:
        print(f'attempting sql query for identifying contaminants: attempt {num_attempts+1} of {tot_attempts}...')
        try:
            job = Gaia.launch_job(query)
            break
        except:
            logger.warning(f"Couldn't run the sync query for "
                           f"{t_target['source_id']}, attempt {num_attempts+1}")
            try:
                job = Gaia.launch_job_async(query)
                break
            except:
                logger.warning(f"Couldn't run the async query for "
                               f"{t_target['source_id']}, attempt {num_attempts+1}")
                num_attempts += 1
    if num_attempts < tot_attempts:
        t_gaia = job.get_results()
        return t_gaia
    else:
        print('Most likely there is a server problem. Try again later.')
        sys.exit()

def flux_fraction_contaminant(ang_sep, s, d_thr=5.e-6):
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
    d_thr : `float`, optional, default=5.e-6
        The threshold value to stop the summations. When the next component contributes a value which is
        less than d_thr, the summation ends. 

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
        if np.abs(n_0) > d_thr:
            n = n+1
        if np.abs(n_0) < d_thr:
            break
    frac_flux_in_aperture = n_z*np.exp(-t)
    return frac_flux_in_aperture



def contamination(t_targets, Rad=1.0, n_cont=10):
    '''Estimate flux from neighbouring contaminant sources.

    The purpose of this function is to estimate the amount of flux incident in
    the TESS aperture that originates from neighbouring, contaminating sources.
    Given that the passbands from TESS (T-band, 600-1000nm) are similar to Gaia
    RP magnitude, and that Gaia can observe targets down to G~21, the Gaia DR3
    catalogue is used to quantify contamination.
    
    For each target in the input file, the function "run_sql_query_contaminants"
    returns a catalogue of Gaia DR3 objects of all neighbouring sources that
    are within a chosen pixel radius and are brighter than $RP_{source} + d_{thr}$.
    
    The Rayleigh formula is used to calculate the fraction of flux incident in
    the aperture from the target, and the function "flux_fraction_contaminant"
    uses an analytical formula `(Biser & Millman 1965, equation 3b-10) <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_ to
    calculate the flux contribution from all neighbouring sources incident in
    the aperture.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table for all the targets.
    Rad : `float`, optional, default=1.0
        The size of the radius aperture (in pixels)
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.

    returns
    -------
    t_targets : `astropy.table.Table`
        The input table for all the targets with 3 extra columns to quantify
        the flux contamination.
    t_cont : `astropy.table.Table`
        A table of Gaia DR3 data for the contaminants.
    '''
    con1, con2, con3 = [], [], []
    # Create empty table to fill with results from the contamination analysis.
    t_cont = Table(names=('source_id_target', 'source_id', 'RA',\
                          'DEC', 'Gmag', 'BPmag', 'RPmag', 'd_as', 'log_flux_frac'),\
                   dtype=(str, str, float, float, float, float, float, float, float))

    for i, t_target in enumerate(t_targets):
        r = run_sql_query_contaminants(t_target)
        print(f"sql search for contaminants completed {t_target['source_id']}, target {i+1} of {len(t_targets)}.")
        # convert the angular separation from degrees to arcseconds
        r["ang_sep"] = r["ang_sep"]*3600.
        if len(r) > 1:
            # make a table of all objects from the SQL except the target itself
            rx = Table(r[r["source_id"].astype(str) != \
                         t_target["source_id"].astype(str)])
            # calculate the fraction of flux from the source object that falls
            # into the aperture using the Rayleigh formula
            s = Rad**(2)/(2.0*exprf**(2)) # measured in pixels
            frp_star = (1.0-np.exp(-s))*10**(-0.4*t_target["RPmag"])
            frp_cont = []
            # calculate the fractional flux incident in the aperture from
            # each contaminant.
            for G_cont, RP_cont, ang_sep in zip(rx["phot_g_mean_mag"], rx["phot_rp_mean_mag"], rx["ang_sep"]):
                # if there is no RP magnitude, use the G magnitude, and add 0.756 to it (equivalent to removing
                # half the G-band flux, which could represent the red part of the G-magnitude passband.)
                if type(RP_cont) == np.ma.core.MaskedConstant:
                    RP_cont = G_cont + 0.756
                    RP_cont = RP_cont.astype(np.float32)
                
                f_frac = flux_fraction_contaminant(ang_sep, s)
                frp_cont.append(f_frac*10**(-0.4*RP_cont))

            rx['log_flux_frac'] = np.log10(frp_cont/frp_star)
            rx['source_id_target'] = t_target["source_id"]
            new_order = ['source_id_target', 'source_id', 'ra', 'dec', \
                         'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'ang_sep', 'log_flux_frac']
            rx.sort(['log_flux_frac'], reverse=True)
            rx = rx[new_order]
            rx['source_id_target'] = rx['source_id_target'].astype(str)
            rx['source_id'] = rx['source_id'].astype(str)
            # store the n_cont highest flux contributors to table
            for rx_row in rx[0:n_cont][:]:
                t_cont.add_row(rx_row)

            con1.append(np.log10(np.sum(frp_cont)/frp_star))
            con2.append(np.log10(max(frp_cont)/frp_star))
            con3.append(len(frp_cont))
        else:
            con1.append(-999)
            con2.append(-999)
            con3.append(0)

    t_targets["log_tot_bg"] = con1
    t_targets["log_max_bg"] = con2
    t_targets["num_tot_bg"] = con3

    return t_targets, t_cont

