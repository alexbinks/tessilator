"""

Alexander Binks & Moritz Guenther, 2024

License: MIT 2024

This module contains functions to quantify the background flux contamination
from neighbouring sources in the TESS full-frame images. Each TESS pixel has
a length of 21", therefore targets are highly susceptible to background
contamination when they (1) are in crowded fields, (2) are faint sources
and/or (3) have high contributing sky-counts.

The background is quantified by acquistioning RP-band magnitudes in the Gaia
DR3 catalogue for potential contaminants in the surrounding image apertures,
and then calculating their flux contribution which is incident within the
target aperture using an analytical formula provided as equation 3b-10 in
Biser & Millman (1965).

An additional function can be called which stores the potential neighbouring
contaminants. Periodogram analyses for these neighbouring sources are then
performed to help distinguish whether the lightcurve/periodogram signal is
that of the target, or from neighbouring sources.
"""

###############################################################################
####################################IMPORTS####################################
###############################################################################
# Internal
import inspect
import sys
import math

# Third party imports
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table


# Local application
from .fixedconstants import pixel_size, exprf
from .file_io import logger_tessilator
###############################################################################
###############################################################################
###############################################################################


# initialize the logger object
logger = logger_tessilator(__name__)





def run_sql_query_contaminants(t_target, cont_rad=10., mag_lim=3.,
                               tot_attempts=3):
    '''Perform an SQL query to identify neighbouring contaminants.

    This function generates the SQL query to identify targets within a
    specified pixel radius. The function returns a table of Gaia
    information on the neighbouring sources which is used to quantify the flux
    contamination within the target aperture.

    parameters
    ----------
    t_target : `astropy.table.Table`
        The input table
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
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
    query = f"SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, \
              phot_rp_mean_mag, \
              DISTANCE(\
              POINT({t_target['ra']}, {t_target['dec']}),\
              POINT(ra, dec)) AS ang_sep\
              FROM gaiadr3.gaia_source\
              WHERE 1 = CONTAINS(\
              POINT({t_target['ra']}, {t_target['dec']}),\
              CIRCLE(ra, dec, {cont_rad*pixel_size/3600.})) \
              AND phot_g_mean_mag < {t_target['RPmag']+mag_lim} \
              ORDER BY phot_g_mean_mag ASC"

    # Attempt a synchronous SQL job, otherwise try the asyncronous method.

    num_attempts = 0
    while num_attempts < tot_attempts:
        print(f'attempting sql query for identifying contaminants: attempt '
              f'{num_attempts+1} of {tot_attempts}...')
        try:
            job = Gaia.launch_job(query)
            break
        except:
            logger.warning(f"Couldn't run the sync query for "
                           f"{t_target['source_id']}, attempt "
                           f"{num_attempts+1}")
            try:
                job = Gaia.launch_job_async(query)
                break
            except:
                logger.warning(f"Couldn't run the async query for "
                               f"{t_target['source_id']}, attempt "
                               f"{num_attempts+1}")
                num_attempts += 1
    if num_attempts < tot_attempts:
        t_gaia = job.get_results()
        return t_gaia
    else:
        print('Most likely there is a server problem. Try again later.')
        sys.exit()


def flux_fraction_contaminant(pix_sep, s, d_thr=5.e-6):
    r"""Quantify the flux contamination from a neighbouring source.

    Calculates the fraction of flux from a neighbouring contaminating source
    that gets scattered into the aperture. The analytic function uses equation
    3b-10 from `Biser & Millman (1965)
    <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_, which is a double
    converging sum with infinite limits, given by

    .. math::

       f_{\rm bg} = e^{-t} \sum_{n=0}^{n\to{\infty}}
       {\Bigg\{\frac{t^{n}}{n!}\bigg[1-e^{-s}\sum_{k=0}^{n}
       {\frac{s^{k}}{k!}}} \bigg]\Bigg\}

    To solve the equation computationally, the summation terminates once the
    difference from the nth iteration is less than some given threshold value,
    `d_thr`.

    parameters
    ----------
    pix_sep : `float`
        The pixel distance between a contaminant and the aperture centre.
    s : `float`
        For a given aperture size, rad (in pixels)
        and an FWHM of the TESS PSF, exprf (set at 0.65 pixels),
        :math:`s = {\rm rad}^2/(2.0*{\rm exprf}^2)`
    d_thr : `float`, optional, default=5.e-6
        The threshold value to stop the summations. When the next component
        contributes a value which is less than d_thr, the summation ends.

    returns
    -------
    frac_flux_in_aperture : `float`
        Fraction of contaminant flux that gets scattered into the aperture.
    """
    n, n_z, n_0, n_sign = 0, 0, 0, 1.
    n_sign_lim = 0
    t = pix_sep**2/(2.0*exprf**(2))
    try:
        while True:
            n_old = n_0
            n_sign_old = n_sign
            sk = np.sum([(s**(k)/math.factorial(k)) for k in range(0,n+1)])
            sx = 1.0 - (np.exp(-s)*sk)
            n_0 = ((t**n)/math.factorial(n))*sx
            n_sign = np.sign(n_0-n_old)
            n_z += n_0

            if (n_sign_old-n_sign) != 0:
                n_sign_lim += 1
                if n_sign_lim > 1:
                    return 0.
            if np.abs(n_0) > d_thr:
                n += 1
            else:
                break
        frac_flux_in_aperture = n_z*np.exp(-t)
    except:
        logger.warning('contamination sum did not converge')
        frac_flux_in_aperture = 0.
    return frac_flux_in_aperture


def contamination(t_targets, ap_rad=1.0, n_cont=10, cont_rad=10., mag_lim=3.,
                  tot_attempts=3):
    '''Estimate flux from neighbouring contaminant sources.

    The purpose of this function is to estimate the amount of flux incident in
    the TESS aperture that originates from neighbouring, contaminating sources.
    Given that the passbands from TESS (T-band, 600-1000nm) are similar to Gaia
    RP magnitude, and that Gaia can observe targets down to G~21, the Gaia DR3
    catalogue is used to quantify contamination.
    
    For each target in the input file, the function
    "run_sql_query_contaminants" returns a catalogue of Gaia DR3 objects of all
    neighbouring sources that are within a chosen pixel radius and are brighter
    than $RP_{\\rm source}+$mag_lim.
    
    The Rayleigh formula is used to calculate the fraction of flux incident in
    the aperture from the target, and the function "flux_fraction_contaminant"
    uses an analytical formula `(Biser & Millman 1965, equation 3b-10)
    <https://books.google.co.uk/books?id=5XBGAAAAYAAJ>`_ to
    calculate the flux contribution from all neighbouring sources incident in
    the aperture.

    parameters
    ----------
    t_targets : `astropy.table.Table`
        The input table for all the targets.
    ap_rad : `float`, optional, default=1.0
        The size of the radius aperture (in pixels)
    n_cont : `int`, optional, default=10
        The maximum number of neighbouring contaminants to store to table.
    cont_rad : `float`, optional, default=10.
        The maximum pixel radius to search for contaminants
    mag_lim : `float`, optional, default=3.
        The faintest magnitude to search for contaminants.
    tot_attempts : `int`, optional, default=3
        The number of sql query attempts to be made to acquire Gaia DR3
        data for a contaminant before a time-out error occurs.

    returns
    -------
    t_targets : `astropy.table.Table`
        The input table for all the targets with 3 extra columns to quantify
        the flux contamination.
    t_cont : `astropy.table.Table`
        A table of Gaia DR3 data for the contaminants.
    '''
    con_tot, con_max, con_num = [], [], []
    # Create empty table to fill with results from the contamination analysis.
    t_cont = Table(names=('source_id_target', 'source_id', 'RA',
                          'DEC', 'Gmag', 'BPmag', 'RPmag', 'd_as',
                          'log_flux_frac'),\
                   dtype=(str, str, float, float, float, float, float, float,
                          float))

    for i, t_target in enumerate(t_targets):
        r = run_sql_query_contaminants(t_target, cont_rad=cont_rad, mag_lim=mag_lim,
                                       tot_attempts=tot_attempts)
        r["SOURCE_ID"] = [f"Gaia DR3 {i}" for i in r["SOURCE_ID"]]
        print(f"sql search for contaminants completed {t_target['source_id']},"
              f" target {i+1} of {len(t_targets)}.")
        # convert the angular separation from degrees to arcseconds
        r["pix_sep"] = r["ang_sep"]*3600./pixel_size
        if len(r) > 1:
            # make a table of all objects from the SQL except the target itself
            rx = Table(r[r["SOURCE_ID"] != t_target["source_id"]])
            # calculate the fraction of flux from the source object that falls
            # into the aperture using the Rayleigh formula
            s = ap_rad**(2)/(2.0*exprf**(2)) # measured in pixels
            frp_star = (1.0-np.exp(-s))*10**(-0.4*t_target["RPmag"])
            frp_conts = []
            # calculate the fractional flux incident in the aperture from
            # each contaminant.
            for G_cont, RP_cont, pix_sep in zip(rx["phot_g_mean_mag"],
                                                rx["phot_rp_mean_mag"],
                                                rx["pix_sep"]):
                # if there is no RP magnitude, use the G magnitude, and add
                # 0.756 to it (equivalent to removing half the G-band flux, 
                # which could represent the red part of the G-magnitude
                # passband.)
                if type(RP_cont) == np.ma.core.MaskedConstant:
                    RP_cont = G_cont + 0.756
                    RP_cont = RP_cont.astype(np.float32)
                
                f_frac = flux_fraction_contaminant(pix_sep, s)
                frp_conts.append(f_frac*10**(-0.4*RP_cont))
            rx['log_flux_frac'] = 0.
            frp_tot, frp_max = 0., 0.
            for f, frp_cont in enumerate(frp_conts):
                if frp_cont > 0.:
                    rx['log_flux_frac'][f] = np.log10(frp_cont/frp_star)
                    frp_tot += frp_cont
                    if frp_cont > frp_max:
                        frp_max = frp_cont
                else:
                    rx['log_flux_frac'][f] = -99.

            rx['source_id_target'] = t_target["source_id"]
            new_order = ['source_id_target', 'SOURCE_ID', 'ra', 'dec', 
                         'phot_g_mean_mag', 'phot_bp_mean_mag',
                         'phot_rp_mean_mag', 'pix_sep', 'log_flux_frac']
            rx.sort(['log_flux_frac'], reverse=True)
            rx = rx[new_order]
            rx['source_id_target'] = rx['source_id_target']
            rx.rename_column('SOURCE_ID', 'source_id')
            rx['source_id'] = rx['source_id']

            # store the n_cont highest flux contributors to table
            for rx_row in rx[0:n_cont][:]:
                t_cont.add_row(rx_row)

            if frp_tot > 0.:
                con_tot.append(np.log10(frp_tot/frp_star))
            else:
                con_tot.append(-99.)
            if frp_max > 0.:
                con_max.append(np.log10(frp_max/frp_star))
            else:
                con_max.append(-99.)
            con_num.append(len(frp_conts))
        else:
            con_tot.append(-999)
            con_max.append(-999)
            con_num.append(0)

    t_targets["log_tot_bg"] = con_tot
    t_targets["log_max_bg"] = con_max
    t_targets["num_tot_bg"] = con_num

    return t_targets, t_cont


def is_period_cont(d_target, d_cont, t_cont, frac_amp_cont=0.5):
    '''Identify neighbouring contaminants that may cause the periodicity.

    If the user selects to measure periods for the neighbouring contaminants
    this function returns a flag to assess if a contaminant may actually be
    the source causing the observed periodicity. The function produces two
    flags: `false_flag` and `reliable_flag`, where the former assesses how
    likely a contaminant might be causing the periodicity, and the latter
    indicates how likely it is that the periodicity comes from the target.

    parameters
    ----------
    d_target : `dict`
        A dictionary containing periodogram data of the target star.
    d_cont : `dict`
        A dictionary containing periodogram data of the contaminant star.
    t_cont : `astropy.table.Table`
        A table containing Gaia data for the contaminant star
    frac_amp_cont : `float`, optional, default=0.5
        The threshold factor to account for the difference in amplitude
        of the two stars. If this is high, then the contaminants will be
        less likely to be flagged as the potential source
    
    returns
    -------
    false_flag : `int`
        Either 1 or 0, with 1 (0) indicating the contaminant is (un)likely to
        cause the periodicity.
    reliable_flag : `int`
        Either 1 or 0, with 1 (0) suggesting that the target provides an
        unreliable (reliable) rotation period.        
    '''
    pix_dist = t_cont["pix_sep"]

    per_targ = d_target["period_1"]
    err_targ = d_target["Gauss_fit_peak_parameters"][2]
    amp_targ = d_target["pops_vals"][1]
    RP_targ = d_target["RPmag"]

    per_cont = d_cont["period_best"]
    err_cont = d_cont["Gauss_fit_peak_parameters"][2]
    amp_cont = d_cont["pops_vals"][1]
    RP_cont = t_cont["RPmag"]
    
    
    false_flag, reliable_flag = 0, 0
    if abs(per_targ - per_cont) < (err_targ + err_cont):
        false_flag = 1

    if pix_dist < 1:
        if RP_targ > RP_cont:
            reliable_flag = 1
    else:
        if amp_targ < amp_cont:
            reliable_flag = 1    
    return false_flag, reliable_flag



__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__], 
           predicate = lambda f: inspect.isfunction(f) and
           f.__module__ == __name__)]
