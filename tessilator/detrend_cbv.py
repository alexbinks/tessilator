import logging
__all__ = ['get_cbv_scc', 'interpolate_cbv', 'fixed_nb', 'sel_nb', 'fit_basis', 'apply_basis']

import warnings

# Third party imports
import numpy as np
import os
import subprocess

from astropy.table import Table
from astropy.io import fits
import astropy.units as u


import pylab as pl
from .VBLinRegARD import bayes_linear_fit_ard as VBF
from .stats_cbv import cdpp, medransig




def fit_basis(flux, basis, scl = None):
    '''Calculate the flux-correction weights for the co-trending basis vectors (CBVs).
    
    This routine is taken directly from `Aigrain et al. 2017 <https://github.com/saigrain/CBVshrink>`_

    parameters
    ----------
    flux : `iterable`
        The list of normalised flux values from the target lightcurve.
    basis : `iterable`
        The list of co-trending basis vectors.
    scl : `float`, optional, default='None'
        An additional scaling factor for the CBVs.

    returns
    -------
    weights : `iterable`
        The weights assigned for each lightcurve data point.
    '''
    # pre-process basis
    nb,nobs = basis.shape
    B = np.matrix(basis.T)
    if scl == None: scl = np.ones(nb)
    Bnorm = np.multiply(B, scl)
    Bs = Bnorm.std()
    Bnorm /= Bs
    Bnorm = np.concatenate((Bnorm, np.ones((nobs,1))), axis=1)
    # array to store weights
    nobj = flux.shape[0]
    weights = np.zeros((nobj,nb))
    for iobj in np.arange(nobj): 
        # pre-process flux
        F = np.matrix(flux[iobj,:]).T
        l = np.isfinite(F)
        Fm = F.mean()
        Fs = F.std()
        Fnorm = (F - Fm) / Fs
        res = VBF(Bnorm, Fnorm)
        w, V, invV, logdetV, an, bn, E_a, L = res
        weights[iobj,:] = np.array(res[0][:-1]).flatten() * scl * Fs / Bs
    return weights

def apply_basis(weights, basis):
    '''Calculate the dot product between the weights and the CBVs.
    
    parameters
    ----------
    weights : `Iterable`
        The weights for each lightcurve data point.
    basis : `Iterable`
        The CBVs for each data point.
    
    returns
    -------
        dot_prod_res : `Iterable`
            The dot product between the weights and the basis vectors.
        corr: (nobj x nobs) correction to apply to light curves
    '''

    dot_prod_res = np.dot(weights, basis)
    return dot_prod_res

def fixed_nb(flux, cbv, nB = 4, use = None, doPlot = True):
    '''
    corrected_flux = fixed_nb(flux, basis, nB = 4, use = None, \
                              doPlot = True)
    Correct light curve for systematics using first nB CBVs.

    Inputs:
        flux: (1-D array) light curves 
        cbv: (2-D array) co-trending basis vectors trends
    Optional inputs:
        nB: number of CBVs to use (the first nB are used)
        use: boolean array, True for data points to use in evaluating correction, 
             False for data points to ignore (NaNs are also ignored)
        doPlot: set to False to suppress plot
    Outputs:
        corrected_flux: (same shape as flux) corrected light curves
        weights: (nB array) basis vector coefficients
    '''
    nobs = len(flux)    
    if cbv.shape[1] == nobs: cbv_ = cbv[:nB,:]
    else: cbv_ = cbv[:,:nB].T
    corrected_flux = np.copy(flux)
    l = np.isfinite(flux)
    if not use is None: l *= use
    weights = fit_basis(flux[l].reshape((1,l.sum())), cbv_[:,l])
    corr = apply_basis(weights, cbv_).reshape(flux.shape)
    corrected_flux = flux - corr
    if doPlot == True:
        pl.clf()
        x = np.arange(nobs)
        pl.plot(x, flux, '-', c = 'grey')
        pl.plot(x[l], flux[l], 'k-')
        pl.plot(x, corr, 'c-')
        pl.plot(x, corrected_flux, 'm-')
        pl.xlabel('Observation number')
        pl.xlabel('Flux')
    return corrected_flux, weights

def sel_nb(flux, cbv, nBmax = None, use = None):
    '''
    (nb_opt, flux_opt, weights_opt), (corr_flux_multi, weights_multi)
                = sel_nb(flux, basis, nBmax = 8, use = None)
    Correct light curve for systematics using upt to nB CBVs 
    (automatically select best number).

    Inputs:
        flux: (1-D array) light curves 
        cbv: (2-D array) co-trending basis vectors trends
    Optional inputs:
        nBmax: maximum number of CBVs to use (starting with the first)
        use: boolean array, True for data points to use in evaluating 
             correction, False for data points to ignore (NaNs are also ignored)
    Outputs:
        nBopt: automatically selected number of CBVs used (<= nBmax)
        corr_flux: (same shape as flux) corrected light curves
        weights: (nBopt array) basis vector coefficients
    '''
    nobs = len(flux)
    if cbv.shape[1] == nobs: cbv_ = np.copy(cbv)
    else: cbv_ = cbv.T
    if nBmax is None: nBmax = cbv.shape[0]
    else: cbv_ = cbv_[:nBmax,:]
        
    corr_flux = np.zeros(nobs)
    corr_flux_multi = np.zeros((nBmax,nobs))
    weights_multi = np.zeros((nBmax,nBmax))
    ran_multi = np.zeros(nBmax)
    sig_multi = np.zeros(nBmax)

    l = np.isfinite(flux)
    if not use is None: l *= use

    med_raw, ran_raw, sig_raw = medransig(flux[l])

    for i in range(nBmax):
        cbv_c = cbv_[:i+1,:]
        w_c = fit_basis(flux[l].reshape((1,l.sum())), cbv_c[:,l])
        w_ext = np.zeros(nBmax)
        w_ext[:i+1] = w_c
        weights_multi[i,:] = w_ext
        corr = apply_basis(w_c, cbv_c).reshape(flux.shape)
        c = flux - corr
        med, ran, sig = medransig(c[l])
        corr_flux_multi[i,:] = c - med + med_raw
        ran_multi[i] = ran
        sig_multi[i] = sig

    # Select the best number of basis functions
    # (smallest number that significantly reduces range)
    med_ran = np.median(ran_multi)
    sig_ran = 1.48 * np.median(abs(ran_multi - med_ran))
    jj = np.where(ran_multi < med_ran + 3 * sig_ran)[0][0]
    # Does that introduce noise? If so try to reduce nB till it doesn't
    while (sig_multi[jj] > 1.1 * sig_raw) and (jj > 0): jj -= 1

    nb_opt = jj + 1
    flux_opt = corr_flux_multi[jj,:].flatten()
    weights_opt = weights_multi[jj,:][:jj+1].flatten()
    ran_opt = ran_multi[jj]
    sig_opt = sig_multi[jj]
    return (nb_opt, flux_opt, weights_opt), \
      (corr_flux_multi, weights_multi)
      
      
      
      
def interpolate_cbv(cbv_file, lc, type_cbv='Single'):
    with fits.open(cbv_file) as hdul:
        if type_cbv == 'Single':
            data = hdul[1].data
        if type_cbv == 'Spike':
            data = hdul[2].data
        if type_cbv == 'Multi1':
            data = hdul[3].data
        if type_cbv == 'Multi2':
            data = hdul[4].data
        if type_cbv == 'Multi3':
            data = hdul[5].data
 
        time = data["TIME"]
        vectors = [v for v in data.names if "VECTOR" in v]
        v_new = np.zeros((len(lc), len(vectors)))
        for i, v in enumerate(vectors):
            x = np.interp(lc["time"], time, data[f'{v}'])
            v_new[:,i] = np.interp(lc["time"], time, data[f'{v}'])
    return v_new.T


def get_cbv_scc(scc, lc):
    with open('./cbv/curl_cbv.scr', 'r') as curl_file:
        lines = curl_file.readlines()
        cbv_comm = [l for l in lines if f'{scc[0]}-{scc[1]}-{scc[2]}' in l]
        cbv_comm = cbv_comm[0]
        cbv_file = cbv_comm.split(' ')[6]
        if os.path.exists(f'./cbv/{cbv_file}') is False:
            subprocess.run(cbv_comm, shell=True)
            subprocess.run(f'mv {cbv_file} ./cbv/', shell=True)
    interpolated_cbv = interpolate_cbv(f'./cbv/{cbv_file}', lc)
    corrected_flux, weights = sel_nb(np.array(lc['flux'].data), interpolated_cbv)    
    return corrected_flux, weights

