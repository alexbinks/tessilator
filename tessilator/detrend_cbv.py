'''

Alexander Binks & Moritz Guenther, December 2024

Licence: MIT 2024

This module contains the functions needed to perform the co-trending basis vector corrections to the TESS lightcurves, if required. The functions were originally designed for Kepler-data analysis, and written by Suzanne Aigrain -> https://github.com/saigrain/CBVshrink
'''

###############################################################################
####################################IMPORTS####################################
###############################################################################
#Internal
import warnings
import inspect
import sys


# Third party 
import numpy as np
import os
import subprocess

from astropy.table import Table
from astropy.io import fits
import astropy.units as u

import pylab as pl


# Local application
import scipy.linalg, scipy.special
from .logger import logger_tessilator
###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__)


















def logdet(a):
    '''
    Compute log of determinant of matrix a using Cholesky decomposition
    '''
    # First make sure that matrix is symmetric:
    if np.allclose(a.T,a) == False:
        print('MATRIX NOT SYMMETRIC')
    # Second make sure that matrix is positive definite:
    eigenvalues = scipy.linalg.eigvalsh(a)
    if min(eigenvalues) <=0:
        print('Matrix is NOT positive-definite')
        print('   min eigv = %.16f' % min(eigenvalues))
    step1 = scipy.linalg.cholesky(a)
    step2 = np.diag(step1.T)
    out = 2. * np.sum(np.log(step2), axis=0)
    return out

def bayes_linear_fit_ard(X, y):
    '''
    Fit linear basis model with design matrix X to data y.
    
    Calling sequence:
    w, V, invV, logdetV, an, bn, E_a, L = bayes_linear_fit_ard(X, y)
    
    Inputs:
    X: design matrix
    y: target data
    
    Outputs
    w: basis function weights
    ***need to document the others!***
    '''
    # uninformative priors
    a0 = 1e-2
    b0 = 1e-4
    c0 = 1e-2
    d0 = 1e-4
    # pre-process data
    [N, D] = X.shape
    X_corr = X.T * X
    Xy_corr = X.T * y    
    an = a0 + N / 2.    
    gammaln_an = scipy.special.gammaln(an)
    cn = c0 + 1 / 2.    
    D_gammaln_cn = D * scipy.special.gammaln(cn)
    # iterate to find hyperparameters
    L_last = -sys.float_info.max
    max_iter = 500
    E_a = np.matrix(np.ones(D) * c0 / d0).T
    for iter in range(max_iter):
        # covariance and weight of linear model
        invV = np.matrix(np.diag(np.array(E_a)[:,0])) + X_corr   
        V = np.matrix(scipy.linalg.inv(invV))
        logdetV = -logdet(invV)    
        w = np.dot(V, Xy_corr)[:,0]
        # parameters of noise model (an remains constant)
        sse = np.sum(np.power(X*w-y, 2), axis=0)
        if np.imag(sse)==0:
            sse = np.real(sse)[0]
        else:
            print('Something went wrong')
        bn = b0 + 0.5 * (sse + np.sum((np.array(w)[:,0]**2) * np.array(E_a)[:,0], axis=0))
        E_t = an / bn 
        # hyperparameters of covariance prior (cn remains constant)
        dn = d0 + 0.5 * (E_t * (np.array(w)[:,0]**2) + np.diag(V))
        E_a = np.matrix(cn / dn).T
        # variational bound, ignoring constant terms for now
        L = -0.5 * (E_t*sse + np.sum(scipy.multiply(X,X*V))) + \
            0.5 * logdetV - b0 * E_t + gammaln_an - an * scipy.log(bn) + an + \
            D_gammaln_cn - cn * np.sum(scipy.log(dn))
        # variational bound must grow!
        if L_last > L:
            # if this happens, then something has gone wrong....
            file = open('ERROR_LOG','w')
            file.write('Last bound %6.6f, current bound %6.6f' % (L, L_last))
            file.close()
            raise Exception('Variational bound should not reduce - see ERROR_LOG')
            return
        # stop if change in variation bound is < 0.001%
        if abs(L_last - L) < abs(0.00001 * L):        
            break
        # print(L, L_last)
        L_last = L
    if iter == max_iter:    
        warnings.warn('Bayes:maxIter ... Bayesian linear regression reached maximum number of iterations.') 
    # augment variational bound with constant terms
    L = L - 0.5 * (N * np.log(2 * np.pi) - D) - scipy.special.gammaln(a0) + \
        a0 * np.log(b0) + D * (-scipy.special.gammaln(c0) + c0 * np.log(d0))
    return w, V, invV, logdetV, an, bn, E_a, L


def savgol_filter(t, f, window = 2.0, order = 2):
    '''The savgol filter'''
    n = len(t)
    f_sm = np.zeros(n) + np.nan
    for i in np.arange(n):
        l = (abs(t-t[i]) <= window/2.)
        if l.sum() == 0: continue
        tt = t[l]
        ff = f[l]
        p = np.polyval(np.polyfit(tt, ff, order), t[l])
        j = np.where(tt == t[i])[0]
        f_sm[i] = p[j]
    return f_sm


def block_mean(t, f, block_size = 13, block_size_min = None):
    '''calculate the block mean'''
    block_size = int(block_size)
    if block_size_min == None:
        block_size_min = block_size / 2 + 1
    n = len(t)
    dt = np.median(t[1:]-t[:-1])
    t_blocks = []
    f_blocks = []
    i = 0
    while t[i] < t[-1]:
        j = np.copy(i)
        while (t[j] - t[i]) < (block_size * dt):
            j+=1
            if j >= n: break
        if j >= (i + block_size_min):
            t_blocks.append(t[i:j].mean())
            f_blocks.append(f[i:j].mean())
        i = np.copy(j)
        if i >= n: break
    t_blocks = np.array(t_blocks)
    f_blocks = np.array(f_blocks)
    return t_blocks, f_blocks
    

def cdpp(time, flux, dfilt = 2.0, bl_sz = 13, exclude=None, plot = False):
    '''cdpp'''
    m = np.isfinite(time) & np.isfinite(flux)
    if exclude is not None:
        assert exclude.size == m.size, 'Exclusion mask size != time and flux array size'
        m &= ~exclude
    t,f = time[m], flux[m] / flux[m].mean()

    # filter out long-term variations
    f_sm = savgol_filter(t, f, dfilt)
    f_res = f - f_sm

    # exclude obvious outliers
    m2, s2 = f_res.mean(), f_res.std()
    l2 = abs(f_res-m2) < (5 * s2)

    # compute bin-averaged fluxes
    t_b, f_b = block_mean(t[l2], f_res[l2], block_size = bl_sz)
    cdpp = f_b.std() * 1e6

    if plot:
        pl.clf()
        pl.subplot(211)
        pl.plot(time, flux/flux[m].mean(), '.', c = 'grey', mec = 'grey')
        pl.plot(t, f, 'k.')
        pl.plot(t, f_sm, 'r-')
        pl.xlim(t.min(), t.max())
        pl.subplot(212)
        pl.plot(t, f_res * 1e6, '.', c = 'grey', mec = 'grey')
        pl.plot(t[l2], f_res[l2] * 1e6, 'k.')
        pl.plot(t_b, f_b * 1e6, 'b.')
        pl.xlim(t.min(), t.max())

    return cdpp

def medransig(array):
    '''calculate the medransig'''
    l = np.isfinite(array)
    med = np.median(array[l])
    norm = array / med
    norm_s = np.sort(norm)
    ng = l.sum()
    ran = norm_s[int(ng*0.95)] - norm_s[int(ng*0.05)]
    diff = norm[1:] - norm[:-1]
    l = np.isfinite(diff)
    sig = 1.48 * np.median(np.abs(diff[l]))
    return med, ran * 1e6, sig * 1e6























def fit_basis(flux, basis, scl=None):
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
        res = bayes_linear_fit_ard(Bnorm, Fnorm)
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



def fixed_nb(flux, cbv, nB=4, use=None, doPlot=True):
    '''Correct light curve for systematics using first nB CBVs.

    parameters
    ----------
    flux : `Iter`
        The 1-D array of light curves 
    cbv : `Iter`
        The 2-D array of co-trending basis vectors trends
    nb : `int`, optional, default=4
        The number of CBVs to use (the first nB are used)
    use : `bool`, optional, default=True
        True for data points to use in evaluating correction, False for data points to ignore (NaNs are also ignored)
    doPlot : `bool`, optional, default=True
        Choose whether to produce a plot of the CBV corrections.

    returns
    -------
    corrected_flux : `Iter`
        The corrected light curves, with the same shape as flux
    weights : `Iter`
        An nB-sized array containing the basis vector coefficients
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



def sel_nb(flux, cbv, nBmax=None, use=None):
    '''Correct light curve for systematics using upt to nB CBVs (automatically select best number).

    parameters
    ----------
    flux : `Iter`
        A 1-D array of light curves 
    cbv : `Iter`
        A 2-D array of co-trending basis vectors trends
    nBmax : `int`, optional, default=None
        The maximum number of CBVs to use (starting with the first)
    use : `bool`, optional, default=True
        True for data points to use in evaluating correction, False for data points to ignore (NaNs are also ignored)

    returns
    -------
    nBopt : `int`
        The automatically selected number of CBVs used (<= nBmax)
    corr_flux : `Iter`
        The corrected light curves (same shape as flux)
    weights : `Iter`
        The co-trending basis vector coefficients (same shape as nBopt).
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
    '''Selects the type of CBV correction and applies them to the lightcurve
    
    parameters
    ----------
    cbv_file : `str`
        The name of the CBV-file
    lc : `astropy.table.Table`
        The lightcurve data in astropy-tabulated format
    type_cbv : `str`, optional, default='Single'
        The type of CBV-corrections to make. Choose from "Single",
        "Spike", "Multi1", "Multi2" or "Multi3"
        
   returns
   -------
   v_fin : `astropy.table.Table`
       The tabulated weights from the CBV fits to be applied to the lightcurve    
    '''

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
    v_fin = v_new.T
    return v_fin



def get_cbv_scc(scc, lc):
    '''Run the CBV fits for a given lightcurve
    
    parameters
    ----------
    scc : `Iter`
        A 3-element list containing the sector, camera and CCD of the TESS image
    lc : `astropy.table.Table`
        The tabulated lightcurve data.

    returns
    -------
    corrected_flux : `list`
        The flux values after ungoing CBV corrections
    weights : `list`
        The weights applied to each lightcurve data point.
    '''
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


__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__], predicate = lambda f: inspect.isfunction(f) and f.__module__ == __name__)]
