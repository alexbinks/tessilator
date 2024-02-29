import scipy.linalg, scipy.special

'''
VBLinRegARD: Linear basis regression with automatic relevance priors
using Variational Bayes.

For more details on the algorithm see Apprendix of
Roberts, McQuillan, Reece & Aigrain, 2013, MNRAS, 354, 3639.

History:
2011: Translated by Thomas Evans from original Matlab code by Stephen J Roberts
2013: Documentation added by Suzanne Aigrain
'''

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
