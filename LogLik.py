# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:53:45 2019

@author: SARATHKUMAR
"""
import numpy as np
def LogLik(X,mu,cov):
    ''' function to calculate Log Likelihood '''
    X_del=X-mu
    prd = np.matmul(X_del,np.linalg.inv(cov))
    prd = np.matmul(prd,X_del.T)
    vals= np.diagonal(prd)
    
    LL = -0.5*(np.linalg.slogdet(cov)[-1]+vals)
    
    return LL