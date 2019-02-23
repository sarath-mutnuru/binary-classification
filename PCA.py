# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:15:54 2019

@author: SARATHKUMAR
"""
import numpy as np
def PCA(X,n_c):
    
    ''' function to perform Principal Component Analysis 
     X is Nxp - each row is a data point
    
    Theory:
        
    We use SVD to get Eigen Values of covariance matrix of X 
    X=U*Sig*V.T
    X.T*X is covaraince matrix of features given that X is centred
    X.T=(Sig*V.T).T*U.T=(V*Sig.T*U.T)
    X.T*X=(V*Sig.T*U.T)*(U*Sig*V.T)
         = V*Sig.T*Sig*V.T  ---> Eigen Decomposition 
    => columns of V are Eigen Vectors of X.T*X 
    => columns of V are Principal Components of X  
    also fortunately np.linalg.SVD from numpy return Eig vecs in descending order of
    their correponding Eig values     
    '''
    
    # centre the data
    mean = np.mean( X, axis=0 )
    X    = X-mean
    U, S, Vt = np.linalg.svd( X )
    V = Vt.T
    PC=V[:,0:n_c]  # selecting first n_c columns
    return PC,mean
    