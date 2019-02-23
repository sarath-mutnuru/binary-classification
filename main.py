# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:56:26 2019

@author: SARATHKUMAR
"""

import numpy as np
import os
from PCA import PCA as PCA_h
from LDAmodel import LDAmodel
from readImages import readImages

import pandas as pd
from  sklearn.metrics import f1_score


#%% Reading and Preparing Data
crnt_path=os.getcwd()

train_f=readImages(os.path.join(crnt_path,'face_train'))
train_nf=readImages(os.path.join(crnt_path,'nonface_train'))


X  = np.concatenate( ( train_f, train_nf ) )
y1 = np.ones( ( train_f.shape[ 0 ], 1 ) )
y2 = np.zeros( ( train_nf.shape[ 0 ], 1 ) )
y  = np.squeeze(np.concatenate( ( y1, y2 ) )).astype(int)

val_f =readImages(os.path.join(crnt_path,'face_test'))
val_nf=readImages(os.path.join(crnt_path,'nonface_test'))

X_val  = np.concatenate( ( val_f, val_nf ) )
y1     = np.ones( ( val_f.shape[ 0 ], 1 ) )
y2     = np.zeros( ( val_nf.shape[ 0 ], 1 ) )
y_val  = np.squeeze(np.concatenate( ( y1, y2 ) )).astype(int)
#%%

# Actual driver code


n_c=100
# =============================================================================
# from sklearn.decomposition import PCA
# pca      = PCA(n_components=n_c)
# X_pca    = pca.fit_transform(X)
# X_val_pca= pca.transform(X_val)
# =============================================================================

# PCA defined in the file
PC,mean=PCA_h(X,n_c)
X_pca=np.matmul(X-mean,PC)
X_val_pca=np.matmul(X_val-mean,PC) 


y_val_pred    = LDAmodel(X,y,X_val,y_val)

y_val_pred_pca= LDAmodel(X_pca,y,X_val_pca,y_val)

# =============================================================================
# from sklearn import linear_model
# classifier=linear_model.LogisticRegression(C=1e5,max_iter=1000,verbose=1)
# classifier.fit(X, y)
# y_val_pred = classifier.predict(X_val)
# 
# classifier_pca=linear_model.LogisticRegression(C=1e5,max_iter=1000,verbose=1)
# classifier_pca.fit(X_pca, y)
# y_val_pred_pca=classifier_pca.predict(X_val_pca)
# =============================================================================

print("................On Raw features.................")

print(pd.crosstab(y_val, y_val_pred, rownames=['Actual labels'], colnames=['Predicted Lables']))
print ("F1 score is ",f1_score(y_val,y_val_pred))


print("................On PCA using ",n_c ,"components .................")

print(pd.crosstab(y_val, y_val_pred_pca, rownames=['Actual labels'], colnames=['Predicted Lables']))
print ("F1 score is ",f1_score(y_val,y_val_pred_pca))








