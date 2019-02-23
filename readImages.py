# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:52:38 2019

@author: SARATHKUMAR
"""
import numpy as np
import cv2
import os
def readImages(path):
    Images=[]
    for fp in os.listdir(path):
        # reading images in GrayScale itself
        img=cv2.imread(os.path.join(path,fp),0)
        #converting them to float
        img=np.float32(img)/255.0
        #vectorizing it
        img=np.reshape(img,(img.size,1));
        Images.append(img)
    Images=np.asarray(Images)
    Images=np.reshape(Images,Images.shape[:-1]) # Nxp
    return Images
