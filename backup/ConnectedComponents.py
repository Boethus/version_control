#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 23:45:01 2016

@author: inhuszar
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import functions as f
from scipy import ndimage as ndi


#print thresh.shape
#ret,thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_OTSU)

"""#Noise removal


opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)"""

img = cv2.imread("coins/all.png")
thresh = cv2.imread("thresholdedimg.png",0)
thresh=thresh/np.max(thresh)
thresh=thresh.astype(np.uint8)


kernel = np.ones((3,3), np.uint8)
#Dilate to find area that is 100% sure background
sure_bg = cv2.dilate(thresh, kernel, iterations=3)

#Erode to find are that pertains 100% sure to coins
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
ret, sure_coins = cv2.threshold(dist_transform, 0.01*dist_transform.max(), 255, 0)

#Find indetermined area that contains borders
sure_coins = np.uint8(sure_coins)
indetermined = cv2.subtract(sure_bg, sure_coins)
                                                 
#Marker labelling
ret, markers = cv2.connectedComponents(sure_coins)

#Add one to all labels so that sure background is not 0 but 1.
markers = markers + 1

#Mark the region of unknown with zero
markers[indetermined == 255] = 0

markers = cv2.watershed(img, markers)


plt.figure()
plt.imshow(markers)