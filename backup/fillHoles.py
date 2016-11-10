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

img = cv2.imread("coins/all.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray=f.gaussian_convolution(gray,5)

gray=gray.astype(np.uint8)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
thresh = np.invert(thresh)
thresh_cop = thresh.copy()
"""

thresh = thresh/np.max(thresh)
thresh=thresh.astype(int)
"""
plt.figure()
plt.imshow(thresh_cop)
plt.title("thresholded image")
plt.colorbar()


"""Fills the holes by detecting the contours and then filling them"""

image,contour,hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

i=0
for cnt in contour:
    cv2.drawContours(image,contour,i,255,-1)
    i+=1
    
plt.figure()
plt.imshow(image)
plt.title("contour image")
plt.colorbar()
print np.max(thresh),np.min(thresh)
print type(thresh[0,0])
thresh=255*thresh.astype(np.uint8)
print np.max(thresh),np.min(thresh)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
res = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
plt.figure()
plt.imshow(res)

