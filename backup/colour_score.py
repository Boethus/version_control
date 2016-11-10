# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:31:02 2016

@author: pi
"""

import cv2
import numpy as np

FILE_NAME = "inFocus pictures/6.png"
srcImg = cv2.imread(FILE_NAME)

def colour_score(BGRImg, segmentMask, mask_value,area):
    # segmentMask must contain 0s for background and mask_value for object pixels
    # Equalize histogram of the color camera image to compensate for occasional poor lighting conditions
    spRadius = 3
    colRadius = 3
    # Apply pyramidal mean shift filtering to get rid of flares and patterns
    filteredBGR = cv2.pyrMeanShiftFiltering(BGRImg, spRadius, colRadius)
    # Convert image from BRG into HSV colourspace.
    hsvImg = cv2.cvtColor(filteredBGR, cv2.COLOR_BGR2HSV)
    # Extract the saturation of coin pixels.
    saturation= hsvImg[:,:,1]
    #saturation = cv2.equalizeHist(saturation)
    bool_array= np.where(segmentMask != mask_value)
    saturation[bool_array]=0
    # Return mean saturation for coin.
    mean_saturation = np.sum(saturation)/area
    print mean_saturation
    return mean_saturation
    