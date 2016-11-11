# -*- coding: utf-8 -*-
"""
This script reads the sample images located in folder and extracts their features of 
size and saturation values, stores them in a list and saves them under the names 
coin_areas.npy and coin_colours_values.npy
"""
import cv2
import matplotlib.pyplot as plt
import functions as f
import numpy as np
import colour_score

def learn_parameters():
    """learns the parameters and saves them in npy files"""
    save=True
    
    folder = "Calibration_ThuA/"
    list_coins = ["2po","1po","50pe","20pe","10pe","5pe","2pe","1pe"]
    sizes =[]
    bad_index=[]
    bad_labeled=[]
    colour_scores = []
    #Gets the image of each coin, finds it and comutes its area. 
    #In case of multiple match shows the litigious images
    for coin in list_coins:
        im = cv2.imread(folder+coin+".tif")
        im = im.astype(np.uint8)
        centroids,labeled,coin_labels,areas = f.findObjects(im)
        if len(coin_labels)!=1:
            print "Caution, the algorithm detected more than one coin"
            bad_index.append(coin)
            bad_labeled.append(labeled)
        sizes.append(areas[0])
        #print "Area of coin",coin," is ",areas[0]
        
        col_score = colour_score.colour_score(im, labeled, coin_labels[0],areas[0])
        print "Saturation value of coin",coin, "is",col_score
        colour_scores.append( col_score )
    print sizes
    print colour_scores
    
    min_color = min(colour_scores)
    range_color = max(colour_scores)-min(colour_scores)
    colour_separators = [min_color+range_color/3,min_color+2*range_color/3 ]
    print colour_separators
    for u in range(len(bad_index)):
        plt.figure()
        plt.imshow(bad_labeled[u])
        plt.title(bad_index[u])
        
    if save:
        np.save("coin_areas",sizes)
        np.save("coin_colours",colour_separators)
        np.save("coin_colours_values",colour_scores)
