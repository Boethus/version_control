# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:06:54 2016

@author: pi
"""

# Standard imports
import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
#import scipy.signal as sg
import matplotlib.pyplot as plt
import identifier
import colour_score

#Model parameters
minimum_size = 3000
maximum_size = 30000
gaussian_kernel_size = 5 #in pixels
value_threshold=240
coins = ["2po","1po","50pe","20pe","10pe","5pe","2pe","1pe"]

def gaussian_convolution(array,size):
    """Convolves array with a gaussian mask of size pixels."""
    x,y=np.ogrid[-size//2:size//2,-size//2:size//2]
    d2=x**2+y**2
    d2 = np.exp(-d2/(size/2)**2 )
    d2/=np.sum(d2)  #normalization
    out = signal.convolve2d(array,d2,mode="same")
    return out

    
def fillHoles(thresh):
    """fills the holes inside a thresholded image by finding and filling its contours"""
    image,contour,hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    i=0
    for cnt in contour:
        cv2.drawContours(image,contour,i,255,-1)
        i+=1
    return image
    
def centroid(array,value):
    """This function computes the centroid of the points equals to value in array.
    returns a list containing the coordinates of the centroid"""
    coordinates = np.where(array==value)
    number_of_points = len(coordinates[0])
    x = np.sum(coordinates[0])/number_of_points
    y = np.sum(coordinates[1])/number_of_points
    return [x,y]
    
def watershed_segmentation(img,thresh):
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
    nr_objects = np.max(markers)
    return markers,nr_objects
    
def labelPoint(array,point):
    size=5
    #use these min and max variables to avoid indexes out of range in the edges
    min_x=max(0,point[0]-size)
    min_y=max(0,point[1]-size)
    max_x=min(array.shape[0],point[0]+size)
    max_y=min(array.shape[1],point[1]+size)
    array[min_x:max_x,min_y:max_y]=np.ones((max_x-min_x,max_y-min_y))*5.5
    
def area(array,value):
    val = np.zeros(array.shape)
    val[array==value]=1
    return np.sum(val)

def coinsFromLabels(labeled_image,coin_labels):
    """Returns an image labeling only the coins from an image with different labels and the list of
    labels corresponding to coins"""
    
    new_image = np.zeros(labeled_image.shape)
    for u in range(len(coin_labels)):
        new_image[np.where(labeled_image==coin_labels[u])]=u+1
    return new_image

def findObjects(image):
    """image: bgr image. 
    Returns: centroid:  list containing the coordinates of 
    the centres of the different objects detected in the image.
    nr_objects:the number of objects found
    labeled: the image with the different objects labeled"""
    grey_im=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    grey_im = gaussian_convolution(grey_im,gaussian_kernel_size)
    grey_im = grey_im.astype(np.uint8)
    ret,thresh = cv2.threshold(grey_im,0,255,cv2.THRESH_OTSU)
    #Gaussian smoothing to eliminate the artefacts
    thresh = np.invert(thresh)
    thresh = fillHoles(thresh)
    #labeled,nr_objects = ndimage.label(thresh>value_threshold)
    labeled,nr_objects = watershed_segmentation(image,thresh)
    centroids=[]
    coin_labels=[]
    actual_number_objects =0
    areas=[]

    print nr_objects, "objects found"
    for i in range(nr_objects):
        size=area(labeled,i+1)
        if(size>minimum_size and size<maximum_size):
            #Determining the minimum enclosing circle to get circular coins
            points = np.where(labeled==i+1)
            points_vector = np.zeros( (len(points[0]),2) )
            points_vector[:,1] = points[0]
            points_vector[:,0] = points[1]
            points_vector=points_vector.astype(int)
            actual_number_objects+=1
            c=centroid(labeled,i+1)
            centroids.append(c)
            coin_labels.append(i+1)
            areas.append(size)
            
            #Set new value to labeled to only label the coins
  
    return centroids,labeled,coin_labels,areas

def identify_from_size(size,colour):
    """Indentifies a coin from its size and colour once identified"""
    sizes = np.load("coin_areas.npy")

    yellow_areas=sizes[0:2]
    grey_areas = sizes[2:6]
    red_areas = sizes[6:8]
    #Finding coin has the area closest from size
    if colour=="grey":
        grey_areas = [abs(x-size) for x in grey_areas]
        index = 2+ grey_areas.index(min(grey_areas))
        return coins[index]
    if colour=="red":
        red_areas = [abs(x-size) for x in red_areas]
        index = 6+ red_areas.index(min(red_areas))
        return coins[index]
    if colour=="yellow":
        yellow_areas = [abs(x-size) for x in yellow_areas]
        index = yellow_areas.index(min(yellow_areas))
        return coins[index]
        
def identifyColour(image,labeled,mask_value,area):
    score = colour_score.colour_score(image, labeled, mask_value,area)
    separators = np.load("coin_colours.npy")

    if score<separators[0]:
        return "grey"
    elif score<separators[1]:
        return "yellow"
    else:
        return "red"
    
def identifyCoins(image):
    im = cv2.imread(image)
    grey_im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY )

    #grey_im = cv2.equalizeHist(grey_im)
    centroids,labeled,coin_labels,areas = findObjects(im)    
    coins_list=[]
    
    sizes = np.load("coin_areas.npy")
    sizes_range = (np.max(sizes)-np.min(sizes))
    sizes_normalized = sizes/sizes_range
    
    colours = np.load("coin_colours_values.npy")
    colours_ranges=(np.max(colours)-np.min(colours))
    colours_normalized = colours/colours_ranges
    font = cv2.FONT_HERSHEY_SIMPLEX   
    for i in range(len(coin_labels)):
        color=identifyColour(im, labeled, coin_labels[i],areas[i])
        print color
        coin = identify_from_size(areas[0],color)
        print coin
        coins_list.append(coin)
        
        current_color_score = colour_score.colour_score(im,labeled,coin_labels[i],areas[i])/colours_ranges
        current_area_score = areas[i]/sizes_range
        res = [np.sqrt((x-current_area_score)**2+(y-current_color_score)**2) for (x,y) in zip(sizes_normalized,colours_normalized)]   
        index = np.argmin(res)
    
        coin = coins[index]
        
        (x,y)=centroid(labeled,coin_labels[i])
        cv2.putText(im,coin,(y,x), font, 1,(0,255,0),2,cv2.LINE_AA)


    
    cv2.imwrite("labeled_image12.png",im)
    labeled=coinsFromLabels(labeled,coin_labels)
    plt.figure()
    plt.imshow(labeled)
    plt.colorbar()
    return coins_list