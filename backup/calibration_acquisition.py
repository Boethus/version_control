# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:57:50 2016

@author: aurelien
"""
try:
    import picamera
except ImportError:
    print "please install picamera first"
def get_calibration_samples():
    """prompts the user to place one sample of coin at the time below the camera
    and saves the pictures.
    """
    camera=picamera.PiCamera()
    
    coins = ["2po","1po","50pe","20pe","10pe","5pe","2pe","1pe"]
    for i in range(len(coins)):
        print "insert the ",coins[i]," coin and then press enter"
        raw_input()
        camera.capture(coins[i]+".png")