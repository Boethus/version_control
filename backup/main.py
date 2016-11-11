# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:06:54 2016

@author: pi
"""

import functions as f
pic = "samplesCropped/Stack0008.png"
#pic = "coinsCropped/1po.png"
print "loading picture",pic
l=f.identifyCoins(pic)
print l

money=0
for u in range(len(l)):
    if l[u][-2:]=="po":
        money+=int(l[u][:-2])
        print "tt"
    else:
        money+=int(l[u][:-2]) * 0.01
print "you have",money,"pounds"

    
