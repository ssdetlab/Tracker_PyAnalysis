#!/usr/bin/python
import os
import math
import array
import numpy as np

import objects
from objects import *

def RecursiveClustering(cluster_pixels,pivot,pixels):
   if(pivot in pixels):
       cluster_pixels.append(pivot) ## add this pixel to the cluster
       pixels.remove(pivot)         ## kill pixel from live pixels list
   for pixel in pixels[:]:
       dx = abs(pixel.x-pivot.x)
       dy = abs(pixel.y-pivot.y)
       if((dx+dy)<=2 and dx<=1 and dy<=1):
           nextpivot = pixel
           RecursiveClustering(cluster_pixels,nextpivot,pixels)


def GetAllClusters(pixels,det):
    clusters  = []
    positions = []
    while len(pixels)>0: ## loop as long as there are live pixels in the list
        pixel = pixels[0] ## this is the pivot pixel for the cluster recursion
        cluster_pixels = []
        RecursiveClustering(cluster_pixels,pixel,pixels)
        cluster = Cls(cluster_pixels,det)
        clusters.append( cluster )
    return clusters