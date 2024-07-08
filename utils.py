#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT
# from ROOT import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit

import config
from config import *


def xofz(r1,r2,z):
   dz = r2[2]-r1[2]
   dx = r2[0]-r1[0]
   if(dz==0):
      print("ERROR in xofz: dx=0 --> r1[0]=%g,r2[0]=%g, r1[1]=%g,r2[1]=%g, r1[2]=%g,r2[2]=%g" % (r1[0],r2[0],r1[1],r2[1],r1[2],r2[2]))
      quit()
   a = dx/dz
   b = r1[0]-a*r1[2]
   x = a*z+b
   return x


def yofz(r1,r2,z):
   dz = r2[2]-r1[2]
   dy = r2[1]-r1[1]
   if(dz==0):
      print("ERROR in yofz: dz=0 --> r1[0]=%g,r2[0]=%g, r1[1]=%g,r2[1]=%g, r1[2]=%g,r2[2]=%g" % (r1[0],r2[0],r1[1],r2[1],r1[2],r2[2]))
      quit()
   a = dy/dz
   b = r1[1]-a*r1[2]
   y = a*z+b
   return y


def xyofz(r1,r2,z):
    x = xofz(r1,r2,z)
    y = yofz(r1,r2,z)
    return x,y


def r1r2(direction, centroid):
    r1 = [centroid[0], centroid[1], centroid[2] ]
    r2 = [centroid[0]+direction[0], centroid[1]+direction[1], centroid[2]+direction[2] ]
    return r1,r2


def rotate(theta,x,y):
    xr = x*math.cos(theta)-y*math.sin(theta)
    yr = x*math.sin(theta)+y*math.cos(theta)
    return xr,yr


def align(det,x,y):
    x,y = rotate(cfg["misalignment"][det]["theta"],x,y)
    x = x+cfg["misalignment"][det]["dx"]
    y = y+cfg["misalignment"][det]["dy"]
    return x,y


def res_track2clusterErr(detector, points, errors, direction, centroid):
    r1,r2 = r1r2(direction, centroid)
    x  = points[:,0]
    y  = points[:,1]
    ex = errors[:,0]
    ey = errors[:,1]
    zpoints = points[:,2]
    i  = cfg["detectors"].index(detector)
    if(len(points)==len(cfg["detectors"])+1): i = i+1 ### when the vertex is the first point in the points array
    z  = zpoints[i]
    xonline,yonline = xyofz(r1,r2,z)
    dx = (xonline-x[i])/ex[i]
    dy = (yonline-y[i])/ey[i]
    return dx,dy

def res_track2cluster(detector, points, direction, centroid):
    r1,r2 = r1r2(direction, centroid)
    x  = points[:,0]
    y  = points[:,1]
    zpoints = points[:,2]
    i  = cfg["detectors"].index(detector)
    if(len(points)==len(cfg["detectors"])+1): i = i+1 ### when the vertex is the first point in the points array
    z  = zpoints[i]
    xonline,yonline = xyofz(r1,r2,z)
    dx = xonline-x[i]
    dy = yonline-y[i]
    return dx,dy


def res_track2truth(detector, mcparticles, pdgIdMatch, points, direction, centroid):
    r1,r2 = r1r2(direction,centroid)
    zpoints = points[:,2]
    i = cfg["detectors"].index(detector)
    j = i
    if(len(points)==len(cfg["detectors"])+1): i = i+1 ### when the vertex is the first point in the points array
    z = zpoints[i]
    trupos = None
    for prt in mcparticles[detector]:
        if(abs(prt.pdg)!=pdgIdMatch): continue ### take only the target pdgId
        trupos = ROOT.Math.XYZPoint( prt.pos1.X(),prt.pos1.Y(),prt.pos1.Z() )        
        break ### take only the first mcparticle that matches
    if(trupos is None): return -9999,-9999
    xtru = trupos.X()
    ytru = trupos.Y()
    xonline,yonline = xyofz(r1,r2,z)
    dx = xonline-xtru
    dy = yonline-ytru
    return dx,dy


def res_track2vertex(vertex, direction, centroid):
    r1,r2 = r1r2(direction, centroid)
    z  = vertex[2]
    xonline = xofz(r1,r2,z)
    yonline = yofz(r1,r2,z)
    dx = xonline-vertex[0]
    dy = yonline-vertex[1]
    return dx,dy


def getChips():
    ### draw the chips: https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
    L1verts = []
    for det in cfg["detectors"]:
        x0 = cfg["rdetectors"][det][0]
        y0 = cfg["rdetectors"][det][1]
        z0 = cfg["rdetectors"][det][2]
        L1verts.append( np.array([ [x0-cfg["chipX"]/2.,y0-cfg["chipY"]/2.,z0],
                                   [x0-cfg["chipX"]/2.,y0+cfg["chipY"]/2.,z0],
                                   [x0+cfg["chipX"]/2.,y0+cfg["chipY"]/2.,z0],
                                   [x0+cfg["chipX"]/2.,y0-cfg["chipY"]/2.,z0] ]) )
    return L1verts


def InitCutflow():
    cutflow = {}
    for cut in cfg["cuts"]: cutflow.update({cut:0})
    return cutflow


