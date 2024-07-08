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
import utils
from utils import *


### similar to https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
 
def calculateSVDchi2(points, errors, direction, centroid):
    r1,r2 = r1r2(direction, centroid)
    x  = points[:,0]
    y  = points[:,1]
    z  = points[:,2]
    ex = errors[:,0]
    ey = errors[:,1]
    ## From CVR: https://gitlab.cern.ch/corryvreckan/corryvreckan/-/blob/master/src/objects/StraightLineTrack.cpp
    ## There's a 2D offset(x0,y0) and slope (dx,dy).
    ## Each hit provides two measurements.
    ## We also have the vertex constraint (the collimator) as another point
    ## ndof = 2*num_points - 4
    ndof = 2*len(points)-4
    chisq = 0
    for i in range(len(z)):
        xonline,yonline = xyofz(r1,r2,z[i])
        dx = xonline-x[i]
        dy = yonline-y[i]
        chisq += (dx*dx/(ex[i]*ex[i])+dy*dy/(ey[i]*ey[i]))
        # chisq += (dx*dx + dy*dy)
    return chisq,ndof


def fit_line_3d_SVD(points):
    """
    Fit a straight line to a set of 3D points using the least-squares method.
    similar to https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    """
    # Find the centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute the matrix A
    A = points - centroid
    # Compute the singular value decomposition of A
    U, s, Vt = np.linalg.svd(A)
    # The direction of the line is given by the first column of Vt
    direction = Vt[0]
    # The point on the line that is closest to the centroid is given by the centroid itself
    centroid
    return direction,centroid,s


def fit_3d_SVD(points,errors):
    # Fit a straight line to the points
    direction, centroid, s = fit_line_3d_SVD(points)
    # point = points[0] ### just the ploint where the line comes from
    point = centroid
    # goodness = s[1]
    chisq,ndof = calculateSVDchi2(points,errors,direction,centroid)
    return chisq,ndof,direction,centroid


def plot_3d_SVD(evt,points,direction,centroid,show=False):
    if(not show): return
    point = centroid
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
    ax.plot([point[0], point[0]+direction[0]*lineScaleUp],
            [point[1], point[1]+direction[1]*lineScaleUp],
            [point[2], point[2]+direction[2]*lineScaleUp], c='b')
    ax.plot([point[0], point[0]-direction[0]*lineScaleDn],
            [point[1], point[1]-direction[1]*lineScaleDn],
            [point[2], point[2]-direction[2]*lineScaleDn], c='b')
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    plt.xlim(world["x"])
    plt.ylim(world["y"])
    ax.set_zlim(world["z"])
    L1verts = getChips()
    ax.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=1, edgecolors='g', alpha=.20))
    ax.axes.set_aspect('equal') if(not isCVMFS) else ax.axes.set_aspect('auto')
    plt.title("SVD fit (evt #"+str(evt)+")", fontdict=None, loc='center', pad=None)
    plt.show()