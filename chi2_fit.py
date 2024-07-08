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


### similar to https://root.cern.ch/doc/master/line3Dfit_8C_source.html

def line(t, params):
    # a parametric line is define from 6 parameters but 4 are independent
    # x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
    # can choose z0 = 0 if line not parallel to x-y plane and z1 = 1;
    x = params[0] + params[1]*t
    y = params[2] + params[3]*t
    z = t
    return x,y,z


def distance2(params, xi,yi,zi, exi,eyi,ezi):
    # distance line point is D=|(xp-x0) cross ux|
    # where ux is direction of line and x0 is a point in the line (like t=0)
    xp = ROOT.Math.XYZVector(xi,yi,zi)
    x0 = ROOT.Math.XYZVector(params[0], params[2], 0. )
    x1 = ROOT.Math.XYZVector(params[0] + params[1], params[2] + params[3], 1. )
    u  = ROOT.Math.XYZVector(x1-x0).Unit()
    v  = (xp-x0).Cross(u)
    # d2 = v.Mag2()
    # d2 = v.X()*v.X()/(exi*exi) + v.Y()*v.Y()/(eyi*eyi) + v.Z()*v.Z()/(ezi*ezi)
    d2 = (v.X()*v.X() + v.Y()*v.Y() + v.Z()*v.Z())/(exi*exi + eyi*eyi + ezi*ezi)
    # d2 = (v.X()*v.X() + v.Y()*v.Y())/(exi*exi + eyi*eyi)
    return d2


def fit_line_3d_chi2err(x,y,z,ex,ey,ez):
    ### Define the objective function to minimize (the chi^2 function)
    ### similar to https://root.cern.ch/doc/master/line3Dfit_8C_source.html
    def chi2(params, x,y,z, ex,ey,ez):
        sum = 0
        for i in range(len(x)):
            d2 = distance2(params, x[i],y[i],z[i], ex[i],ey[i],ez[i])
            sum += d2
        return sum
    
    ### Perform the chi^2 fit using minimize
    ### https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due
    initial_params = [1,0,0,0]
    result = minimize(chi2, initial_params, method='Nelder-Mead', args=(x,y,z, ex,ey,ez)) ### first fit to get closer
    result = minimize(chi2, result.x,       method='Powell',      args=(x,y,z, ex,ey,ez)) ### second fit to finish
    ### get the chi^2 value and the number of degrees of freedom
    chisq = result.fun
    ndof = 2*len(x) - len(initial_params)
    params  = result.x
    success = result.success
    # status  = result.status
    # message = result.message
    # print(success,status,message)
    return params,chisq,ndof,success


def fit_3d_chi2err(points,errors):
    x = points[0]
    y = points[1]
    z = points[2]
    ex = errors[0]
    ey = errors[1]
    ez = errors[2]
    params,chisq,ndof,success = fit_line_3d_chi2err(x,y,z,ex,ey,ez)
    # Plot the points and the fitted line
    x0,y0,z0 = line(cfg["zFirst"], params)
    x1,y1,z1 = line(cfg["zLast"],  params)
    #TODO: need to check this:
    xm,ym,zm = line((cfg["zLast"]-cfg["zFirst"])/2., params) #TODO
    centroid  = [xm,ym,zm]                     #TODO
    direction = [x1-x0,y1-y0,z1-z0]            #TODO
    return chisq,ndof,direction,centroid,params,success


def plot_3d_chi2err(evt,points,params,show=False):
    if(not show): return
    x = points[0]
    y = points[1]
    z = points[2]
    # Plot the points and the fitted line
    x0,y0,z0 = line(cfg["zFirst"], params)
    x1,y1,z1 = line(cfg["zLast"],  params)
    #TODO: need to check this:
    xm,ym,zm = line((cfg["zLast"]-cfg["zFirst"])/2., params) #TODO
    centroid  = [xm,ym,zm]                     #TODO
    direction = [x1-x0,y1-y0,z1-z0]            #TODO
    # Plot the data and the fit
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.plot([x0, x1], [y0, y1], [z0, z1], c='b')            
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.xlim(cfg["world"]["x"])
    plt.ylim(cfg["world"]["y"])
    ax.set_zlim(cfg["world"]["z"])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    L1verts = getChips()
    ax.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=1, edgecolors='g', alpha=.20))
    ax.axes.set_aspect('equal') if(not cfg["isCVMFS"]) else ax.axes.set_aspect('auto')
    plt.title("Chi2 w/err fit (evt #"+str(evt)+")", fontdict=None, loc='center', pad=None)
    plt.show()