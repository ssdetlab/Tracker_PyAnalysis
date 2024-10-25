#!/usr/bin/python
import os
import math
import subprocess
import time
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib
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
    # a parametric line is defined from 6 parameters but 4 are independent
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
    result0 = None
    result1 = None
    if(cfg["fast"]):
        result1 = minimize(chi2, initial_params, method=cfg["method0"],      args=(x,y,z, ex,ey,ez))
    else:
        result0 = minimize(chi2, initial_params, method=cfg["method1"][0], args=(x,y,z, ex,ey,ez)) ### first fit to get closer
        result1 = minimize(chi2, result0.x,      method=cfg["method0"][1], args=(x,y,z, ex,ey,ez)) ### second fit to finish
    ### get the chi^2 value and the number of degrees of freedom
    chisq = result1.fun
    ndof = 2*len(x) - len(initial_params)
    params  = result1.x
    success = result1.success
    # status  = result1.status
    # message = result1.message
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

    
# def plot_event(run,start,duration,evt,fname,clusters,tracks,chi2threshold=1.):
#     if(len(tracks)<1): return
#     ### turn interactive plotting off
#     plt.ioff()
#     matplotlib.use('Agg')
#     ### define the plot
#     fig = plt.figure(figsize=(8,4),frameon=False)
#     plt.title(f"Run {run}, Start: {start}, Duration: ~{duration} [h], Event {evt}", fontdict=None, loc='center', pad=None)
#     plt.box(False)
#     plt.axis('off')
#
#     ## the views
#     ax1 = fig.add_subplot(111, projection='3d')
#     ax1.set_xlabel("x [mm]")
#     ax1.set_ylabel("y [mm]")
#     ax1.set_zlabel("z [mm]")
#     ### the chips
#     L1verts = getChips()
#     ax1.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
#     # ax1.axes.set_aspect('equal')
#     ax1.set_box_aspect((1, 1, 1))
#
#     window = getWindowRealSpace()
#     ax1.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
#
#     ### print all clusters
#     clsx = []
#     clsy = []
#     clsz = []
#     for det in cfg["detectors"]:
#         for cluster in clusters[det]:
#             # clsx.append( cluster.xmm )
#             # clsy.append( cluster.ymm )
#             # clsz.append( cluster.zmm )
#             r = transform_to_real_space([cluster.xmm,cluster.ymm,cluster.zmm])
#             clsx.append( r[0] )
#             clsy.append( r[1] )
#             clsz.append( r[2] )
#     ax1.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
#     ### then the track
#     trkcols = ['r','b','m','c','y','k','g']
#     goodtrk = 0
#     for track in tracks:
#         if(track.chi2ndof>chi2threshold): continue
#         trkcol = trkcols[goodtrk]
#         goodtrk += 1
#
#         # x = track.points[0]
#         # y = track.points[1]
#         # z = track.points[2]
#         r = transform_to_real_space([track.points[0],track.points[1],track.points[2]])
#         x = r[0]
#         y = r[1]
#         z = r[2]
#
#         # Plot the points and the fitted line
#         x0,y0,z0 = line(cfg["zFirst"], track.params)
#         x1,y1,z1 = line(cfg["zLast"],  track.params)
#         x2,y2,z2 = line(getRealSpaceLimit("z","min")-(11.43+2.01)*10,  track.params)
#         r0 = transform_to_real_space([x0,y0,z0])
#         r1 = transform_to_real_space([x1,y1,z1])
#         r2 = transform_to_real_space([x2,y2,z2])
#         x0 = r0[0]
#         y0 = r0[1]
#         z0 = r0[2]
#         x1 = r1[0]
#         y1 = r1[1]
#         z1 = r1[2]
#         x2 = r2[0]
#         y2 = r2[1]
#         z2 = r2[2]
#
#         # plot the tracks clusters
#         ax1.scatter(x,y,z,s=0.92,c='r',marker='o')
#         ### plot the tracks lines
#         ax1.plot([x0, x1], [y0, y1], [z0, z1], c=trkcol, linewidth=0.7)
#         ### plot the extrapolated tracks lines
#         ax1.plot([x1, x2], [y1, y2], [z1, z2], c=trkcol, linewidth=0.7, linestyle='dashed')
#
#     ### add beampipe
#     us = np.linspace(0, 2 * np.pi, 100)
#     zs = np.linspace(getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max"), 100)
#     us, zs = np.meshgrid(us,zs)
#     Radius = 304.8 #180
#     xs = Radius * np.cos(us)
#     ys = Radius * np.sin(us)
#     ys = ys-Radius+5.8*10
#     ax1.plot_surface(xs, ys, zs, color='b',alpha=0.3)
#
#     ## world limits
#     # ax1.set_xlim(cfg["world"]["x"])
#     # ax1.set_ylim(cfg["world"]["y"])
#     # ax1.set_zlim(cfg["world"]["z"])
#     ax1.set_xlim([getRealSpaceLimit("x","min"),getRealSpaceLimit("x","max")])
#     ax1.set_ylim([getRealSpaceLimit("y","min"),getRealSpaceLimit("y","max")])
#     ax1.set_zlim([getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max")])
#
#
#     # ### add some text to ax1
#     # stracks = "tracks" if(goodtrk>1) else "track"
#     # ax1.text(+15,-15,0,f"{goodtrk} {stracks}", fontsize=7)
#     # for det in cfg["detectors"]:
#     #     z = cfg["rdetectors"][det][2]
#     #     n = len(clusters[det])
#     #     ax1.text(-30,-20,z,f"{det}", fontsize=7)
#     #     ax1.text(+15,+10,z,f"{n} clusters", fontsize=7)
#
#     ### change view of the 2nd plot
#     ### x-y view:
#     # ax1.elev = 90
#     # ax1.azim = 270
#     ax1.elev = 40
#     ax1.azim = 230 #270  # 270 is xz view, 0 is yz view, and -90 is xy view
#
#     ### finish
#     plt.savefig(fname)
#     plt.close(fig)



def plot_event(run,start,duration,evt,fname,clusters,tracks,chi2threshold=1.):
    if(len(tracks)<1): return
    ### turn interactive plotting off
    plt.ioff()
    matplotlib.use('Agg')
    ### define the plot
    fig = plt.figure(figsize=(8,10),frameon=False)
    plt.title(f"Run {run}, Start: {start}, Duration: ~{duration} [h], Event {evt}", fontdict=None, loc='center', pad=None)
    plt.box(False)
    plt.axis('off')
    
    ## the views
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_zlabel("z [mm]")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_zlabel("z [mm]")
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]")
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_zlabel("z [mm]")
    
    ### avoid ticks and lables for projections
    ax2.zaxis.set_label_position('none')
    ax2.zaxis.set_ticks_position('none')
    ax3.xaxis.set_label_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax4.yaxis.set_label_position('none')
    ax4.yaxis.set_ticks_position('none')
    
    
    ### the chips
    L1verts = getChips()
    ax1.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax2.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax3.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax4.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    # ax1.axes.set_aspect('equal')
    # ax2.axes.set_aspect('equal')
    # ax3.axes.set_aspect('equal')
    ax1.set_box_aspect((1, 1, 1))
    ax2.set_box_aspect((1, 1, 1))
    ax3.set_box_aspect((1, 1, 1))
    ax4.set_box_aspect((1, 1, 1))
    
    window = getWindowRealSpace()
    ax1.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    ax2.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    ax3.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    ax4.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    
    ### print all clusters
    clsx = []
    clsy = []
    clsz = []
    for det in cfg["detectors"]:
        for cluster in clusters[det]:
            # clsx.append( cluster.xmm )
            # clsy.append( cluster.ymm )
            # clsz.append( cluster.zmm )
            r = transform_to_real_space([cluster.xmm,cluster.ymm,cluster.zmm])
            clsx.append( r[0] )
            clsy.append( r[1] )
            clsz.append( r[2] )
    ax1.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ax2.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ax3.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ax4.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ### then the track
    trkcols = ['r','b','m','c','y','k','g']
    goodtrk = 0
    for track in tracks:
        if(track.chi2ndof>chi2threshold): continue
        trkcol = trkcols[goodtrk]
        goodtrk += 1
        
        # x = track.points[0]
        # y = track.points[1]
        # z = track.points[2]
        r = transform_to_real_space([track.points[0],track.points[1],track.points[2]])
        x = r[0]
        y = r[1]
        z = r[2]
        
        # Plot the points and the fitted line
        x0,y0,z0 = line(cfg["zFirst"], track.params)
        x1,y1,z1 = line(cfg["zLast"],  track.params)
        x2,y2,z2 = line(getRealSpaceLimit("z","min")-(11.43+2.01)*10,  track.params)
        r0 = transform_to_real_space([x0,y0,z0])
        r1 = transform_to_real_space([x1,y1,z1])
        r2 = transform_to_real_space([x2,y2,z2])
        x0 = r0[0]
        y0 = r0[1]
        z0 = r0[2]
        x1 = r1[0]
        y1 = r1[1]
        z1 = r1[2]
        x2 = r2[0]
        y2 = r2[1]
        z2 = r2[2]
        
        # plot the tracks clusters
        ax1.scatter(x,y,z,s=0.92,c='r',marker='o')
        ax2.scatter(x,y,z,s=0.92,c='r',marker='o')
        ax3.scatter(x,y,z,s=0.92,c='r',marker='o')
        ax4.scatter(x,y,z,s=0.92,c='r',marker='o')
        ### plot the tracks lines
        ax1.plot([x0, x1], [y0, y1], [z0, z1], c=trkcol, linewidth=0.7)
        ax2.plot([x0, x1], [y0, y1], [z0, z1], c=trkcol, linewidth=0.7)
        ax3.plot([x0, x1], [y0, y1], [z0, z1], c=trkcol, linewidth=0.7)
        ax4.plot([x0, x1], [y0, y1], [z0, z1], c=trkcol, linewidth=0.7)
        ### plot the extrapolated tracks lines
        ax1.plot([x1, x2], [y1, y2], [z1, z2], c=trkcol, linewidth=0.7, linestyle='dashed')
        ax2.plot([x1, x2], [y1, y2], [z1, z2], c=trkcol, linewidth=0.7, linestyle='dashed')
        ax3.plot([x1, x2], [y1, y2], [z1, z2], c=trkcol, linewidth=0.7, linestyle='dashed')
        ax4.plot([x1, x2], [y1, y2], [z1, z2], c=trkcol, linewidth=0.7, linestyle='dashed')
    
    ### add beampipe
    us = np.linspace(0, 2 * np.pi, 100)
    zs = np.linspace(getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max"), 100)
    us, zs = np.meshgrid(us,zs)
    Radius = 304.8 #180
    xs = Radius * np.cos(us)
    ys = Radius * np.sin(us)
    ys = ys-Radius+5.8*10
    # ax1.plot_surface(xs, ys, zs, color='b',alpha=0.3)
    ax2.plot_surface(xs, ys, zs, color='b',alpha=0.3)
    ax3.plot_surface(xs, ys, zs, color='b',alpha=0.3)
    # ax4.plot_surface(xs, ys, zs, color='b',alpha=0.3)
        
    ## world limits
    # ax1.set_xlim(cfg["world"]["x"])
    # ax1.set_ylim(cfg["world"]["y"])
    # ax1.set_zlim(cfg["world"]["z"])
    ax1.set_xlim([getRealSpaceLimit("x","min"),getRealSpaceLimit("x","max")])
    ax1.set_ylim([getRealSpaceLimit("y","min"),getRealSpaceLimit("y","max")])
    ax1.set_zlim([getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max")])
    
    ax2.set_xlim([getRealSpaceLimit("x","min"),getRealSpaceLimit("x","max")])
    ax2.set_ylim([getRealSpaceLimit("y","min"),getRealSpaceLimit("y","max")])
    ax2.set_zlim([getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max")])
    
    ax3.set_xlim([getRealSpaceLimit("x","min"),getRealSpaceLimit("x","max")])
    ax3.set_ylim([getRealSpaceLimit("y","min"),getRealSpaceLimit("y","max")])
    ax3.set_zlim([getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max")])
    
    ax4.set_xlim([getRealSpaceLimit("x","min"),getRealSpaceLimit("x","max")])
    ax4.set_ylim([getRealSpaceLimit("y","min"),getRealSpaceLimit("y","max")])
    ax4.set_zlim([getRealSpaceLimit("z","min"),getRealSpaceLimit("z","max")])
    
    
    # ### add some text to ax1
    # stracks = "tracks" if(goodtrk>1) else "track"
    # ax1.text(+15,-15,0,f"{goodtrk} {stracks}", fontsize=7)
    # for det in cfg["detectors"]:
    #     z = cfg["rdetectors"][det][2]
    #     n = len(clusters[det])
    #     ax1.text(-30,-20,z,f"{det}", fontsize=7)
    #     ax1.text(+15,+10,z,f"{n} clusters", fontsize=7)

    ### change view of the 2nd plot: 270 is xz view, 0 is yz view, and -90 is xy view
    ax1.elev = 40
    ax1.azim = 230
    ### x-y view:
    ax2.elev = 90
    ax2.azim = 270
    ### y-z view:
    ax3.elev = 0
    ax3.azim = 0
    ### x-z view:
    ax4.elev = 0
    ax4.azim = 270

    ### finish
    plt.savefig(fname)
    plt.close(fig)


