#!/usr/bin/python
import os
import time
import datetime
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
import glob
from pathlib import Path
import re

import config
from config import *


def format_run_number(run):
    if(run<0 or run>=10000000):
        print(f"run number {run} is not supported. Quitting.")
        quit()
    if(run<10):                        return f"run_000000{run}"
    if(run>=10 and run<100):           return f"run_00000{run}"
    if(run>=100 and run<1000):         return f"run_0000{run}"
    if(run>=1000 and run<10000):       return f"run_000{run}"
    if(run>=10000 and run<100000):     return f"run_00{run}"
    if(run>=100000 and run<1000000):   return f"run_0{run}"
    if(run>=1000000 and run<10000000): return f"run_{run}" # assume no more than 9,999,999 events...
    return ""

def get_run_from_file(name):
    ## example: name = tree_09_02_2024_21_39_47_Run128.root
    words = name.split("_")
    word = words[-1]
    srun = word.replace("Run","").replace(".root","")
    run = int(srun)
    return run

def make_run_dirs(name):
    print(f"Got input file {name}")
    if(not os.path.isfile(name)):
        print(f"Input file {name} does not exist. Quitting.")
        quit()
    run    = get_run_from_file(name)
    srun   = format_run_number(run)
    paths  = name.split("/")
    infile = paths[-1]
    rundir = ""
    for i in range(len(paths)-1): rundir += paths[i]+"/"
    rundir += srun
    evtdir = rundir+"/event_displays"
    trgdir = rundir+"/beam_quality"
    filecopy = f"{rundir}/{infile}"
    if(not os.path.isdir(rundir)):
        print(f"Making dir {rundir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {rundir}")
    if(not os.path.isdir(evtdir)):
        print(f"Making dir {evtdir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {evtdir}")
    if(not os.path.isdir(trgdir)):
        print(f"Making dir {trgdir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {trgdir}")
    # if(not os.path.isfile(filecopy)):
    #     print(f"Copying input file {name} to run dir {rundir}")
    #     ROOT.gSystem.Exec(f"/bin/cp -f {name} {rundir}/")
    print(f"Always(!) copying input file {name} to run dir {rundir}")
    ROOT.gSystem.Exec(f"/bin/cp -f {name} {rundir}/")
    return filecopy


def make_multirun_dir(name,runs):
    print(f"Got input file {name}")
    if(not os.path.isfile(name)):
        print(f"Input file {name} does not exist. Quitting.")
        quit()
    run = get_run_from_file(name)
    if(run not in runs):
        print(f"Input run {run} is not in the run list. Quitting.")
        quit()
    paths  = name.split("/")
    rundir = ""
    for i in range(len(paths)-1): rundir += paths[i]+"/"
    ### make the list of files to be hadded
    infiles = ""
    pklfiles = []
    for r in runs:
        srun = format_run_number(r)
        fname = rundir+srun+"/tree_*_multiprocess_histograms.root "
        pname = rundir+srun+"/tree_*.pkl"
        if(not len(glob.glob(fname))<1):
            print(f"Input file {fname} does not exist. Quitting.")
            quit()
        infiles += fname+" "
        pklfiles.extend( glob.glob(pname) )
    ### get the combined rundir
    runs.sort()
    sruns = format_run_number(runs[0])
    for i,r in enumerate(runs):
        if(i==0): continue
        srun = str(r)
        sruns += ("-"+srun)
    rundir += sruns
    if(not os.path.isdir(rundir)):
        print(f"Making dir {rundir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {rundir}")
    ### hadd the file from scratch in that dir
    ftarget = f"{rundir}/tree_multiprocess_histograms.root"
    print(f"hadding input files:")
    ROOT.gSystem.Exec(f"hadd -f {ftarget} {infiles}")
    return ftarget, pklfiles


def get_human_timestamp(timestamp_ms,fmt="%d/%m/%Y, %H:%M:%S"):
    unix_timestamp = timestamp_ms/1000
    human_timestamp = time.strftime(fmt,time.localtime(unix_timestamp))
    return human_timestamp

def get_human_timestamp_ns(timestamp_ns,fmt="%d/%m/%Y, %H:%M:%S"):
    unix_timestamp = timestamp_ns/1e9
    human_timestamp = time.strftime(fmt,time.localtime(unix_timestamp))
    return human_timestamp

def get_run_length(run_start,run_end,fmt="hours"):
    run_start  = run_start/1000
    run_end    = run_end/1000
    run_length = datetime.datetime.fromtimestamp(run_end) - datetime.datetime.fromtimestamp(run_start)
    X = -1
    if(fmt=="hours"): X = 60*60
    if(fmt=="days"):  X = 60*60*24
    run_length_X = round(run_length.total_seconds()/X)
    return run_length_X


def transform_to_real_space(v):
    Rz = [[math.cos(cfg["thetaz"]),-math.sin(cfg["thetaz"]),0], [math.sin(cfg["thetaz"]),math.cos(cfg["thetaz"]),0], [0,0,1]]
    ### rotate x to y
    r = [0,0,0]
    r[0] = Rz[0][0]*v[0]+Rz[0][1]*v[1]+Rz[0][2]*v[2]
    r[1] = Rz[1][0]*v[0]+Rz[1][1]*v[1]+Rz[1][2]*v[2]
    r[2] = Rz[2][0]*v[0]+Rz[2][1]*v[1]+Rz[2][2]*v[2]
    ### introduce the offsets of the real space position of the detector (this is not the alignment offests!)
    r[0] += cfg["xOffset"]
    r[1] += cfg["yOffset"]
    r[2] += cfg["zOffset"]
    return r

def tilt_in_real_space(v):
    Rx = [[1,0,0],[0,math.cos(cfg["thetax"]),-math.sin(cfg["thetax"])], [0,math.sin(cfg["thetax"]),math.cos(cfg["thetax"])]]
    Ry = [[math.cos(cfg["thetay"]),0,math.sin(cfg["thetay"])], [0,1,0], [-math.sin(cfg["thetay"]),0,math.cos(cfg["thetay"])]]
    ### rotate around x
    vx = [0,0,0]
    vx[0] = Rx[0][0]*v[0]+Rx[0][1]*v[1]+Rx[0][2]*v[2]
    vx[1] = Rx[1][0]*v[0]+Rx[1][1]*v[1]+Rx[1][2]*v[2]
    vx[2] = Rx[2][0]*v[0]+Rx[2][1]*v[1]+Rx[2][2]*v[2]
    vy = [0,0,0]
    vy[0] = Ry[0][0]*vx[0]+Ry[0][1]*vx[1]+Ry[0][2]*vx[2]
    vy[1] = Ry[1][0]*vx[0]+Ry[1][1]*vx[1]+Ry[1][2]*vx[2]
    vy[2] = Ry[2][0]*vx[0]+Ry[2][1]*vx[1]+Ry[2][2]*vx[2]
    r = vy
    return r

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

def line(t, params):
    # a parametric line is defined from 6 parameters but 4 are independent
    # x0,y0,z0,z1,y1,z1 which are the coordinates of two points on the line
    # can choose z0 = 0 if line not parallel to x-y plane and z1 = 1;
    x = params[0] + params[1]*t
    y = params[2] + params[3]*t
    z = t
    return x,y,z

def get_pars_from_points(kA,kB,zA,zB):
    p1 = (kB-kA)/(zB-zA)
    # p0 = ((kB+kA)-p1*(zB+zA))/2.
    p0 = kA-p1*zA
    return p0,p1
    
def get_pars_from_centroid_and_direction(centroid,direction,isRealWorld=False):
    xA = centroid[0]
    xB = centroid[0]+direction[0]
    yA = centroid[1]
    yB = centroid[1]+direction[1]
    zA = centroid[2]
    zB = centroid[2]+direction[2]
    rA = transform_to_real_space( [xA,yA,zA] ) if(isRealWorld) else [xA,yA,zA]
    rB = transform_to_real_space( [xB,yB,zB] ) if(isRealWorld) else [xB,yB,zB]
    p0x,p1x = get_pars_from_points(rA[0],rB[0],rA[2],rB[2])
    p0y,p1y = get_pars_from_points(rA[1],rB[1],rA[2],rB[2])
    return [p0x,p1x,p0y,p1y]

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
    i = cfg["detectors"].index(detector)
    if(cfg["doVtx"]):
        if(len(points)==len(cfg["detectors"])+1): i = i+1 ### when the vertex is the first point in the points array
        else:
            print("In res_track2clusterErr")
            print(f"Problem with vertex or length of points. Quitting")
            quit()
    z = zpoints[i]
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
    if(cfg["doVtx"]):
        if(len(points)==len(cfg["detectors"])+1): i = i+1 ### when the vertex is the first point in the points array
        else:
            print("In res_track2cluster()")
            print(f"Problem with vertex or length of points. Quitting")
            quit()
    z  = zpoints[i]
    xonline,yonline = xyofz(r1,r2,z)
    # print(f"det={detector}: z={z}, xfit={xonline}, xpoint{x[i]}, yfit={yonline}, xpoint{y[i]}")
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


def get_track_point_at_z(track,z):
    x,y,z = line(z,track.params)
    r = transform_to_real_space( [x,y,z] )
    return r


def get_track_point_at_extremes(track):
    det0 = cfg["det_frst"]
    detN = cfg["det_last" ]
    z0 = cfg["rdetectors"][det0][2]
    zN = cfg["rdetectors"][detN][2]
    zW = -cfg["zOffset"] ### this is not 0 before transforming to the the real world
    zD = -(cfg["zOffset"]-cfg["zDipoleExit"])
    r0 = get_track_point_at_z(track,z0)
    rN = get_track_point_at_z(track,zN)
    rW = get_track_point_at_z(track,zW)
    rD = get_track_point_at_z(track,zD)
    ### tilt the detector around x and y
    r0 = tilt_in_real_space(r0)
    rN = tilt_in_real_space(rN)
    rW = tilt_in_real_space(rW)
    rD = tilt_in_real_space(rD)
    return r0,rN,rW,rD
    

def get_pdc_window_bounds():
    xWinL = cfg["xWindow"]-cfg["xWindowWidth"]/2.
    xWinR = cfg["xWindow"]+cfg["xWindowWidth"]/2.
    yWinB = cfg["yWindowMin"]
    yWinT = cfg["yWindowMin"]+cfg["yWindowHeight"]
    return xWinL,xWinR,yWinB,yWinT

def get_pdc_dipole_exit_bounds():
    xDipL = cfg["xDipoleExitMin"]
    xDipR = cfg["xDipoleExitMax"]
    yDipB = cfg["yDipoleExitMin"]
    yDipT = cfg["yDipoleExitMax"]
    return xDipL,xDipR,yDipB,yDipT



def getChips2D():
    chips = {}
    for det in cfg["detectors"]:
        x0,y0 = align(det,cfg["rdetectors"][det][0],cfg["rdetectors"][det][1])
        chips.update({ det: np.array([ [x0-cfg["chipX"]/2.,y0-cfg["chipY"]/2.],
                                       [x0-cfg["chipX"]/2.,y0+cfg["chipY"]/2.],
                                       [x0+cfg["chipX"]/2.,y0+cfg["chipY"]/2.],
                                       [x0+cfg["chipX"]/2.,y0-cfg["chipY"]/2.] ]) })
    return chips


def getChips(translatez=True):
    ### draw the chips: https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
    L1verts = []
    for det in cfg["detectors"]:
        xalgn,yalgn = align(det,cfg["rdetectors"][det][0],cfg["rdetectors"][det][1])
        ralgn = [xalgn, yalgn, cfg["rdetectors"][det][2]]
        # r = transform_to_real_space( [cfg["rdetectors"][det][0],cfg["rdetectors"][det][1],cfg["rdetectors"][det][2]] )
        r = transform_to_real_space( ralgn )
        x0 = r[0]
        y0 = r[1]
        z0 = r[2]
        ### (x,y) in the chip frame are (y,x) in the lab frame
        chipXLabFrame = cfg["chipY"]
        chipYLabFrame = cfg["chipX"]
        ### set the chips
        L1verts.append( np.array([ [x0-chipXLabFrame/2.,y0-chipYLabFrame/2.,z0],
                                   [x0-chipXLabFrame/2.,y0+chipYLabFrame/2.,z0],
                                   [x0+chipXLabFrame/2.,y0+chipYLabFrame/2.,z0],
                                   [x0+chipXLabFrame/2.,y0-chipYLabFrame/2.,z0] ]) )
    return L1verts


def getThetaAperture(yD):
    zF = -1e10
    zL = -1e10
    yMin = -1e10
    yMax = -1e10
    for det in cfg["detectors"]:
        xalgn,yalgn = align(det,cfg["rdetectors"][det][0],cfg["rdetectors"][det][1])
        ralgn = [xalgn, yalgn, cfg["rdetectors"][det][2]]
        r = transform_to_real_space( ralgn )
        x0 = r[0]
        y0 = r[1]
        z0 = r[2]
        ### (x,y) in the chip frame are (y,x) in the lab frame
        chipXLabFrame = cfg["chipY"]
        chipYLabFrame = cfg["chipX"]
        ### set the chips
        if(det==cfg["det_frst"]):
            zF = z0
            yMin = y0-chipYLabFrame/2.
            yMax = y0+chipYLabFrame/2.
        if(det==cfg["det_last"]):
            zL = z0
    zD = abs(cfg["zDipoleExit"])
    
    print(f"zF={zF}, zL={zL}")
    
    theta_min = math.atan((yMin-yD)/(zF+zD))
    theta_max = math.atan((yMax-yD)/(zL+zD))
    
    return theta_min, theta_max
        

def getWindowRealSpace():
    zWindow       = cfg["zWindow"]
    xWindowWidth  = cfg["xWindowWidth"]
    yWindowHeight = cfg["yWindowHeight"]
    xWindow       = cfg["xWindow"]
    yWindowMin    = cfg["yWindowMin"]
    window = np.array([ [xWindow-xWindowWidth/2., yWindowMin,               zWindow],
                        [xWindow-xWindowWidth/2., yWindowMin+yWindowHeight, zWindow],
                        [xWindow+xWindowWidth/2., yWindowMin+yWindowHeight, zWindow],
                        [xWindow+xWindowWidth/2., yWindowMin,               zWindow] ])
    return [window]


def getDipoleRealSpace():
    zDipole    = cfg["zDipoleExit"]
    xMin       = cfg["xDipoleExitMin"]
    xMax       = cfg["xDipoleExitMax"]
    yMin       = cfg["yDipoleExitMin"]
    yMax       = cfg["yDipoleExitMax"]
    dipole = np.array([ [xMin, yMin, zDipole],
                        [xMin, yMax, zDipole],
                        [xMax, yMax, zDipole],
                        [xMax, yMin, zDipole] ])
    return [dipole]


def InitCutflow():
    cutflow = {}
    for cut in cfg["cuts"]: cutflow.update({cut:0})
    return cutflow

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

### pickle files
def getfileslist(directory,pattern,suff):
    path = Path(os.path.expanduser(directory))
    ff = [str(file) for file in path.glob(pattern + '*' + suff)]
    ff.sort(key=natural_keys)
    return ff

### pickle files
def getfiles(tfilenamein):
    words = tfilenamein.split("/")
    directory = ""
    for w in range(len(words)-1):
        directory += words[w]+"/"
    strippedname = words[-1].split(".pkl")[0]
    words = strippedname.split("_")
    pattern = ""
    for w in range(len(words)):
        word = words[w].replace(".root","")
        pattern += word+"_"
    print("directory:",directory)
    print("pattern:",pattern)
    files = getfileslist(directory,pattern,".pkl")
    return files

