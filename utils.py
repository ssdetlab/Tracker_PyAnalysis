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
    filecopy = f"{rundir}/{infile}"
    if(not os.path.isdir(rundir)):
        print(f"Making dir {rundir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {rundir}")
    if(not os.path.isdir(evtdir)):
        print(f"Making dir {evtdir}")
        ROOT.gSystem.Exec(f"/bin/mkdir -p {evtdir}")
    if(not os.path.isfile(filecopy)):
        print(f"Copying input file {name} to run dir {rundir}")
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
    for r in runs:
        srun = format_run_number(r)
        fname = rundir+srun+"/tree_*_multiprocess_histograms.root "
        if(not len(glob.glob(fname))<1):
            print(f"Input file {fname} does not exist. Quitting.")
            quit()
        infiles += fname+" "
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
    return ftarget


def get_human_timestamp(timestamp_ms,fmt="%d/%m/%Y, %H:%M:%S"):
    unix_timestamp = timestamp_ms/1000
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


