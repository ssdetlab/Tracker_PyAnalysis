#!/usr/bin/python
import multiprocessing as mp
# from multiprocessing.pool import ThreadPool
import time
import os
import os.path
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
from scipy.optimize import curve_fit,basinhopping
# from skspatial.objects import Line, Sphere
# from skspatial.plotting import plot_3d
import pickle
from pathlib import Path
import ctypes
import random

import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-ref', metavar='reference detector', required=False,  help='reference detector')
parser.add_argument('-mult', metavar='multi run?',  required=False, help='is this a multirun? [0/1]')
argus = parser.parse_args()
configfile = argus.conf
refdet     = argus.ref if(argus.ref is not None) else ""
ismutirun  = argus.mult if(argus.mult is not None and int(argus.mult)==1) else False

import config
from config import *
### must be called here (first) and only once!
init_config(configfile,False)


import utils
from utils import *
import svd_fit
from svd_fit import *
import chi2_fit
from chi2_fit import *
import hists
from hists import *

import objects
from objects import *
import pixels
from pixels import *
import clusters
from clusters import *
import truth
from truth import *
import noise
from noise import *
import candidate
from candidate import *

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)

### defined below as global
allhistos = {}


def getfileslist(directory,pattern,suff):
    files = Path( os.path.expanduser(directory) ).glob(pattern+'*'+suff)
    ff = []
    for f in files: ff.append(f)
    return ff


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


def fitSVD(event,dx,dy,theta,refdet=""):
    clsx  = {}
    clsy  = {}
    clsz  = {}
    clsdx = {}
    clsdy = {}
    
    ### initialize the misalignments per detector
    dX    = {}
    dY    = {}
    Theta = {}
    i = 0
    for det in cfg["detectors"]:
        if(refdet!="" and det==refdet): continue
        dX.update({det:dx[i]})
        dY.update({det:dy[i]})
        Theta.update({det:theta[i]})
        i += 1
    
    ### prepare the clusters with misalignments
    for det in cfg["detectors"]:
        x = event.clusters[det][0].xmm
        y = event.clusters[det][0].ymm
        z = event.clusters[det][0].zmm
        
        ### only for the non-reference detectors or all detectors?
        if(refdet!=""):
            if(det!=refdet):
                x,y = rotate(Theta[det],x,y)
                x = x+dX[det]
                y = y+dY[det]
        else:
            x,y = rotate(Theta[det],x,y)
            x = x+dX[det]
            y = y+dY[det]
        
        clsx.update({det:x})
        clsy.update({det:y})
        clsz.update({det:z})
        clsdx.update({det:event.clusters[det][0].dxmm})
        clsdy.update({det:event.clusters[det][0].dymm})
    vtx  = [cfg["xVtx"], cfg["yVtx"],  cfg["zVtx"]]  if(cfg["doVtx"]) else []
    evtx = [cfg["exVtx"],cfg["eyVtx"], cfg["ezVtx"]] if(cfg["doVtx"]) else []
    points_SVD,errors_SVD = SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
    chisq_SVD,ndof_SVD,direction_SVD,centroid_SVD = fit_3d_SVD(points_SVD,errors_SVD)
    
    dabs = 0
    for det in cfg["detectors"]:
        dx,dy = res_track2cluster(det,points_SVD,direction_SVD,centroid_SVD)
        dabs += math.sqrt(dx*dx + dy*dy)
    chi2ndof = chisq_SVD/ndof_SVD if(ndof_SVD>0) else -99999
    return chi2ndof,dabs,dx,dy


def init_params(axes,ndet2align,params):
    dxFinal    = [0]*ndet2align
    dyFinal    = [0]*ndet2align
    thetaFinal = [0]*ndet2align
    nparperdet = -1
    if(axes=="xytheta"):
        nparperdet = 3
        dxFinal    = params[0:ndet2align]
        dyFinal    = params[ndet2align:ndet2align*2]
        thetaFinal = params[ndet2align*2:ndet2align*3]
    elif(axes=="xy"):
        nparperdet = 2
        dxFinal    = params[0:ndet2align]
        dyFinal    = params[ndet2align:ndet2align*2]
    elif(axes=="xtheta"):
        nparperdet = 2
        dxFinal    = params[0:ndet2align]
        thetaFinal = params[ndet2align:ndet2align*2]
    elif(axes=="ytheta"):
        nparperdet = 2
        dyFinal    = params[0:ndet2align]
        thetaFinal = params[ndet2align:ndet2align*2]
    elif(axes=="x"):
        nparperdet = 1
        dxFinal = params[0:ndet2align]
    elif(axes=="y"):
        nparperdet = 1
        dyFinal = params[0:ndet2align]
    elif(axes=="theta"):
        nparperdet = 1
        thetaFinal = params[0:ndet2align]
    else:
        print("Unknown axes combination. Quitting.")
        quit()
    return dxFinal,dyFinal,thetaFinal,nparperdet


def fit_misalignment(events,ndet2align,refdet,axes):
    ### Define the objective function to minimize (the chi^2 function)
    ### similar to https://root.cern.ch/doc/master/line3Dfit_8C_source.html
    def avg_chi2(params,events):
        dx,dy,dt,nparperdet = init_params(axes,ndet2align,params)
        sum_dx = 0
        sum_dy = 0
        sum_dabs = 0
        sum_chi2 = 0
        for event in events:
            chisq,dabs,dX,dY = fitSVD(event,dx,dy,dt,refdet)
            sum_dabs += dabs
            sum_dx += dX
            sum_dy += dY
            sum_chi2 += chisq
        # return sum_chi2/len(events)
        return sum_dabs/len(events)
    
    nparperdet = -1
    if  (axes=="xytheta"):                                nparperdet = 3
    elif(axes=="xy" or axes=="xtheta" or axes=="ytheta"): nparperdet = 2
    elif(axes=="x"  or axes=="y"      or axes=="theta"):  nparperdet = 1
    else:
        print("Unknown axes combination. Quitting.")
        quit()
    
    ### https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due
    initial_params = [0]*(nparperdet*ndet2align)
    dx_range = [(cfg["alignmentbounds"]["dx"]["min"],    cfg["alignmentbounds"]["dx"]["max"])]*ndet2align    if("x"     in axes) else []
    dy_range = [(cfg["alignmentbounds"]["dy"]["min"],    cfg["alignmentbounds"]["dy"]["max"])]*ndet2align    if("y"     in axes) else []
    dt_range = [(cfg["alignmentbounds"]["theta"]["min"], cfg["alignmentbounds"]["theta"]["max"])]*ndet2align if("theta" in axes) else []
    ranges = []
    if("x"     in axes): ranges.extend(dx_range)
    if("y"     in axes): ranges.extend(dy_range)
    if("theta" in axes): ranges.extend(dt_range)
    range_params = tuple(ranges)
    # for n in range(len(initial_params)): initial_params[n] = random.uniform(range_params[n][0],range_params[n][1])
    print("initial_params:",initial_params)
    print("range_params:",range_params)
    ### https://stackoverflow.com/questions/52438263/scipy-optimize-gets-trapped-in-local-minima-what-can-i-do
    ### https://stackoverflow.com/questions/25448296/scipy-basin-hopping-minimization-on-function-with-free-and-fixed-parameters
    # result = minimize(avg_chi2, initial_params, method='TNC', args=(events), bounds=range_params, jac='2-point', options={'disp': True, 'finite_diff_rel_step': 0.0000001, 'accuracy': 0.001}) ### first fit to get closer
    # result = minimize(avg_chi2, initial_params, method='SLSQP',       args=(events), bounds=range_params, options={'disp': True ,'eps' : 1e-3})
    # result = minimize(avg_chi2, initial_params, method='Nelder-Mead', args=(events), bounds=range_params) ### first fit to get closer
    # result = minimize(avg_chi2, result.x,       method='Powell',      args=(events), bounds=range_params) ### second fit to finish
    # result = basinhopping(avg_chi2, initial_params, niter=50, minimizer_kwargs={"method": "L-BFGS-B", "args":(events,), "bounds":range_params})
    result = basinhopping(avg_chi2, initial_params, niter=cfg["naligniter"], minimizer_kwargs={"method":"SLSQP", "args":(events,), "bounds":range_params})
    
    ### get the chi^2 value and the number of degrees of freedom
    chisq = result.fun
    params  = result.x
    success = result.success
    return params,chisq,success


if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    # print config once
    show_config()
    if(refdet!="" and refdet not in cfg["detectors"]):
        print("Unknown detector:",refdet," --> quitting")
        quit()
    
    ### get all the files
    tfilenamein = ""
    files = []
    if(ismutirun):
        tfilenamein,files = make_multirun_dir(cfg["inputfile"],cfg["runnums"])
    else:
        tfilenamein = make_run_dirs(cfg["inputfile"])
        files = getfiles(tfilenamein)
    
    ###
    axes       = cfg["axes2align"]
    ndet2align = len(cfg["detectors"])-1 if(refdet!="") else len(cfg["detectors"])
    
    ### save all events
    events = []
    chisq0 = 0
    chisq0_werr = 0
    dabs0  = 0
    dX0    = 0
    dY0    = 0
    allevents = 0
    for fpkl in files:
        suff = str(fpkl).split("_")[-1].replace(".pkl","")
        with open(fpkl,'rb') as handle:
            data = pickle.load(handle)
            for event in data:
                if(allevents%50==0 and allevents>0): print("Reading event #",allevents)
                allevents += 1
                
                ### TODO: one track per event. need to adapt the code to take all tracks...
                if(len(event.tracks)!=1): continue
                
                chi2dof,dabs,dX,dY = fitSVD(event,[0]*ndet2align,[0]*ndet2align,[0]*ndet2align,refdet)
                chi2dof_werr = event.tracks[0].chi2ndof 
                # if(chi2dof>cfg["maxchi2align"]): continue
                if(chi2dof_werr>cfg["maxchi2align"]): continue
                events.append(event)
                chisq0 += chi2dof
                chisq0_werr += chi2dof_werr
                dabs0  += dabs
                dX0    += dX
                dY0    += dY
    ncollectedevents = len(events)
    print("Collected events:",ncollectedevents,", SVD fit chi2/dof=",chisq0,", Chi2Err fit chi2/dof=",chisq0_werr)
    if(ncollectedevents<5):
        print("Too few events collected ("+str(ncollectedevents)+") for the chi2/dof cut of maxchi2align=",cfg["maxchi2align"],"--> try to increase it in the config file.")
        print("Quitting")
        quit()
    chisq0 = chisq0/ncollectedevents
    dabs0  = dabs0/ncollectedevents
    print("Done collecting",ncollectedevents,"events with chisq0=",chisq0," and dabs0=",dabs0,". Now going to fit misalignments")
    
    ### fit
    params,result,success = fit_misalignment(events,ndet2align,refdet,axes)
    
    ### check
    chisq1 = 0
    dabs1  = 0
    dX1    = 0
    dY1    = 0
    allevents1 = 0
    dxFinal,dyFinal,thetaFinal,nparperdet = init_params(axes,ndet2align,params)
    for event in events:
        chi2dof,dabs,dX,dY = fitSVD(event,dxFinal,dyFinal,thetaFinal,refdet)
        chisq1 += chi2dof
        dabs1  += dabs
        dX1    += dX
        dY1    += dY
    chisq1 = chisq1/len(events)
    dabs1  = dabs1/len(events)
    dX1    = dX1/len(events)
    dY1    = dY1/len(events)
    
    ### sumarize
    print("\n----------------------------------------")
    print("Alignment axes:",axes)
    if(refdet!=""): print("Reference detector:",refdet)
    else:           print("No reference detector")
    print("Events used:",len(events),"out of",allevents)
    print("Success?",success)
    print("chi2:",chisq1,"(original:",chisq0,")")
    print("dabs:",dabs1,"(original:",dabs0,")")
    print("dx final   :",dxFinal)
    print("dy final   :",dyFinal)
    print("theta final:",thetaFinal)
    print("----------------------------------------\n")
    salignment = "misalignment  = "
    k = 0
    for det in cfg["detectors"]:
        if(det==refdet):
            salignment += det+":dx=0,dy=0,theta=0 "
        else:
            salignment += det+":dx="+str(dxFinal[k])+",dy="+str(dyFinal[k])+",theta="+str(thetaFinal[k])+" "
            k += 1
    print(salignment)
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('ֿֿ\nExecution time:', elapsed_time, 'seconds')
