import time
import sys
import copy
import os

import numpy as np
from scipy import linalg
from PIL import Image, ImageFilter
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io
from scipy import stats

import pickle
from scipy.ndimage import gaussian_filter, median_filter
import glob
from pathlib import Path
import re
import ROOT


import argparse
parser = argparse.ArgumentParser(description='....py...')
parser.add_argument('-run',  metavar='alpide run number',        required=True,  help='alpide run number')
parser.add_argument('-dat',  metavar='EPICS date e.g. 20250518', required=True,  help='EPICS date e.g. 20250518')
parser.add_argument('-set',  metavar='EPICS dataset e.g. 21',    required=True,  help='EPICS dataset e.g. 21')
argus = parser.parse_args()
alpide_run = argus.run
date       = argus.dat
dataset    = argus.set




################################################################
################################################################

### EPICS time offest
NYEARS   = 20
NDAYS    = 5
NHOURS   = 0
NMINUTES = 1.569509
NSECONDS = NYEARS*365*24*60*60 + NDAYS*24*60*60 + NHOURS*60*60 + NMINUTES*60
print(f"EPICS time offset [s]: {NSECONDS} for NYEARS={NYEARS}, NDAYS={NDAYS}, NHOURS={NHOURS} and NMINUTES={NMINUTES}")

#############################################################
#############################################################



### 
NTRIM0 = 1325
NTRIM1 = 2400

### 
NTRIM_ALPIDE = 1

#############################################################
#############################################################


### pickle files
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def getfileslist(directory,pattern,suff):
    path = Path(os.path.expanduser(directory))
    ff = [str(file) for file in path.glob(pattern + '*' + suff)]
    ff.sort(key=natural_keys)
    return ff


datasetname = f"{date}_{dataset}"
directory   = f"../test_data/e320_prototype_beam_May2025_17-19/pickle/{datasetname}/"
pattern     = f"scalar_"
suffix      = ".pickle"
files       = getfileslist(directory,pattern,suffix)
print(f"Files:")
for fh in files: print(fh)


PV_VAL = {
    "PMT_3360":[],
    "PMT_3070":[],
    "PMT_3179":[],
    "PMT_3350":[],
    "PMT_3360":[],
}
PV_PID = {
    "PMT_3360":[],
    "PMT_3070":[],
    "PMT_3179":[],
    "PMT_3350":[],
    "PMT_3360":[],
}
PV_TIM = {
    "PMT_3360":[],
    "PMT_3070":[],
    "PMT_3179":[],
    "PMT_3350":[],
    "PMT_3360":[],
}
PV_TS = {
    "PMT_3360":[],
    "PMT_3070":[],
    "PMT_3179":[],
    "PMT_3350":[],
    "PMT_3360":[],
}
PV_LT = {
    "PMT_3360":[],
    "PMT_3070":[],
    "PMT_3179":[],
    "PMT_3350":[],
    "PMT_3360":[],
}
pvnames = list(PV_TIM.keys())


for fname in files:
    fh = open(fname, "rb")
    pv_data  = pickle.load(fh)
    fh.close()
    for key,vals in pv_data.items():
        # print(f"{len(vals)} items in {fname}")
        for pvname in pvnames:
            if(pvname.replace("PMT_","") not in key): continue
            for i,val in enumerate(vals):
                PV_VAL[pvname].append(val["value"])
                PV_PID[pvname].append(val["pid"])
                PV_TIM[pvname].append(val["SLAC_time"]+NSECONDS)
                PV_TS[pvname].append(val["ts"])
                PV_LT[pvname].append(val["lt"])

print(f"Total number of triggers: {len(PV_TIM[pvnames[0]])}")

def sort_array_pair(primary_arr,secondary_arr):
    # pair elements and sort based on primary_arr
    combined_arr = sorted(zip(primary_arr, secondary_arr))
    # unpack back into separate arrays
    sorted_primary_arr, sorted_secondary_arr = zip(*combined_arr)
    return sorted_primary_arr, sorted_secondary_arr

for pvname in pvnames:    
    prim_arr = PV_TS[pvname]
    scnd_arr = PV_VAL[pvname]
    srtd_prim_arr, sretd_scnd_arr = sort_array_pair(prim_arr,scnd_arr)
    PV_VAL[pvname] = sretd_scnd_arr
    
    prim_arr = PV_TS[pvname]
    scnd_arr = PV_PID[pvname]
    srtd_prim_arr, sretd_scnd_arr = sort_array_pair(prim_arr,scnd_arr)
    PV_PID[pvname] = sretd_scnd_arr
    
    prim_arr = PV_TS[pvname]
    scnd_arr = PV_TIM[pvname]
    srtd_prim_arr, sretd_scnd_arr = sort_array_pair(prim_arr,scnd_arr)
    PV_TIM[pvname] = sretd_scnd_arr
    
    prim_arr = PV_TS[pvname]
    scnd_arr = PV_LT[pvname]
    srtd_prim_arr, sretd_scnd_arr = sort_array_pair(prim_arr,scnd_arr)
    PV_LT[pvname] = sretd_scnd_arr
    
    ### finally the primary array
    PV_TS[pvname] = srtd_prim_arr


def restore_skipped_triggers(arrt,arrx,step=0.1,default_x=0,interpolate_x=False,increment=False):
    new_arrt = [arrt[0]]
    new_arrx = [arrx[0]]
    
    for i in range(1, len(arrt)):
        t_prev = arrt[i-1]
        t_curr = arrt[i]
        x_prev = arrx[i-1]
        x_curr = arrx[i]
        
        delta = t_curr - t_prev
        n_missing = int(round(delta / step)) - 1
        
        ### fill the missing
        for j in range(1, n_missing + 1):
            t_new = t_prev + j * step
            if(interpolate_x):
                x_new = x_prev + (x_curr - x_prev) * (t_new - t_prev) / delta ### linear interpolation
            else:
                if(increment):
                    x_new = x_prev + default_x
                else:
                    x_new = default_x
            new_arrt.append(t_new)
            new_arrx.append(x_new)
        
        ### add the current value
        new_arrt.append(t_curr)
        new_arrx.append(x_curr)
    
    return np.array(new_arrt), np.array(new_arrx)

for pvname in pvnames:
    arrt,PV_VAL[pvname] = restore_skipped_triggers(PV_TS[pvname],PV_VAL[pvname])
    arrt,PV_PID[pvname] = restore_skipped_triggers(PV_TS[pvname],PV_PID[pvname],default_x=36,increment=True)
    arrt,PV_TIM[pvname] = restore_skipped_triggers(PV_TS[pvname],PV_TIM[pvname],interpolate_x=True)
    arrt,PV_LT[pvname]  = restore_skipped_triggers(PV_TS[pvname],PV_LT[pvname],interpolate_x=True)

    ### finally the ts array
    PV_TS[pvname] = arrt
    

def trim_arr(arr,n0,n1):
    if(n0>0): arr = arr[n0:]
    if(n1>0): arr = arr[:-n1]
    return arr

for pvname in pvnames:
    PV_VAL[pvname] = trim_arr(PV_VAL[pvname],NTRIM0,NTRIM1)
    PV_PID[pvname] = trim_arr(PV_PID[pvname],NTRIM0,NTRIM1)
    PV_TIM[pvname] = trim_arr(PV_TIM[pvname],NTRIM0,NTRIM1)
    PV_TS[pvname]  = trim_arr(PV_TS[pvname],NTRIM0,NTRIM1)
    PV_LT[pvname]  = trim_arr(PV_LT[pvname],NTRIM0,NTRIM1)


################################################################
################################################################

data = {}
ALPIDES = {}
ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )

detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
planes    = [8,6,4,2,0]
plane2det = {8:"ALPIDE_0", 6:"ALPIDE_1", 4:"ALPIDE_2", 2:"ALPIDE_3", 0:"ALPIDE_4"}

tfile = ROOT.TFile(f"../test_data/e320_prototype_beam_May2025_17-19/runs/run_0000{alpide_run}/tree_Run{alpide_run}.root","READ")
ttree = tfile.Get("MyTree")
nevents = ttree.GetEntries()
print(f"nevents={nevents}")
doprint    = False
pixelsloop = False

### holder of the pixels
pixels = {}
for det in detectors:
    pixels.update({det:np.zeros(nevents)})

### read the pixels
trigger_number = np.zeros(nevents, dtype=np.uint64)
trigger_timbeg = np.zeros(nevents)
trigger_timend = np.zeros(nevents)
for ievt in range(nevents):
    ### get the event
    ttree.GetEntry(ievt)
    ### get the metadata
    trigger_number[ievt] = ttree.event.trg_n
    trigger_timbeg[ievt] = ttree.event.ts_begin/1e9
    trigger_timend[ievt] = ttree.event.ts_end/1e9
    ### get the staves
    staves = ttree.event.st_ev_buffer
    ### get the chips
    for istv in range(staves.size()):
        staveid  = staves[istv].stave_id
        chips    = staves[istv].ch_ev_buffer
        for ichp in range(chips.size()):
            chipid = chips[ichp].chip_id
            if(chipid not in planes): continue
            detector = plane2det[chipid]
            nhits    = chips[ichp].hits.size()
            if(doprint): print(f"chipid: {chipid} det: {detector} --> npixels: {nhits}")
            pixels[detector][ievt] = nhits
            if(pixelsloop):
                for ipix in range(nhits):
                    ix,iy = chips[ichp].hits[ipix]
                    if(doprint): print(f"   pixel[{ipix}]: ({ix},{iy})")

NTRIM_ALPIDE = 1
for det in detectors:
    ALPIDES.update( {det:pixels[det]} )
    ALPIDES[det] = ALPIDES[det][NTRIM_ALPIDE:]
data.update({"ALPIDES":ALPIDES})

trigger_number = trigger_number[NTRIM_ALPIDE:]
trigger_timbeg = trigger_timbeg[NTRIM_ALPIDE:]
trigger_timend = trigger_timend[NTRIM_ALPIDE:]

data.update({"ALPIDES_trg_number":trigger_number})
data.update({"ALPIDES_trg_timbeg":trigger_timbeg})
data.update({"ALPIDES_trg_timend":trigger_timend})
##################################################
##################################################



### plot
def plotter(name, pdf, arrx, arryL, arrsyR, xmin, xmax, xtitle, ytitleL="", ytitleR="", colL="blue", colR="red", logyL=True, logyR=True, lstyleL="-", lstyleR="-"):
    ### first page
    fig = plt.figure(figsize=(25, 5))
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    
    ax1.plot(arrx, arryL, linewidth=0.7, color=colL,alpha=0.7, linestyle=lstyleL)
    for i,arry in enumerate(arrsyR):
        ax2.plot(arrx, arry, linewidth=0.7, color=colR, alpha=1-i*0.15, linestyle=lstyleR)
    
    plt.locator_params(axis='x', nbins=10)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax1.grid(which='major', linestyle='-', linewidth='0.25', color='gray',alpha=0.5)
    ax1.grid(which='minor', linestyle=':', linewidth='0.25', color='gray',alpha=0.3)
    
    ax1.set_xlim(xmin,xmax)
    ax1.set_xlabel(xtitle)
    ax1.set_ylabel(ytitleL, color=colL)
    if(logyL): ax1.set_yscale("log")
    if(len(arrsyR)>0):
        ax2.set_xlim(xmin,xmax)
        ax2.set_xlabel(xtitle)
        ax2.set_ylabel(ytitleR, color=colR)
        if(logyR): ax2.set_yscale("log")
    
    plt.title(name, loc='center')
    
    pdf.savefig(fig)
    plt.close(fig)


def plot_arr_diff(name,pdf,arr,xtitle,xmin=-1,xmax=-1,nbins=500):
    d = []
    for i,t in enumerate(arr):
        if(i==0): continue
        d.append(t-arr[i-1])
    dmin = min(d)
    dmax = max(d)
    print(f"dmin={dmin}, dmax={dmax}")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), tight_layout=True)
    rng = (dmin*0.8,dmax*1.2) if(dmin>0) else (dmin*1.2,dmax*1.2)
    if(xmin!=-1 and xmax!=-1): rng = (xmin,xmax)
    hdt = ax.hist(d, bins=nbins, range=rng, rasterized=True)

    ax.set_xlim(rng[0],rng[1])
    ax.set_xlabel(xtitle)
    ax.set_ylabel('Triggers')
    ax.set_title(name)
    plt.locator_params(axis='x', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25)
    ax.set_yscale("log")
    ax.set_ylim(0.5,None)

    plt.tight_layout()
    
    pdf.savefig(fig)
    plt.close(fig)


def plot_time(name,pdf,tarr,xtitle):
    tmin = min(tarr)
    tmax = max(tarr)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5), tight_layout=True)
    hdt = ax.hist(tarr, bins=200, range=(tmin*0.8,tmax*1.2), rasterized=True)

    ax.set_xlim(tmin*0.9,tmax*1.1)
    ax.set_xlabel(xtitle)
    ax.set_ylabel('Triggers')
    ax.set_title(name)
    plt.locator_params(axis='x', nbins=10)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.grid(True,linewidth=0.25,alpha=0.25)
    ax.set_yscale("log")

    plt.tight_layout()
    
    pdf.savefig(fig)
    plt.close(fig)


##############################################################
##############################################################

def pad_arr_by_x(arrx,arry):
    ny = len(arry)
    nx = len(arrx)
    if(ny<nx): arry = np.pad(arry, (0, nx-ny), 'constant', constant_values=(0))
    return arry

with PdfPages(f'{datasetname}_RUN_{alpide_run}.pdf') as pdf:
    xarr   = PV_TIM["PMT_3070"]
    arryL  = PV_VAL["PMT_3070"]
    xmin   = xarr[0]
    xmax   = xarr[-1]
    plotter(f'{datasetname}_RUN_{alpide_run}', pdf, xarr, arryL,[], xmin, xmax, xtitle='SLAC time [s]', ytitleL='PMT 3070 [counts]', colL="red")

    arryL  = ALPIDES["ALPIDE_0"]
    xarr   = data["ALPIDES_trg_timbeg"]
    xmin   = xarr[0]
    xmax   = xarr[-1]
    plotter(f'{datasetname}_RUN_{alpide_run}', pdf, xarr, arryL,[], xmin, xmax, xtitle='EUDAQ time [s]', ytitleL='ALPIDE_0 Pixels', colL="black")
    
    xarr   = PV_TIM["PMT_3070"]
    arryL  = PV_PID["PMT_3070"]
    xmin   = xarr[0]
    xmax   = xarr[-1]
    plotter(f'{datasetname}_RUN_{alpide_run}', pdf, xarr, arryL,[], xmin, xmax, xtitle='SLAC time [s]', ytitleL='PID', colL="blue", logyL=True)
    
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, PV_PID["PMT_3070"],'PID shot-to-shot diff',nbins=100)
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, PV_PID["PMT_3070"],'PID shot-to-shot diff',xmin=-20,xmax=80, nbins=100)
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, PV_TIM["PMT_3070"],'SLAC_time shot-to-shot diff [s]')
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, PV_TS["PMT_3070"],'ts shot-to-shot diff [s]')
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, PV_LT["PMT_3070"],'lt shot-to-shot diff [s]')
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, data["ALPIDES_trg_timbeg"],'EUDAQ start time shot-to-shot diff [s]')
    plot_arr_diff(f'{datasetname}_RUN_{alpide_run}', pdf, data["ALPIDES_trg_timend"],'EUDAQ end time shot-to-shot diff [s]')
    
    proctime = np.subtract(data["ALPIDES_trg_timend"],data["ALPIDES_trg_timbeg"])
    plot_time(f'{datasetname}_RUN_{alpide_run}', pdf, proctime,'EUDAQ process time [s]')
    
    
    print(f'PV_VAL["PMT_3070"]  length: {len(PV_VAL["PMT_3070"])}')
    print(f'ALPIDES["ALPIDE_0"] length: {len(ALPIDES["ALPIDE_0"])}')


    
    
    
    
