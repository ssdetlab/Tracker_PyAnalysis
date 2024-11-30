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
parser.add_argument('-mult', metavar='multi run?',  required=False, help='is this a multirun? [0/1]')
argus = parser.parse_args()
configfile = argus.conf
ismutirun  = argus.mult if(argus.mult is not None and int(argus.mult)==1) else False
print(f"ismutirun={ismutirun}")

import config
from config import *
### must be called here (first) and only once!
init_config(configfile,False)


import utils
from utils import *
import objects
from objects import *
import evtdisp
from evtdisp import *
import counters
from counters import *
import selections
from selections import *

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)


if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    # print config once
    show_config()
    
    ### get all the files
    tfilenamein = ""
    files = []
    if(ismutirun):
        tfilenamein,files = make_multirun_dir(cfg["inputfile"],cfg["runnums"])
    else:
        tfilenamein = make_run_dirs(cfg["inputfile"])
        files = getfiles(tfilenamein)
    for f in files: print(f)
    
    ###. counters
    init_global_counters()
    Ndet = len(cfg["detectors"])
    
    ### save all events
    nevents = 0
    ntracks = 0
    for fpkl in files:
        suff = str(fpkl).split("_")[-1].replace(".pkl","")
        with open(fpkl,'rb') as handle:
            data = pickle.load(handle)
            for ievt,event in enumerate(data):
                if(ievt%50==0 and ievt>0): print(f"Reading event #{allevents}")
                nevents += 1
                
                counters_x_trg.append( event.trigger )
                append_global_counters()
                icounter = len(counters_x_trg)-1
                
                ### check errors
                if(len(event.errors)!=len(cfg["detectors"])): continue
                nErrors = 0
                for det in cfg["detectors"]: nErrors += len(event.errors[det])
                if(nErrors>0): continue
                
                ### check pixels
                if(len(event.pixels)!=len(cfg["detectors"])): continue
                n_pixels = 0
                pass_pixels = True
                for det in cfg["detectors"]:
                    npix = len( event.pixels[det] )
                    if(npix==0): pass_pixels = False
                    n_pixels += npix
                set_global_counter("Pixels/chip",icounter,n_pixels/Ndet)
                if(not pass_pixels): continue

                ### check clusters
                if(len(event.clusters)!=len(cfg["detectors"])): continue
                n_clusters = 0
                pass_clusters = True
                for det in cfg["detectors"]:
                    ncls = len(event.clusters[det])
                    if(ncls==0): pass_clusters = False
                    n_clusters += ncls
                set_global_counter("Clusters/chip",icounter,n_clusters/Ndet)
                if(not pass_clusters): continue
                ### check seeds
                n_seeds = len(event.seeds)
                set_global_counter("Seeds",icounter,n_seeds)
                if(n_seeds==0): continue

                ### check tracks
                n_tracks = len(event.tracks)
                if(n_tracks==0): continue

                good_tracks = []
                selected_tracks = []
                for track in event.tracks:
                    ### require chi2
                    if(track.chi2ndof>cfg["cut_chi2dof"]): continue
                    good_tracks.append(track)
                    ### require pointing to the pdc window, inclined up as a positron
                    if(not pass_slope_and_window_selection(track)): continue
                    selected_tracks.append(track)
                    ntracks += 1
                set_global_counter("Good Tracks",icounter,len(good_tracks))
                set_global_counter("Selected Tracks",icounter,len(selected_tracks))
                
                ### check for overlaps
                passing_tracks = remove_tracks_with_shared_clusters(selected_tracks)
                if(len(passing_tracks)!=len(selected_tracks)): print(f"nsel:{len(selected_tracks)} --> npas={len(passing_tracks)}")
                
                ### event displays
                if(len(good_tracks)>0):
                    fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_offline_{event.trigger}.pdf")
                    plot_event(event.meta.run,event.meta.start,event.meta.dur,event.trigger,fevtdisplayname,event.clusters,event.tracks,chi2threshold=cfg["cut_chi2dof"])
                    print(fevtdisplayname)

    print(f"Events:{nevents}, Tracks:{ntracks}")                
    
    
    gmax = -1e10
    gmin = +1e10
    for i,counter in enumerate(COUNTERS):
        mx = max(counters_y_val[counter])
        mn = min(counters_y_val[counter])
        gmax = mx if(mx>gmax) else gmax
        gmin = mn if(mn<gmin) else gmin
    gmin = gmin if(gmin>0) else 0.1
    gmax = gmax*10
    print(f"gmin={gmin}, gmax={gmax}")
    
    graphs = {}
    for i,counter in enumerate(COUNTERS):
        counter_name = counter.replace("/","_per_")
        gname = f"{counter}_vs_trg"
        graphs.update( {gname:ROOT.TGraph( len(counters_x_trg), counters_x_trg, counters_y_val[counter] )} )
        # graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
        graphs[gname].GetXaxis().SetLimits(counters_x_trg[0],counters_x_trg[-1])
        graphs[gname].SetLineColor(counters_cols[i])
        graphs[gname].SetMaximum(gmax)
        graphs[gname].SetMinimum(gmin)
    cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
    cnv.SetTicks(1,1)
    cnv.SetLogy()
    leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    mg = ROOT.TMultiGraph()
    for i,counter in enumerate(COUNTERS):
        counter_name = counter.replace("/","_per_")
        gname = f"{counter}_vs_trg"
        leg.AddEntry(graphs[gname],f"{counter}","l")
        mg.Add(graphs[gname])
    mg.Draw("al")
    leg.Draw("same")
    mg.SetTitle(f";Trigger number;Multiplicity")
    mg.SetMaximum(gmax)
    mg.SetMinimum(gmin)
    mg.GetXaxis().SetLimits(counters_x_trg[0],counters_x_trg[-1])
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("multiplicities_vs_triggers.pdf")

    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time} seconds')