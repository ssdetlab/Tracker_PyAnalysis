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
ROOT.gStyle.SetOptStat(0)
# ROOT.gStyle.SetPalette(ROOT.kRust)
# ROOT.gStyle.SetPalette(ROOT.kSolar)
# ROOT.gStyle.SetPalette(ROOT.kInvertedDarkBodyRadiator)
ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
# ROOT.gStyle.SetPalette(ROOT.kRainbow)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.16)

B  = cfg["fDipoleTesla"]
LB = cfg["zDipoleLenghMeters"]
mm2m = 1e-3

def getP(r0,rN,rD):
    # tan_theta = (rN[1]-r0[1])/(rN[2]-r0[2])
    dExit = rD[1]*mm2m
    theta = 2*math.atan(dExit/LB)
    # tan_theta = (rN[1]-r0[1])/(rN[2]-r0[2])
    # p = 0.3 * B * (LB/tan_theta + dExit)
    p = (0.3 * B * LB)/math.sin(theta)
    return p if(dExit>0) else -999


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
    
    ### counters
    init_global_counters()
    Ndet = len(cfg["detectors"])
    
    ### some histos
    
    histos = {}
    histos.update({ "hD_before_cuts": ROOT.TH2D("hD_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hD_after_cuts":  ROOT.TH2D("hD_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hW_before_cuts": ROOT.TH2D("hW_before_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    histos.update({ "hW_after_cuts":  ROOT.TH2D("hW_after_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    histos.update({ "hTanTheta_yz":   ROOT.TH1D("hTanTheta_yz",";tan(#theta_{yz});Tracks",100,0,0.05)})
    histos.update({ "hTanTheta_xz":   ROOT.TH1D("hTanTheta_xz",";tan(#theta_{xz});Tracks",100,-0.01,0.01)})
    histos.update({ "hTheta_yz":      ROOT.TH1D("hTheta_yz",";#theta_{yz} [rad];Tracks",100,0,0.1)})
    histos.update({ "hTheta_xz":      ROOT.TH1D("hTheta_xz",";#theta_{xz} [rad];Tracks",100,-0.01,0.01)})
    histos.update({ "hdExit":         ROOT.TH1D("hdExit",";d_{exit} [mm];Tracks",120,-70,+90)})
    histos.update({ "hP":             ROOT.TH1D("hP",";p [GeV];Tracks",100,0,10)})
    
    dipole = ROOT.TPolyLine()
    xMinD = cfg["xDipoleExitMin"]
    xMaxD = cfg["xDipoleExitMax"]
    yMinD = cfg["yDipoleExitMin"]
    yMaxD = cfg["yDipoleExitMax"]    
    dipole.SetNextPoint(xMinD,yMinD)
    dipole.SetNextPoint(xMinD,yMaxD)
    dipole.SetNextPoint(xMaxD,yMaxD)
    dipole.SetNextPoint(xMaxD,yMinD)
    dipole.SetNextPoint(xMinD,yMinD)
    dipole.SetLineColor(ROOT.kBlue)
    dipole.SetLineWidth(1)
    
    window = ROOT.TPolyLine()
    xMinW = -cfg["xWindowWidth"]/2.
    xMaxW = +cfg["xWindowWidth"]/2.
    yMinW = cfg["yWindowMin"]
    yMaxW = cfg["yWindowMin"]+cfg["yWindowHeight"]
    window.SetNextPoint(xMinW,yMinW)
    window.SetNextPoint(xMinW,yMaxW)
    window.SetNextPoint(xMaxW,yMaxW)
    window.SetNextPoint(xMaxW,yMinW)
    window.SetNextPoint(xMinW,yMinW)    
    window.SetLineColor(ROOT.kBlue)
    window.SetLineWidth(1)
    
    ### save all events
    nevents = 0
    ntracks = 0
    for fpkl in files:
        suff = str(fpkl).split("_")[-1].replace(".pkl","")
        with open(fpkl,'rb') as handle:
            data = pickle.load(handle)
            for ievt,event in enumerate(data):
                # print(f"Reading event #{ievt}, trigger:{event.trigger}, ts:[{get_human_timestamp_ns(event.timestamp_bgn)}, {get_human_timestamp_ns(event.timestamp_end)}]")
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
                set_global_counter("Track Seeds",icounter,n_seeds)
                if(n_seeds==0): continue

                ### check tracks
                n_tracks = len(event.tracks)
                if(n_tracks==0): continue

                good_tracks = []
                acceptance_tracks = []
                for track in event.tracks:
                    
                    ### require max cluster and chi2
                    if(track.maxcls>cfg["cut_maxcls"]):    continue
                    if(track.chi2ndof>cfg["cut_chi2dof"]): continue
                    good_tracks.append(track)
                    
                    r0,rN,rW,rD = get_track_point_at_extremes(track)
                    tan_theta_yz = +track.params[1] ### the slope p1x transformed to real space (stays as is)
                    tan_theta_xz = -track.params[3] ### the slope p2x transformed to real space (gets minus sign)
                    theta_yz = math.atan(tan_theta_yz)
                    theta_xz = math.atan(tan_theta_xz)
                    
                    dExit = rD[1]*mm2m
                    theta_yz = 2*math.atan(dExit/LB)
                    print(f"dExit={dExit}, LB={LB}, theta_yz={theta_yz}")
                    
                    p = getP(r0,rN,rD)
                    
                    histos["hD_before_cuts"].Fill(rD[0],rD[1])
                    histos["hW_before_cuts"].Fill(rW[0],rW[1])
                    
                    ### require pointing to the pdc window, inclined up as a positron
                    if(not pass_geoacc_selection(track)): continue
                    
                    histos["hD_after_cuts"].Fill(rD[0],rD[1])
                    histos["hW_after_cuts"].Fill(rW[0],rW[1])
                    histos["hTheta_yz"].Fill(theta_yz)
                    histos["hTanTheta_yz"].Fill(tan_theta_yz)
                    histos["hTheta_xz"].Fill(theta_xz)
                    histos["hTanTheta_xz"].Fill(tan_theta_xz)
                    histos["hdExit"].Fill(rD[1])
                    if(p>0): histos["hP"].Fill(p)
                    acceptance_tracks.append(track)
                    ntracks += 1
                    
                    # print(f"tan(theta_yz)={theta_yz}, tan(theta_xz)={theta_xz}, track pars={track.params}")
                
                ### the graph of the good tracks
                set_global_counter("Good Tracks",icounter,len(good_tracks))
                
                ### check for overlaps
                selected_tracks = remove_tracks_with_shared_clusters(acceptance_tracks)
                if(len(selected_tracks)!=len(acceptance_tracks)): print(f"nsel:{len(acceptance_tracks)} --> npas={len(selected_tracks)}")
                set_global_counter("Selected Tracks",icounter,len(selected_tracks))
                
                ### event displays
                if(len(good_tracks)>0):
                    fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_offline_{event.trigger}.pdf")
                    plot_event(event.meta.run,event.meta.start,event.meta.dur,event.trigger,fevtdisplayname,event.clusters,event.tracks,chi2threshold=cfg["cut_chi2dof"])
                    print(fevtdisplayname)

    print(f"Events:{nevents}, Tracks:{ntracks}")                
    
    ### plot the counters
    plot_counters()

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hD_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hD_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("cnv_dipole_window.pdf(")
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hW_before_cuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hW_after_cuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("cnv_dipole_window.pdf")
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_yz"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_xz"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("cnv_dipole_window.pdf")
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    histos["hTanTheta_yz"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    histos["hTanTheta_xz"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("cnv_dipole_window.pdf")
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetLogy()
    cnv.SetTicks(1,1)
    histos["hdExit"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("cnv_dipole_window.pdf")
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetLogy()
    cnv.SetTicks(1,1)
    histos["hP"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("cnv_dipole_window.pdf)")
    
    
    fout = ROOT.TFile("cnv_dipole_window.root","RECREATE")
    fout.cd()
    for hname,hist in histos.items(): hist.Write()
    fout.Write()
    fout.Close()
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time} seconds')