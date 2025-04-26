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
import candidate
from candidate import *
import svd_fit
from svd_fit import *
import chi2_fit
from chi2_fit import *
import hists
from hists import *

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

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

B  = cfg["fDipoleTesla"]
LB = cfg["zDipoleLenghMeters"]
mm2m = 1e-3


def h1h2max(h1,h2):
    hmax = -1
    y1 = h1.GetMaximum()
    y2 = h2.GetMaximum()
    hmax = y1 if(y1>y2) else y2
    return hmax

def fit1(h,col,xmin,xmax):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g1.SetLineColor(col)
    h.Fit(g1,"EMRS")
    chi2dof = g1.GetChisquare()/g1.GetNDF() if(g1.GetNDF()>0) else -1
    print("g1 chi2/Ndof=",chi2dof)
    return g1

def refit(track):
    nonzeromisalignment = False
    for key1 in cfg["misalignment"]:
        for key2 in cfg["misalignment"][key1]:
            if(cfg["misalignment"][key1][key2]!=0): 
                nonzeromisalignment = True
                break
        if(nonzeromisalignment): break
    
    ############################################### 
    ### don't bother if there's no misalignment ###
    if(not nonzeromisalignment): return track
    ###############################################
    
    clusters = track.trkcls
    seed_x = {}
    seed_y = {}
    seed_z = {}
    seed_dx = {}
    seed_dy = {}
    for det in cfg["detectors"]:
        ### first align!!
        if(nonzeromisalignment):
            clusters[det].xmm,clusters[det].ymm = align(det,clusters[det].xmm,clusters[det].ymm)
        ### then prepare for refit
        seed_x.update({  det : clusters[det].xmm  })
        seed_y.update({  det : clusters[det].ymm  })
        seed_z.update({  det : clusters[det].zmm  })
        seed_dx.update({ det : clusters[det].xsizemm if(cfg["use_large_clserr_for_algnmnt"]) else clusters[det].dxmm })
        seed_dy.update({ det : clusters[det].ysizemm if(cfg["use_large_clserr_for_algnmnt"]) else clusters[det].dymm })
    ### then prepare for refit
    vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
    evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
    points_SVD, errors_SVD  = SVD_candidate(seed_x,seed_y,seed_z,seed_dx,seed_dy,vtx,evtx)
    points_Chi2,errors_Chi2 = Chi2_candidate(seed_x,seed_y,seed_z,seed_dx,seed_dy,vtx,evtx)
    
    chisq     = None
    ndof      = None
    direction = None
    centroid  = None
    params    = None
    success   = None
    
    ### svd fit
    if("SVD" in cfg["fit_method"]):
        chisq,ndof,direction,centroid = fit_3d_SVD(points_SVD,errors_SVD)
        params = get_pars_from_centroid_and_direction(centroid,direction)
        success = True
    ### chi2 fit
    if("CHI2" in cfg["fit_method"]):
        chisq,ndof,direction,centroid,params,success = fit_3d_chi2err(points_Chi2,errors_Chi2,par_guess)
    
    ### set the track
    track = Track(clusters,points_SVD,errors_SVD,chisq,ndof,direction,centroid,params,success)
    return track

def pass_dk_at_detector(track,detector,dxMin,dxMax,dyMin,dyMax):
    dx,dy = res_track2cluster(detector,track.points,track.direction,track.centroid)
    if(dx<dxMin or dx>dxMax): return False
    if(dy<dyMin or dy>dyMax): return False
    return True


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
    files = [fx for fx in files if '_BadTriggers' not in fx]
    for f in files: print(f)
    
    
    ### bad triggers
    fpkltrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl")
    badtriggers = []
    if(not cfg["isMC"]):
        fpkltrigger = open(fpkltrgname,'rb')
        badtriggers = pickle.load(fpkltrigger)
        fpkltrigger.close()
    nbadtrigs = len(badtriggers)
    print(f"Found {nbadtrigs} bad triggers")
    
    
    ### counters
    init_global_counters()
    Ndet = len(cfg["detectors"])
    
    
    ### some histos
    histos = {}
    histos.update({ "hChi2DoF_alowshrcls": ROOT.TH1D("hChi2DoF_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,50)})
    histos.update({ "hChi2DoF_zeroshrcls": ROOT.TH1D("hChi2DoF_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,50)})
    
    histos.update({ "hChi2DoF_full_alowshrcls": ROOT.TH1D("hChi2DoF_full_alowshrcls",";#chi^{2}/N_{DoF};Tracks",400,0,200)})
    histos.update({ "hChi2DoF_full_zeroshrcls": ROOT.TH1D("hChi2DoF_full_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",400,0,200)})
    
    histos.update({ "hChi2DoF_zoom_alowshrcls": ROOT.TH1D("hChi2DoF_zoom_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,5)})
    histos.update({ "hChi2DoF_zoom_zeroshrcls": ROOT.TH1D("hChi2DoF_zoon_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,5)})
    
    histos.update({ "hChi2DoF_0to1_alowshrcls": ROOT.TH1D("hChi2DoF_0to1_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,1)})
    histos.update({ "hChi2DoF_0to1_zeroshrcls": ROOT.TH1D("hChi2DoF_0to1_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,1)})
    
    histos.update({ "hPf_vs_dExit": ROOT.TH2D("hPf_vs_dExit",";d_{exit} [mm];p(#theta(fit)) [GeV];Tracks",50,0,+35, 50,0,10) })
    histos.update({ "hPd_vs_dExit": ROOT.TH2D("hPd_vs_dExit",";d_{exit} [mm];p(#theta(d_{exit}) [GeV];Tracks",50,0,+35, 50,0,10) })
    histos.update({ "hPr_vs_dExit": ROOT.TH2D("hPr_vs_dExit",";d_{exit} [mm];p(#theta(r) [GeV];Tracks",50,0,+35, 50,0,10) })

    histos.update({ "hPf_vs_thetaf": ROOT.TH2D("hPf_vs_thetaf",";#theta_{yz}(fit) [rad];p(#theta(fit)) [GeV];Tracks",50,0,0.05, 50,0,10) })
    histos.update({ "hPd_vs_thetad": ROOT.TH2D("hPd_vs_thetad",";#theta_{yz}(d_{exit}) [rad];p(#theta(d_{exit})) [GeV];Tracks",50,0,0.05, 50,0,10) })
    histos.update({ "hPr_vs_thetar": ROOT.TH2D("hPr_vs_thetar",";#theta_{yz}(r) [rad];p(#theta(r)) [GeV];Tracks",50,0,0.05, 50,0,10) })

    histos.update({ "hDexit_vs_thetaf": ROOT.TH2D("hDexit_vs_thetaf",";#theta_{yz}(fit) [rad];d_{exit} [mm];Tracks",50,0,0.05, 50,0,+35) })
    histos.update({ "hDexit_vs_thetad": ROOT.TH2D("hDexit_vs_thetad",";#theta_{yz}(d_{exit}) [rad];d_{exit} [mm];Tracks",50,0,0.05, 50,0,+35) })
    histos.update({ "hDexit_vs_thetar": ROOT.TH2D("hDexit_vs_thetar",";#theta_{yz}(r) [rad];d_{exit} [mm];Tracks",50,0,0.05, 50,0,+35) })
    
    histos.update({ "hThetad_vs_thetaf": ROOT.TH2D("hThetad_vs_thetaf",";#theta_{yz}(fit) [rad];#theta(d_{exit}) [rad];Tracks",50,0,0.05, 50,0,0.05) })
    histos.update({ "hThetar_vs_thetaf": ROOT.TH2D("hThetar_vs_thetaf",";#theta_{yz}(fit) [rad];#theta(r) [rad];Tracks",50,0,0.05, 50,0,0.05) })
    
    histos.update({ "hD_before_cuts": ROOT.TH2D("hD_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hD_after_cuts":  ROOT.TH2D("hD_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hD_zoomout_before_cuts": ROOT.TH2D("hD_zoomout_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-1000,+1000, 120,-1000,+1000) })
    histos.update({ "hD_zoomout_after_cuts":  ROOT.TH2D("hD_zoomout_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-1000,+1000, 120,-1000,+1000) })
    
    histos.update({ "hW_before_cuts": ROOT.TH2D("hW_before_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    histos.update({ "hW_after_cuts":  ROOT.TH2D("hW_after_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    
    histos.update({ "hThetaf_yz": ROOT.TH1D("hThetaf_yz",";#theta_{yz}(fit) [rad];Tracks",100,0,0.1)})
    histos.update({ "hThetad_yz": ROOT.TH1D("hThetad_yz",";#theta_{yz}(d_{exit}) [rad];Tracks",100,0,0.1)})
    histos.update({ "hThetar_yz": ROOT.TH1D("hThetar_yz",";#theta_{yz}(r) [rad];Tracks",100,0,0.1)})
    
    histos.update({ "hTheta_xz": ROOT.TH1D("hTheta_xz",";#theta_{xz} [rad];Tracks",100,-0.006,0.006)})
    histos.update({ "hTheta_yz": ROOT.TH1D("hTheta_yz",";#theta_{yz} [rad];Tracks",100,0,0.035)})
    
    histos.update({ "hTheta_xz_tru": ROOT.TH1D("hTheta_xz_tru",";#theta_{xz} [rad];Tracks",100,-0.006,0.006)})
    histos.update({ "hTheta_yz_tru": ROOT.TH1D("hTheta_yz_tru",";#theta_{yz} [rad];Tracks",100,0,0.035)})
    
    histos.update({ "hTheta_xz_tru_all": ROOT.TH1D("hTheta_xz_tru_all",";#theta_{xz} [rad];Tracks",100,-0.006,0.006)})
    histos.update({ "hTheta_yz_tru_all": ROOT.TH1D("hTheta_yz_tru_all",";#theta_{yz} [rad];Tracks",100,0,0.035)})
    
    histos.update({ "hdExit":    ROOT.TH1D("hdExit",";d_{exit} [mm];Tracks",120,-70,+90)})
    
    histos.update({ "hTheta_xz_response": ROOT.TH1D("hThetaf_xz_response",";#frac{#theta_{xz}^{rec}-#theta_{xz}^{tru}}{#theta_{xz}^{tru}};Tracks",100,-0.5,0.5)})
    histos.update({ "hTheta_yz_response": ROOT.TH1D("hThetaf_yz_response",";#frac{#theta_{yz}^{rec}-#theta_{yz}^{tru}}{#theta_{yz}^{tru}};Tracks",100,-0.05,0.05)})
    histos.update({ "hD_x_response": ROOT.TH1D("hD_x_response",";#frac{x_{vtx}^{rec}-x_{vtx}^{tru}}{x_{vtx}^{tru}};Tracks",100,-0.5,0.5)})
    histos.update({ "hD_y_response": ROOT.TH1D("hD_y_response",";#frac{y_{vtx}^{rec}-y_{vtx}^{tru}}{y_{vtx}^{tru}};Tracks",100,-0.5,0.5)})
    
    histos.update({ "hPf": ROOT.TH1D("hPf",";p(fit) [GeV];Tracks",100,0,10)})
    histos.update({ "hPd": ROOT.TH1D("hPd",";p(d_{exit}) [GeV];Tracks",100,0,10)})
    histos.update({ "hPr": ROOT.TH1D("hPr",";p(r) [GeV];Tracks",100,0,10)})
    
    histos.update({ "hPf_zoom": ROOT.TH1D("hPf_zoom",";p(fit) [GeV];Tracks",50,1,5)})
    histos.update({ "hPd_zoom": ROOT.TH1D("hPd_zoom",";p(d_{exit}) [GeV];Tracks",50,1,5)})
    histos.update({ "hPr_zoom": ROOT.TH1D("hPr_zoom",";p(r) [GeV];Tracks",50,1,5)})
    
    absRes   = 0.05
    nResBins = 50
    nResBins
    for det in cfg["detectors"]:
        name = f"h_residual_alowshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_alowshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_alowshrcls_x_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_alowshrcls_y_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_alowshrcls_x_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_alowshrcls_y_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

        name = f"h_response_alowshrcls_x_sml_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_y_sml_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_x_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_y_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        
        name = f"h_residual_zeroshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_zeroshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_zeroshrcls_x_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_y_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_x_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_zeroshrcls_y_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

        name = f"h_response_zeroshrcls_x_sml_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-5,+5) } )
        name = f"h_response_zeroshrcls_y_sml_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-5,+5) } )
        name = f"h_response_zeroshrcls_x_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_zeroshrcls_y_ful_{det}"; histos.update( {name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        
        name = f"h_residual_zeroshrcls_xy_{det}";    histos.update( { name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3, nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_xy_mid_{det}";histos.update( { name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*5,+absRes*5, nResBins,-absRes*5,+absRes*5) } )
    
    
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

                
                ### skip bad triggers...
                if(not cfg["isMC"] and cfg["runtype"]=="beam" and (int(event.trigger) in badtriggers)): continue
                

                ### check errors
                if(not cfg["isMC"]):
                    if(len(event.errors)!=len(cfg["detectors"])): continue
                    nErrors = 0
                    for det in cfg["detectors"]: nErrors += len(event.errors[det])
                    if(nErrors>0): continue

                
                ### check pixels
                # if(len(event.pixels)!=len(cfg["detectors"])): continue
                if(len(event.npixels)!=len(cfg["detectors"])): continue
                n_pixels = 0
                pass_pixels = True
                for det in cfg["detectors"]:
                    #npix = len( event.pixels[det] )
                    npix = event.npixels[det]
                    if(npix==0): pass_pixels = False
                    n_pixels += npix
                set_global_counter("Pixels/chip",icounter,n_pixels/Ndet)
                if(not pass_pixels): continue

                ### check clusters
                # if(len(event.clusters)!=len(cfg["detectors"])): continue
                if(len(event.nclusters)!=len(cfg["detectors"])): continue
                n_clusters = 0
                pass_clusters = True
                for det in cfg["detectors"]:
                    # ncls = len(event.clusters[det])
                    ncls = event.nclusters[det]
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
                # print(f"I see {len(event.tracks)} tracks in event {nevents-1}")
                for track in event.tracks:
                    
                    ##################################
                    ### first require max cluster ####
                    ##################################
                    if(track.maxcls>cfg["cut_maxcls"]): continue
                    
                    ### fill some quantities before alignment
                    if(track.chi2ndof<=cfg["cut_chi2dof"] and pass_geoacc_selection(track)): ##TODO: missing the shared hits cut here...
                        histos["hChi2DoF_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_full_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_zoom_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_0to1_alowshrcls"].Fill(track.chi2ndof)
                        for det in cfg["detectors"]:
                            dx,dy = res_track2cluster(det,track.points,track.direction,track.centroid)
                            histos[f"h_residual_alowshrcls_x_sml_{det}"].Fill(dx)
                            histos[f"h_residual_alowshrcls_x_mid_{det}"].Fill(dx)
                            histos[f"h_residual_alowshrcls_x_ful_{det}"].Fill(dx)
                            histos[f"h_residual_alowshrcls_y_sml_{det}"].Fill(dy)
                            histos[f"h_residual_alowshrcls_y_mid_{det}"].Fill(dy)
                            histos[f"h_residual_alowshrcls_y_ful_{det}"].Fill(dy)
                            histos[f"h_response_alowshrcls_x_sml_{det}"].Fill(dx/track.trkcls[det].dxmm)
                            histos[f"h_response_alowshrcls_x_ful_{det}"].Fill(dx/track.trkcls[det].dxmm)
                            histos[f"h_response_alowshrcls_y_sml_{det}"].Fill(dy/track.trkcls[det].dymm)
                            histos[f"h_response_alowshrcls_y_ful_{det}"].Fill(dy/track.trkcls[det].dymm)
                    
                    ########################
                    track = refit(track) ###
                    ### will be the same if
                    ### misalignment is 0
                    ########################

                    if(cfg["isMC"] and cfg["isFakeMC"]):
                        slp = event.fakemcparticles[0].slp
                        itp = event.fakemcparticles[0].itp
                        vtx = event.fakemcparticles[0].vtx
                        histos["hTheta_xz_tru_all"].Fill(slp[0])
                        histos["hTheta_yz_tru_all"].Fill(slp[1])
                    
                    #########################
                    ### then require chi2 ###
                    #########################
                    if(track.chi2ndof>cfg["cut_chi2dof"]): continue ### this is the new chi2!
                    good_tracks.append(track)
                    
                    ### get the coordinates at extreme points in real space and after tilting the detector
                    r0,rN,rW,rD = get_track_point_at_extremes(track)

                    ### the y distance from y=0 in the dipole exit plane
                    dExit = rD[1]

                    ### fill histos before cuts
                    histos["hD_before_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomout_before_cuts"].Fill(rD[0],rD[1])
                    histos["hW_before_cuts"].Fill(rW[0],rW[1])
                    
                    
                    ##########################################
                    ### require pointing to the pdc window ###
                    ### and the dipole exit aperture       ###
                    ### and inclined up as a positron      ###
                    ##########################################
                    if(not pass_geoacc_selection(track)): continue
                                        
                    ### the fit angles
                    tan_theta_yz = +track.params[1] ### the slope p1x transformed to real space (stays as is)
                    tan_theta_xz = -track.params[3] ### the slope p2x transformed to real space (gets minus sign)
                    thetaf_yz = math.atan(tan_theta_yz) - cfg["thetax"] ###TODO: check if - or +
                    thetaf_xz = math.atan(tan_theta_xz) - cfg["thetay"] ###TODO: check if - or +
                    
                    if(cfg["isMC"] and cfg["isFakeMC"]):
                        slp = event.fakemcparticles[0].slp
                        itp = event.fakemcparticles[0].itp
                        vtx = event.fakemcparticles[0].vtx
                        histos["hTheta_xz_tru"].Fill(slp[0])
                        histos["hTheta_yz_tru"].Fill(slp[1])
                        # print(f"thetaf_xz={thetaf_xz}, slp={slp[0]}, thetaf_yz={thetaf_yz}, slp={slp[1]}")
                        histos["hTheta_xz_response"].Fill( (thetaf_xz-slp[0])/slp[0] if(slp[0]!=0) else -1. )
                        histos["hTheta_yz_response"].Fill( (thetaf_yz-slp[1])/slp[1] if(slp[1]!=0) else -1. )
                        histos["hD_x_response"].Fill( (rD[0]-vtx[0])/vtx[0] if(vtx[0]!=0) else -1. )
                        histos["hD_y_response"].Fill( (rD[1]-vtx[1])/vtx[1] if(vtx[1]!=0) else -1. )
                        
                    
                    ### the angle in y-z calculated from d_exit
                    thetad_yz = 2.*math.atan(dExit*mm2m/LB)
                    ### the angle in y-z calculated from the tilted detector extremes
                    thetar_yz = math.atan( (rN[1]-r0[1])/(rN[2]-r0[2]) )
                    
                    ### the momentum magnitudes
                    pf = (0.3 * B * LB)/math.sin( thetaf_yz )
                    pd = (0.3 * B * LB)/math.sin( thetad_yz )
                    pr = (0.3 * B * LB)/math.sin( thetar_yz )
                    
                    histos["hThetad_vs_thetaf"].Fill(thetaf_yz,thetad_yz)
                    histos["hThetar_vs_thetaf"].Fill(thetaf_yz,thetar_yz)
                    
                    histos["hPf_vs_dExit"].Fill(dExit,pf)
                    histos["hPd_vs_dExit"].Fill(dExit,pd)
                    histos["hPr_vs_dExit"].Fill(dExit,pr)
                    
                    histos["hPf_vs_thetaf"].Fill(thetaf_yz,pf)
                    histos["hPd_vs_thetad"].Fill(thetad_yz,pd)
                    histos["hPr_vs_thetar"].Fill(thetar_yz,pr)
                    
                    histos["hDexit_vs_thetaf"].Fill(thetaf_yz,dExit)
                    histos["hDexit_vs_thetad"].Fill(thetad_yz,dExit)
                    histos["hDexit_vs_thetar"].Fill(thetar_yz,dExit)
                    
                    histos["hD_after_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomout_after_cuts"].Fill(rD[0],rD[1])
                    histos["hW_after_cuts"].Fill(rW[0],rW[1])
                    
                    histos["hThetaf_yz"].Fill(thetaf_yz)
                    histos["hThetad_yz"].Fill(thetad_yz)
                    histos["hThetar_yz"].Fill(thetar_yz)
                    
                    histos["hTheta_xz"].Fill(thetaf_xz)
                    histos["hTheta_yz"].Fill(thetaf_yz)
                    
                    histos["hdExit"].Fill(dExit)
                    
                    if(pf>0):
                        histos["hPf"].Fill(pf)
                        histos["hPf_zoom"].Fill(pf)
                    if(pd>0):
                        histos["hPd"].Fill(pd)
                        histos["hPd_zoom"].Fill(pd)
                    if(pr>0):
                        histos["hPr"].Fill(pr)
                        histos["hPr_zoom"].Fill(pr)
                    
                    acceptance_tracks.append(track)
                    ntracks += 1
                
                ### the graph of the good tracks
                set_global_counter("Good Tracks",icounter,len(good_tracks))
                
                ### check for overlaps
                selected_tracks = acceptance_tracks if(cfg["cut_allow_shared_clusters"]) else remove_tracks_with_shared_clusters(acceptance_tracks)
                # if(len(selected_tracks)!=len(acceptance_tracks)): print(f"nsel:{len(acceptance_tracks)} --> npas={len(selected_tracks)}")
                set_global_counter("Selected Tracks",icounter,len(selected_tracks))
                
                ### event displays
                if(cfg["plot_online_evtdisp"] and len(good_tracks)>0):
                    fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_offline_{event.trigger}.pdf")
                    plot_event(event.meta.run,event.meta.start,event.meta.dur,event.trigger,fevtdisplayname,event.clusters,event.tracks,chi2threshold=cfg["cut_chi2dof"])
                
                ### plot some selected tracks
                for track in selected_tracks:
                    
                    # dx,dy = res_track2cluster("ALPIDE_3",track.points,track.direction,track.centroid)
                    # if(dx>-0.02): continue
                    # if(dy>-0.02): continue
                    
                    histos["hChi2DoF_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_full_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_zoom_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_0to1_zeroshrcls"].Fill(track.chi2ndof)
                    for det in cfg["detectors"]:
                        dx,dy = res_track2cluster(det,track.points,track.direction,track.centroid)
                        histos[f"h_residual_zeroshrcls_x_sml_{det}"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_x_mid_{det}"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_x_ful_{det}"].Fill(dx)
                        histos[f"h_residual_zeroshrcls_y_sml_{det}"].Fill(dy)
                        histos[f"h_residual_zeroshrcls_y_mid_{det}"].Fill(dy)
                        histos[f"h_residual_zeroshrcls_y_ful_{det}"].Fill(dy)
                        
                        histos[f"h_residual_zeroshrcls_xy_{det}"].Fill(dx,dy)
                        histos[f"h_residual_zeroshrcls_xy_mid_{det}"].Fill(dx,dy)
                        
                        histos[f"h_response_zeroshrcls_x_sml_{det}"].Fill(dx/track.trkcls[det].dxmm)
                        histos[f"h_response_zeroshrcls_x_ful_{det}"].Fill(dx/track.trkcls[det].dxmm)
                        histos[f"h_response_zeroshrcls_y_sml_{det}"].Fill(dy/track.trkcls[det].dymm)
                        histos[f"h_response_zeroshrcls_y_ful_{det}"].Fill(dy/track.trkcls[det].dymm)
                    
                    
                
                print(f"Event[{nevents-1}], Trigger[{event.trigger}] --> Good tracks: {len(good_tracks)}, Acceptance tracks: {len(acceptance_tracks)}, Selected tracks: {len(selected_tracks)}")

    # print(f"Events:{nevents}, Tracks:{ntracks}")
    print(f"Tracks:{ntracks}, GoodTriggers:{nevents-nbadtrigs}  (with AllTriggers:{nevents} and BadTriggers: {nbadtrigs})")
    
    ### plot the counters
    fmultpdfname = tfilenamein.replace(".root",f"_multiplicities_vs_triggers.pdf")
    plot_counters(fmultpdfname)


    ### plot the geometry distributions
    foupdfname = tfilenamein.replace(".root",f"_dipole_window.pdf")

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hD_zoomout_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hD_zoomout_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}(")

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
    cnv.SaveAs(f"{foupdfname}")
    
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
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    # cnv.SetLogy()
    cnv.SetTicks(1,1)
    histos["hTheta_xz"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_xz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_xz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    # cnv.SetLogy()
    cnv.SetTicks(1,1)
    histos["hTheta_yz"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_yz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_yz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    if(cfg["isMC"] and cfg["isFakeMC"]):
        cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        histos["hTheta_xz_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        histos["hTheta_yz_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.Update()
        cnv.SaveAs(f"{foupdfname}")
        
        cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        hTheta_xz_eff = histos["hTheta_xz_tru"].Clone("hTheta_xz_tru_clone") 
        hTheta_xz_eff.Divide(histos["hTheta_xz_tru_all"])    
        hTheta_xz_eff.Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        hTheta_yz_eff = histos["hTheta_yz_tru"].Clone("hTheta_yz_tru_clone") 
        hTheta_yz_eff.Divide(histos["hTheta_yz_tru_all"])
        hTheta_yz_eff.Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.Update()
        cnv.SaveAs(f"{foupdfname}")
        
        
        
        cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        histos["hD_x_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        histos["hD_y_response"].Draw("hist")
        ROOT.gPad.RedrawAxis()
        cnv.Update()
        cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    histos["hdExit"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv.Divide(2,1)
    # cnv.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hdExit"].Draw("hist")
    # cnv.RedrawAxis()
    # cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hThetad_yz"].Draw("hist")
    # cnv.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hThetaf_yz"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hThetad_yz"].Draw("hist")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hThetar_yz"].Draw("hist")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf_zoom"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd_zoom"].Draw("hist")
    # cnv.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr_zoom"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd"].Draw("hist")
    # cnv.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    histos["hPf"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    # cnv.SetTicks(1,1)
    # histos["hPd"].Draw("hist")
    # cnv.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    histos["hPr"].Draw("hist")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    # cnv.Divide(2,1)
    # # cnv.Divide(3,1)
    # cnv.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPf_vs_dExit"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(2)
    # # ROOT.gPad.SetTicks(1,1)
    # # histos["hPd_vs_dExit"].Draw("colz")
    # # ROOT.gPad.RedrawAxis()
    # # cnv.cd(3)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPr_vs_dExit"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    cnv.Divide(2,1)
    # cnv.Divide(3,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hPf_vs_thetaf"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd_vs_thetad"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr_vs_thetar"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1500,500)
    # cnv.Divide(3,1)
    # cnv.cd(1)
    # # ROOT.gPad.SetLogy()
    # ROOT.gPad.SetTicks(1,1)
    # histos["hDexit_vs_thetaf"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(2)
    # # ROOT.gPad.SetLogy()
    # ROOT.gPad.SetTicks(1,1)
    # histos["hDexit_vs_thetad"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(3)
    # # ROOT.gPad.SetLogy()
    # ROOT.gPad.SetTicks(1,1)
    # histos["hDexit_vs_thetar"].Draw("colz")
    # ROOT.gPad.RedrawAxis()
    # cnv.Update()
    # cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    # cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    # cnv.Divide(2,1)
    # cnv.cd(1)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hThetad_vs_thetaf"].Draw("colz")
    # dipole.Draw()
    # ROOT.gPad.RedrawAxis()
    # cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hThetar_vs_thetaf"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_full_alowshrcls"],histos["hChi2DoF_full_zeroshrcls"])
    histos["hChi2DoF_full_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_full_zeroshrcls"].SetMaximum(1.1*hmax)  
    histos["hChi2DoF_full_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_full_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_full_alowshrcls"].Draw("hist")
    histos["hChi2DoF_full_zeroshrcls"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_alowshrcls"],histos["hChi2DoF_zeroshrcls"])
    histos["hChi2DoF_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_alowshrcls"].Draw("hist")
    histos["hChi2DoF_zeroshrcls"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_zoom_alowshrcls"],histos["hChi2DoF_zoom_zeroshrcls"])
    histos["hChi2DoF_zoom_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_zoom_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_zoom_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_zoom_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_zoom_alowshrcls"].Draw("hist")
    histos["hChi2DoF_zoom_zeroshrcls"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_0to1_alowshrcls"],histos["hChi2DoF_0to1_zeroshrcls"])
    histos["hChi2DoF_0to1_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_0to1_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_0to1_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_0to1_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_0to1_alowshrcls"].Draw("hist")
    histos["hChi2DoF_0to1_zeroshrcls"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_residual_zeroshrcls_xy_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_residual_zeroshrcls_xy_mid_{det}"].Draw("colz")
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    

    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_sml_{det}"].Draw("e1p")
        xmin = histos[f"h_residual_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_residual_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmax()
        mm2um = 1e3
        func = fit1(histos[f"h_residual_zeroshrcls_x_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")

    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_x_mid_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_x_mid_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_mid_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_mid_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_mid_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")

    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_x_ful_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_x_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_x_ful_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_x_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_sml_{det}"].Draw("e1p")
        xmin = histos[f"h_residual_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_residual_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmax()
        mm2um = 1e3
        func = fit1(histos[f"h_residual_zeroshrcls_y_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_y_mid_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_y_mid_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_mid_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_mid_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_mid_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMinimum(0)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_residual_alowshrcls_y_ful_{det}"].GetMaximum()
        hamax = histos[f"h_residual_zeroshrcls_y_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMaximum(hmax)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_residual_alowshrcls_y_ful_{det}"].Draw("ep")
        
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_residual_zeroshrcls_y_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_sml_{det}"].Draw("e1p")
        
        xmin = histos[f"h_response_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_response_zeroshrcls_x_sml_{det}"].GetXaxis().GetXmax()
        func = fit1(histos[f"h_response_zeroshrcls_x_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_response_alowshrcls_x_ful_{det}"].GetMaximum()
        hamax = histos[f"h_response_zeroshrcls_x_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMaximum(hmax)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_x_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_x_ful_{det}"].Draw("ep")
        
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_x_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_sml_{det}"].Draw("e1p")
        
        xmin = histos[f"h_response_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmin()
        xmax = histos[f"h_response_zeroshrcls_y_sml_{det}"].GetXaxis().GetXmax()
        func = fit1(histos[f"h_response_zeroshrcls_y_sml_{det}"],ROOT.kRed,xmin,xmax)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.045)
        s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f" % (func.GetParameter(1))))
        s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f" % (func.GetParameter(2))))
        if(func.GetNDF()>0): s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)

        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMinimum(0)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMinimum(0)
        hbmax = histos[f"h_response_alowshrcls_y_ful_{det}"].GetMaximum()
        hamax = histos[f"h_response_zeroshrcls_y_ful_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMaximum(hmax)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMaximum(hmax)
        
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMarkerStyle(20)
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_y_ful_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_response_alowshrcls_y_ful_{det}"].Draw("ep")
        
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMarkerStyle(24)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetMarkerColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_response_zeroshrcls_y_ful_{det}"].Draw("ep same")
        
        ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname})")
    
    ### save as root file
    foutrootname = tfilenamein.replace(".root",f"_dipole_window.root")
    fout = ROOT.TFile(foutrootname,"RECREATE")
    fout.cd()
    for hname,hist in histos.items(): hist.Write()
    fout.Write()
    fout.Close()
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time} seconds')