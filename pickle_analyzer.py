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
    hough_coords = track.hough_coords
    clusters = track.trkcls
    seed_x = {}
    seed_y = {}
    seed_z = {}
    seed_dx = {}
    seed_dy = {}
    for det in cfg["detectors"]:
        ### first align!!
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
    track = Track(clusters,points_SVD,errors_SVD,chisq,ndof,direction,centroid,params,success,hough_coords)
    return track


# def pass_dk_at_detector(track,detector,dxMin,dxMax,dyMin,dyMax):
#     dx,dy = res_track2cluster(detector,track.points,track.direction,track.centroid)
#     if(dx<dxMin or dx>dxMax): return False
#     if(dy<dyMin or dy>dyMax): return False
#     return True


def get_wave(z,k,thetamin,thetamax):
    ### rho = k*sin(theta) + z*cos(theta)
    func = ROOT.TF1(f"func_{name}","[1]*sin(x)+[0]*cos(x)",thetamin,thetamax,2)
    func.SetParameter(0,z)
    func.SetParameter(1,k)
    return func

def find_waves_intersect(k1,z1,k2,z2):
    dk = (k1-k2) if(abs(k1-k2)>1e-15) else 1e15*np.sign(k1-k2)
    theta = math.atan2((z2-z1),dk) # the arc tangent of (y/x) in radians
    rho   = k1*math.sin(theta) + z1*math.cos(theta)
    # print(f"k1={k1},z1={z1}, k2={k2},z1={z2} --> theta={theta},rho={rho}")
    return theta,rho
    
def fill_pair(a,b,track,hx,hy):
    pair = [f"ALPIDE_{a}",f"ALPIDE_{b}"]
    rA = [track.trkcls[pair[0]].xmm,track.trkcls[pair[0]].ymm,track.trkcls[pair[0]].zmm]
    rB = [track.trkcls[pair[1]].xmm,track.trkcls[pair[1]].ymm,track.trkcls[pair[1]].zmm]
    thetax,rhox = find_waves_intersect(rA[0],rA[2],rB[0],rB[2])
    thetay,rhoy = find_waves_intersect(rA[1],rA[2],rB[1],rB[2])
    hx.Fill(thetax,rhox)
    hy.Fill(thetay,rhoy)

def get_par_lin(theta_k,rho_k): ### theta and rho from Hough transform
    if(math.sin(theta_k)==0):
        print(f"in get_par_lin, sin(theta)=0: quitting.")
        quit()
    if(math.tan(theta_k)==0):
        print(f"in get_par_lin, 1/tan(theta)=0: quitting.")
        quit()
    AK = -1./math.tan(theta_k)
    BK = rho_k/math.sin(theta_k)
    # print(f"theta_k={theta_k}, rho_k={rho_k} --> AK={AK}, BK={BK}")
    return AK,BK

def k_of_z(z,AK,BK):
    k = AK*z + BK
    # print(f"AK={AK}, BK={BK}, z={z} --> k={k}")
    return k

def get_edges_from_theta_rho_corners(det,theta_x,rho_x,theta_y,rho_y):
    xmin = +1e20
    xmax = -1e20
    ymin = +1e20
    ymax = -1e20
    for i in range(2):
        AX,BX = get_par_lin(theta_x[i],rho_x[i])
        AY,BY = get_par_lin(theta_y[i],rho_y[i])
        zdet = cfg["rdetectors"][det][2]
        XX = k_of_z(zdet,AX,BX)
        YY = k_of_z(zdet,AY,BY)
        # print(f"get_edges_from_theta_rho_corners cornere[i]: eventid={self.eventid}  -->  {det} prediction: x={XX}, y={YY}, z={zdet}")
        xmin = XX if(XX<xmin) else xmin
        xmax = XX if(XX>xmax) else xmax
        ymin = YY if(YY<ymin) else ymin
        ymax = YY if(YY>ymax) else ymax
    xmin = xmin-cfg["lut_widthx_mid"]
    xmax = xmax+cfg["lut_widthx_mid"]
    ymin = ymin-cfg["lut_widthy_mid"]
    ymax = ymax+cfg["lut_widthy_mid"]
    return xmin,xmax,ymin,ymax



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
    
    
    ### read production config
    fpklcfgname = tfilenamein.replace("tree_","config_used/tree_").replace(".root","_config.pkl")
    fpklconfig = open(fpklcfgname,'rb')
    prod_cfg = pickle.load(fpklconfig)
    fpklconfig.close()
    ### was it aligned during production?
    isAlignedAtProd = False
    for det in prod_cfg["detectors"]:
        for axis,value in prod_cfg["misalignment"][det].items():
            if(value!=0):
                isAlignedAtProd = True
                break
        if(isAlignedAtProd): break
    ### should we apply misalignemnt here?
    isNon0Mislaignment = False
    for det in cfg["detectors"]:
        for axis,value in cfg["misalignment"][det].items():
            if(value!=0):
                isNon0Mislaignment = True
                break
        if(isNon0Mislaignment): break
    
    
    
    ### bad triggers
    fpkltrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl")
    badtriggers = []
    if(not cfg["isMC"]):
        fpkltrigger = open(fpkltrgname,'rb')
        badtriggers = pickle.load(fpkltrigger)
        fpkltrigger.close()
    nbadtrigs = len(badtriggers)
    print(f"Found {nbadtrigs} bad triggers in the entire run")
    
    
    
    ### counters
    init_global_counters()
    Ndet = len(cfg["detectors"])
    
    
    ### some histos
    histos = {}
    
    histos.update({ "hTriggers": ROOT.TH1D("hTriggers",";;Triggers",2,0,2)})
    histos["hTriggers"].GetXaxis().SetBinLabel(1,"All")
    histos["hTriggers"].GetXaxis().SetBinLabel(2,"Good")
    
    histos.update({ "hChi2DoF_alowshrcls": ROOT.TH1D("hChi2DoF_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,50)})
    histos.update({ "hChi2DoF_zeroshrcls": ROOT.TH1D("hChi2DoF_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,50)})
    
    histos.update({ "hChi2DoF_full_alowshrcls": ROOT.TH1D("hChi2DoF_full_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,cfg["cut_chi2dof"])})
    histos.update({ "hChi2DoF_full_zeroshrcls": ROOT.TH1D("hChi2DoF_full_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,cfg["cut_chi2dof"])})
    
    histos.update({ "hChi2DoF_mid_alowshrcls": ROOT.TH1D("hChi2DoF_mid_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,200)})
    histos.update({ "hChi2DoF_mid_zeroshrcls": ROOT.TH1D("hChi2DoF_mid_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,200)})
    
    histos.update({ "hChi2DoF_small_alowshrcls": ROOT.TH1D("hChi2DoF_small_alowshrcls",";#chi^{2}/N_{DoF};Tracks",100,0,20)})
    histos.update({ "hChi2DoF_small_zeroshrcls": ROOT.TH1D("hChi2DoF_small_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",100,0,20)})
    
    histos.update({ "hChi2DoF_zoom_alowshrcls": ROOT.TH1D("hChi2DoF_zoom_alowshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,5)})
    histos.update({ "hChi2DoF_zoom_zeroshrcls": ROOT.TH1D("hChi2DoF_zoon_zeroshrcls",";#chi^{2}/N_{DoF};Tracks",200,0,5)})
    
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

    histos.update({ "hF_before_cuts": ROOT.TH2D("hF_before_cuts","Dipole flange plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hF_after_cuts":  ROOT.TH2D("hF_after_cuts","Dipole flange plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    
    histos.update({ "hD_before_cuts": ROOT.TH2D("hD_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hD_after_cuts":  ROOT.TH2D("hD_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-80,+80, 120,-70,+90) })
    histos.update({ "hD_zoomout_before_cuts": ROOT.TH2D("hD_zoomout_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-1000,+1000, 120,-1000,+1000) })
    histos.update({ "hD_zoomout_after_cuts":  ROOT.TH2D("hD_zoomout_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",120,-1000,+1000, 120,-1000,+1000) })
    histos.update({ "hD_zoomin_before_cuts":  ROOT.TH2D("hD_zoomin_before_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks",200,1.2*cfg["xDipoleExitMin"],1.2*cfg["xDipoleExitMax"], 200,1.1*cfg["yDipoleExitMin"],1.1*cfg["yDipoleExitMax"]) })
    histos.update({ "hD_zoomin_after_cuts":   ROOT.TH2D("hD_zoomin_after_cuts","Dipole exit plane;x [mm];y [mm];Extrapolated Tracks", 200,1.2*cfg["xDipoleExitMin"],1.2*cfg["xDipoleExitMax"], 200,1.1*cfg["yDipoleExitMin"],1.1*cfg["yDipoleExitMax"]) })
    
    
    histos.update({ "hW_before_cuts": ROOT.TH2D("hW_before_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    histos.update({ "hW_after_cuts":  ROOT.TH2D("hW_after_cuts","Vacuum window plane;x [mm];y [mm];Extrapolated Tracks",120,-70,+70, 120,50,+190) })
    
    histos.update({ "hThetaf_yz": ROOT.TH1D("hThetaf_yz",";#theta_{yz}^{trk}(fit) [rad];Tracks",100,0,0.1)})
    histos.update({ "hThetad_yz": ROOT.TH1D("hThetad_yz",";#theta_{yz}(d_{exit}) [rad];Tracks",100,0,0.1)})
    histos.update({ "hThetar_yz": ROOT.TH1D("hThetar_yz",";#theta_{yz}(r) [rad];Tracks",100,0,0.1)})
    
    histos.update({ "hTheta_xz_before_cuts": ROOT.TH1D("hTheta_xz_before_cuts",";#theta_{xz}^{trk} [rad];Tracks",50,-0.015,0.015)})
    histos.update({ "hTheta_xz_after_cuts":  ROOT.TH1D("hTheta_xz_after_cuts",";#theta_{xz}^{trk} [rad];Tracks",50,-0.015,0.015)})
    histos.update({ "hTheta_yz_before_cuts": ROOT.TH1D("hTheta_yz_before_cuts",";#theta_{yz}^{trk} [rad];Tracks",50,0,0.045)})
    histos.update({ "hTheta_yz_after_cuts":  ROOT.TH1D("hTheta_yz_after_cuts",";#theta_{yz}^{trk} [rad];Tracks",50,0,0.045)})
    
    histos.update({ "hTheta_xz_tru": ROOT.TH1D("hTheta_xz_tru",";#theta_{xz} [rad];Tracks",100,-0.01,0.01)})
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
    
    histos.update({ "hPf_small": ROOT.TH1D("hPf_small",";p(fit) [GeV];Tracks",50,1.5,4.5)})
    histos.update({ "hPd_small": ROOT.TH1D("hPd_small",";p(d_{exit}) [GeV];Tracks",50,1.5,4.5)})
    histos.update({ "hPr_small": ROOT.TH1D("hPr_small",";p(r) [GeV];Tracks",50,1.5,4.5)})
    
    histos.update({ "hPf_zoom": ROOT.TH1D("hPf_zoom",";p(fit) [GeV];Tracks",40,1.5,3.5)})
    histos.update({ "hPd_zoom": ROOT.TH1D("hPd_zoom",";p(d_{exit}) [GeV];Tracks",40,1.5,3.5)})
    histos.update({ "hPr_zoom": ROOT.TH1D("hPr_zoom",";p(r) [GeV];Tracks",40,1.5,3.5)})

    thetaxmin = 0     #np.pi/2-cfg["seed_thetax_scale_mid"]*np.pi/2.
    thetaxmax = np.pi #np.pi/2+cfg["seed_thetax_scale_mid"]*np.pi/2.
    thetaymin = 0     #np.pi/2-cfg["seed_thetay_scale_mid"]*np.pi/2.
    thetaymax = np.pi #np.pi/2+cfg["seed_thetay_scale_mid"]*np.pi/2.
    minthetarhobins = 2000
    nthetarhobins = minthetarhobins if(cfg["seed_nbins_thetarho_mid"]<minthetarhobins) else cfg["seed_nbins_thetarho_mid"]
    histos.update({ "hWaves_zx" : ROOT.TH2D("hWaves_zx",";#theta_{zx};#rho_{zx};",nthetarhobins,thetaxmin,thetaxmax,nthetarhobins,-90,90) })
    histos.update({ "hWaves_zy" : ROOT.TH2D("hWaves_zy",";#theta_{zy};#rho_{zy};",nthetarhobins,thetaymin,thetaymax,nthetarhobins,-90,90) })
    histos.update({ "hWaves_zx_intersections" : ROOT.TH2D("hWaves_zx_intersections",";#theta_{zx};#rho_{zx};",nthetarhobins,thetaxmin,thetaxmax,nthetarhobins,-90,90) })
    histos.update({ "hWaves_zy_intersections" : ROOT.TH2D("hWaves_zy_intersections",";#theta_{zy};#rho_{zy};",nthetarhobins,thetaymin,thetaymax,nthetarhobins,-90,90) })
    
    absRes   = 0.05
    nResBins = 50
    limtnl = {"ALPIDE_0":[0.0,0.35], "ALPIDE_1":[0.0,0.50], "ALPIDE_2":[0.0,0.65], "ALPIDE_3":[0.0,0.8], "ALPIDE_4":[0.0,0.95]}
    bintnl = 60
    for det in cfg["detectors"]:
        name = f"h_residual_alowshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_alowshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_alowshrcls_x_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_alowshrcls_y_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_alowshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_alowshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

        name = f"h_response_alowshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_alowshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        
        name = f"h_residual_zeroshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_zeroshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",int(nResBins*0.6),-absRes*0.6,+absRes*0.6) } )
        name = f"h_residual_zeroshrcls_x_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_y_mid_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";x_{trk}-x_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )
        name = f"h_residual_zeroshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";y_{trk}-y_{cls} [mm];Tracks",nResBins*2,-absRes*5,+absRes*5) } )

        name = f"h_response_zeroshrcls_x_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-5,+5) } )
        name = f"h_response_zeroshrcls_y_sml_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-5,+5) } )
        name = f"h_response_zeroshrcls_x_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-12.5,+12.5) } )
        name = f"h_response_zeroshrcls_y_ful_{det}"; histos.update( { name:ROOT.TH1D(name,det+";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-12.5,+12.5) } )
        
        name = f"h_residual_zeroshrcls_xy_{det}";    histos.update( { name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*3,+absRes*3, nResBins,-absRes*3,+absRes*3) } )
        name = f"h_residual_zeroshrcls_xy_mid_{det}";histos.update( { name:ROOT.TH2D(name,det+";x_{trk}-x_{cls} [mm];y_{trk}-y_{cls} [mm];Tracks",nResBins,-absRes*5,+absRes*5, nResBins,-absRes*5,+absRes*5) } )
    
        name = f"h_tunnel_width_x_{det}"; histos.update( { name:ROOT.TH1D(name,det+";Tunnel width [mm];Tracks",bintnl,limtnl[det][0],limtnl[det][1]) } )
        name = f"h_tunnel_width_y_{det}"; histos.update( { name:ROOT.TH1D(name,det+";Tunnel width [mm];Tracks",bintnl,limtnl[det][0],limtnl[det][1]) } )
    
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

    flange = ROOT.TPolyLine()
    xMinF = cfg["xFlangeMin"]
    xMaxF = cfg["xFlangeMax"]
    yMinF = cfg["yFlangeMin"]
    yMaxF = cfg["yFlangeMax"]    
    flange.SetNextPoint(xMinF,yMinF)
    flange.SetNextPoint(xMinF,yMaxF)
    flange.SetNextPoint(xMaxF,yMaxF)
    flange.SetNextPoint(xMaxF,yMinF)
    flange.SetNextPoint(xMinF,yMinF)
    flange.SetLineColor(ROOT.kBlue)
    flange.SetLineWidth(1)
    
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
    
    #################################
    ### prepare for eudaq writeup ###
    #################################
    ### declare the data tree and its classes
    ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
    ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; std::vector<TVector3> cls0; std::vector<TVector3> cls1; std::vector<TVector3> cls2; std::vector<TVector3> cls3; };" )
    ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
    ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
    ### declare the meta-data tree and its classes
    ROOT.gROOT.ProcessLine("struct run_meta_data  { Int_t run_number; Double_t run_start; Double_t run_end; };" )
    ### the main root gile
    runnum = get_run_from_file(cfg["inputfile"])
    fEUDAQout = ROOT.TFile.Open(f"tree_with_HT_selected_tracks_only_Run{runnum}.root", "RECREATE")
    ### data tree
    tEUDAQout = ROOT.TTree("MyTree","")
    eudaq_event = ROOT.event()
    tEUDAQout.Branch("event", eudaq_event)
    ### meta-data tree
    tEUDAQoutMeta = ROOT.TTree("MyTreeMeta","")
    run_meta_data = ROOT.run_meta_data()
    tEUDAQoutMeta.Branch("run_meta_data", run_meta_data)
    ### fill meta-data tree
    run_meta_data.run_number = runnum ### dummy
    run_meta_data.run_start  = -1.    ### dummy
    run_meta_data.run_end    = -1.    ### dummy
    tEUDAQoutMeta.Fill()
    
    
    
    
    ### save all events
    nevents = 0
    ntracks = 0
    nbadtrigs_actual = 0
    ntrigs_actual = 0
    tracks_triggers_dict = { "all": {"trgs":{"all":0,"good":0},"pix":{"all":0,"good":0},"cls":{"all":0,"good":0},"trks":0},
                             "even":{"trgs":{"all":0,"good":0},"pix":{"all":0,"good":0},"cls":{"all":0,"good":0},"trks":0},
                             "odd": {"trgs":{"all":0,"good":0},"pix":{"all":0,"good":0},"cls":{"all":0,"good":0},"trks":0} }
    for fpkl in files:
        suff = str(fpkl).split("_")[-1].replace(".pkl","")
        with open(fpkl,'rb') as handle:
            data = pickle.load(handle)
            for ievt,pkl_event in enumerate(data):
                # print(f"Reading event #{ievt}, trigger:{event.trigger}, ts:[{get_human_timestamp_ns(event.timestamp_bgn)}, {get_human_timestamp_ns(event.timestamp_end)}]")
                
                
                ########################################
                ### nicely clear per event for eudaq ###
                ########################################
                for s in range(eudaq_event.st_ev_buffer.size()):
                    for c in range(eudaq_event.st_ev_buffer[s].ch_ev_buffer.size()):
                        eudaq_event.st_ev_buffer[s].ch_ev_buffer[c].hits.clear()
                        eudaq_event.st_ev_buffer[s].ch_ev_buffer[c].cls0.clear()
                        eudaq_event.st_ev_buffer[s].ch_ev_buffer[c].cls1.clear()
                        eudaq_event.st_ev_buffer[s].ch_ev_buffer[c].cls2.clear()
                        eudaq_event.st_ev_buffer[s].ch_ev_buffer[c].cls3.clear()
                    eudaq_event.st_ev_buffer[s].ch_ev_buffer.clear()
                eudaq_event.st_ev_buffer.clear()
                eudaq_event.trg_n    = pkl_event.trigger
                eudaq_event.ts_begin = -1.
                eudaq_event.ts_end   = -1.
                eudaq_event.st_ev_buffer.push_back( ROOT.stave() )
                ########################################
                
                ### check parity
                iseven = (int(pkl_event.trigger)%2==0)
                
                
                ### calculate the average occupancies
                avgnpix = 0
                avgncls = 0
                if( len(pkl_event.npixels)==len(cfg["detectors"]) and len(pkl_event.nclusters)==len(cfg["detectors"])):
                    for det in cfg["detectors"]:
                        avgnpix += pkl_event.npixels[det]
                        avgncls += pkl_event.nclusters[det]
                    avgnpix /= len(cfg["detectors"])
                    avgncls /= len(cfg["detectors"])
                else:
                    print("---------------------------------------------------------------------------------------")
                    print(f"Problem with length of pixels array {len(pkl_event.npixels)} or clusters array {len(pkl_event.nclusters)}")
                    print("---------------------------------------------------------------------------------------")
                
                ### some counters
                tracks_triggers_dict["all"]["trgs"]["all"] += 1
                tracks_triggers_dict["all"]["pix"]["all"]  += avgnpix
                tracks_triggers_dict["all"]["cls"]["all"]  += avgncls
                if(iseven):
                    tracks_triggers_dict["even"]["trgs"]["all"] += 1
                    tracks_triggers_dict["even"]["pix"]["all"]  += avgnpix
                    tracks_triggers_dict["even"]["cls"]["all"]  += avgncls
                else:
                    tracks_triggers_dict["odd"]["trgs"]["all"] += 1
                    tracks_triggers_dict["odd"]["pix"]["all"]  += avgnpix
                    tracks_triggers_dict["odd"]["cls"]["all"]  += avgncls
                
                ntrigs_actual += 1
                
                nevents += 1
                
                histos["hTriggers"].Fill(0.5)
                
                
                ### counters
                counters_x_trg.append( pkl_event.trigger )
                append_global_counters()
                icounter = len(counters_x_trg)-1
                

                ### skip bad triggers...
                if(not cfg["isMC"] and cfg["runtype"]=="beam" and (int(pkl_event.trigger) in badtriggers)):
                    nbadtrigs_actual += 1
                    continue
                histos["hTriggers"].Fill(1.5)
                
                
                tracks_triggers_dict["all"]["trgs"]["good"] += 1
                tracks_triggers_dict["all"]["pix"]["good"]  += avgnpix
                tracks_triggers_dict["all"]["cls"]["good"]  += avgncls
                if(iseven):
                    tracks_triggers_dict["even"]["trgs"]["good"] += 1
                    tracks_triggers_dict["even"]["pix"]["good"]  += avgnpix
                    tracks_triggers_dict["even"]["cls"]["good"]  += avgncls
                else:
                    tracks_triggers_dict["odd"]["trgs"]["good"] += 1
                    tracks_triggers_dict["odd"]["pix"]["good"]  += avgnpix
                    tracks_triggers_dict["odd"]["cls"]["good"]  += avgncls


                ### check errors
                if(not cfg["isMC"]):
                    if(len(pkl_event.errors)!=len(cfg["detectors"])): continue
                    nErrors = 0
                    for det in cfg["detectors"]: nErrors += len(pkl_event.errors[det])
                    if(nErrors>0): continue

                
                ### check pixels
                # if(len(pkl_event.pixels)!=len(cfg["detectors"])): continue
                if(len(pkl_event.npixels)!=len(cfg["detectors"])): continue
                n_pixels = 0
                pass_pixels = True
                for det in cfg["detectors"]:
                    #npix = len( pkl_event.pixels[det] )
                    npix = pkl_event.npixels[det]
                    if(npix==0): pass_pixels = False
                    n_pixels += npix
                set_global_counter("Pixels/chip",icounter,n_pixels/Ndet)
                if(not pass_pixels): continue
            


                ### check clusters
                # if(len(pkl_event.clusters)!=len(cfg["detectors"])): continue
                if(len(pkl_event.nclusters)!=len(cfg["detectors"])): continue
                n_clusters = 0
                pass_clusters = True
                for det in cfg["detectors"]:
                    # ncls = len(pkl_event.clusters[det])
                    ncls = pkl_event.nclusters[det]
                    if(ncls==0): pass_clusters = False
                    n_clusters += ncls
                set_global_counter("Clusters/chip",icounter,n_clusters/Ndet)
                if(not pass_clusters): continue


                ### check seeds
                n_seeds = len(pkl_event.seeds)
                set_global_counter("Track Seeds",icounter,n_seeds)
                if(n_seeds==0): continue


                ### check tracks
                n_tracks = len(pkl_event.tracks)
                if(n_tracks==0): continue
                

                good_tracks = []
                acceptance_tracks = []
                for track in pkl_event.tracks:
                    
                    ##################################
                    ### first require max cluster ####
                    ##################################
                    if(track.maxcls>cfg["cut_maxcls"]): continue
                    
                    
                    
                    
                    # #####################
                    # ### pixel ROI cut ###
                    # #####################
                    # inROI = True
                    # for det in cfg["detectors"]:
                    #     for pix in track.trkcls[det].pixels:
                    #         # print(f"{det}: pix={pix.x,pix.y}")
                    #         if(pix.x<cfg["cut_ROI_xmin"] or pix.x>cfg["cut_ROI_xmax"]): inROI = False
                    #         if(pix.y<cfg["cut_ROI_ymin"] or pix.y>cfg["cut_ROI_ymax"]): inROI = False
                    #         if(not inROI): break
                    #     if(not inROI): break
                    # if(not inROI): continue
                    
                    
                    ### fill some quantities before alignment
                    if(track.chi2ndof<=cfg["cut_chi2dof"] and pass_geoacc_selection(track)): ##TODO: missing the shared hits cut here...
                        histos["hChi2DoF_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_full_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_mid_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_zoom_alowshrcls"].Fill(track.chi2ndof)
                        histos["hChi2DoF_small_alowshrcls"].Fill(track.chi2ndof)
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
                    
                    
                    #################################################
                    ### refit the track if necessary
                    if(not isAlignedAtProd and isNon0Mislaignment):
                        track = refit(track)
                    ### will be the same if misalignment is 0
                    #################################################


                    if(cfg["isMC"] and cfg["isFakeMC"]):
                        slp = pkl_event.fakemcparticles[0].slp
                        itp = pkl_event.fakemcparticles[0].itp
                        vtx = pkl_event.fakemcparticles[0].vtx
                        histos["hTheta_xz_tru_all"].Fill(slp[0])
                        histos["hTheta_yz_tru_all"].Fill(slp[1])
                    
                    #########################
                    ### then require chi2 ###
                    #########################
                    if(track.chi2ndof>cfg["cut_chi2dof"]): continue ### this is the new chi2!
                    good_tracks.append(track)
                    
                    ### get the coordinates at extreme points in real space and after tilting the detector
                    r0,rN,rW,rF,rD = get_track_point_at_extremes(track)

                    ### the y distance from y=0 in the dipole exit plane
                    dExit = rD[1]
                    
                    ### calculate the fit angles
                    tan_theta_yz = +track.params[1] ### the slope p1x transformed to real space (stays as is)
                    tan_theta_xz = -track.params[3] ### the slope p2x transformed to real space (gets minus sign)
                    thetaf_yz = math.atan(tan_theta_yz) #- cfg["thetax"] ###TODO: check if - or +
                    thetaf_xz = math.atan(tan_theta_xz) #- cfg["thetay"] ###TODO: check if - or +

                    ### fill histos before cuts
                    histos["hF_before_cuts"].Fill(rF[0],rF[1])
                    histos["hD_before_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomin_before_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomout_before_cuts"].Fill(rD[0],rD[1])
                    histos["hW_before_cuts"].Fill(rW[0],rW[1])
                    histos["hTheta_xz_before_cuts"].Fill(thetaf_xz)
                    histos["hTheta_yz_before_cuts"].Fill(thetaf_yz)
                    
                    
                    ##########################################
                    ### require pointing to the pdc window ###
                    ### and the dipole exit aperture       ###
                    ### and inclined up as a positron      ###
                    ##########################################
                    if(not pass_geoacc_selection(track)): continue
                    
                    
                    if(cfg["isMC"] and cfg["isFakeMC"]):
                        slp = pkl_event.fakemcparticles[0].slp
                        itp = pkl_event.fakemcparticles[0].itp
                        vtx = pkl_event.fakemcparticles[0].vtx
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
                    pf = (0.3 * B * LB)/math.sin( thetaf_yz - cfg["thetax"] )
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
                    
                    histos["hF_after_cuts"].Fill(rF[0],rF[1])
                    histos["hD_after_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomin_after_cuts"].Fill(rD[0],rD[1])
                    histos["hD_zoomout_after_cuts"].Fill(rD[0],rD[1])
                    histos["hW_after_cuts"].Fill(rW[0],rW[1])
                    
                    histos["hThetaf_yz"].Fill(thetaf_yz)
                    histos["hThetad_yz"].Fill(thetad_yz)
                    histos["hThetar_yz"].Fill(thetar_yz)
                    
                    histos["hTheta_xz_after_cuts"].Fill(thetaf_xz)
                    histos["hTheta_yz_after_cuts"].Fill(thetaf_yz)
                    
                    histos["hdExit"].Fill(dExit)
                    
                    if(pf>0):
                        histos["hPf"].Fill(pf)
                        histos["hPf_small"].Fill(pf)
                        histos["hPf_zoom"].Fill(pf)
                    if(pd>0):
                        histos["hPd"].Fill(pd)
                        histos["hPd_small"].Fill(pd)
                        histos["hPd_zoom"].Fill(pd)
                    if(pr>0):
                        histos["hPr"].Fill(pr)
                        histos["hPr_small"].Fill(pr)
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
                if(cfg["plot_offline_evtdisp"] and len(good_tracks)>0):
                    fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_offline_{pkl_event.trigger}.pdf")
                    plot_event(pkl_event.meta.run,pkl_event.meta.start,pkl_event.meta.dur,pkl_event.trigger,fevtdisplayname,pkl_event.clusters,pkl_event.tracks,chi2threshold=cfg["cut_chi2dof"])
                
                
                ### the Hough space (for the tunnel widths)
                hzx = ROOT.TH2D("hzx","",pkl_event.hough_space["zx_xbins"],pkl_event.hough_space["zx_xmin"],pkl_event.hough_space["zx_xmax"],  pkl_event.hough_space["zx_ybins"],pkl_event.hough_space["zx_ymin"],pkl_event.hough_space["zx_ymax"])
                hzy = ROOT.TH2D("hzy","",pkl_event.hough_space["zy_xbins"],pkl_event.hough_space["zy_xmin"],pkl_event.hough_space["zy_xmax"],  pkl_event.hough_space["zy_ybins"],pkl_event.hough_space["zy_ymin"],pkl_event.hough_space["zy_ymax"])
                
                ### count selected tracks
                tracks_triggers_dict["all"]["trks"] += len(selected_tracks)
                if(iseven): tracks_triggers_dict["even"]["trks"] += len(selected_tracks)
                else:       tracks_triggers_dict["odd"]["trks"]  += len(selected_tracks)
                
                ### plot some selected tracks
                for itrk,track in enumerate(selected_tracks):
                    
                    ###########################
                    ### fill the eudaq tree ###
                    ###########################
                    for det in cfg["detectors"]:
                        chipid = cfg["det2plane"][det]
                        eudaq_event.st_ev_buffer[0].ch_ev_buffer.push_back( ROOT.chip() )
                        ichip = eudaq_event.st_ev_buffer[0].ch_ev_buffer.size()-1
                        eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id = int(chipid)
                        cls0 = ROOT.TVector3( track.trkcls[det].xmm0,track.trkcls[det].ymm0, track.trkcls[det].zmm )
                        cls1 = ROOT.TVector3( track.trkcls[det].xmm, track.trkcls[det].ymm,  track.trkcls[det].zmm )
                        v0 = transform_to_real_space([ track.trkcls[det].xmm0,track.trkcls[det].ymm0, track.trkcls[det].zmm  ])
                        v1 = transform_to_real_space([ track.trkcls[det].xmm, track.trkcls[det].ymm,  track.trkcls[det].zmm  ])
                        cls2 = ROOT.TVector3( v0[0],v0[1],v0[2] )
                        cls3 = ROOT.TVector3( v1[0],v1[1],v1[2] )
                        
                        eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].cls0.push_back( cls0 )
                        eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].cls1.push_back( cls1 )
                        eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].cls2.push_back( cls2 )
                        eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].cls3.push_back( cls3 )
                        trkpixels = []
                        for pixel in track.trkcls[det].pixels:
                            eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.push_back( ROOT.pixel() )
                            ihit = eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.size()-1
                            eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].ix = pixel.x
                            eudaq_event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].iy = pixel.y
                            trkpixels.append([pixel.x,pixel.y])
                        # print(f"itrk[{itrk}]: chipid={chipid} --> trkpixels={trkpixels}")
                    ###########################
                    
                    # dx,dy = res_track2cluster("ALPIDE_3",track.points,track.direction,track.centroid)
                    # if(dx>-0.02): continue
                    # if(dy>-0.02): continue
                    
                    histos["hChi2DoF_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_full_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_mid_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_zoom_zeroshrcls"].Fill(track.chi2ndof)
                    histos["hChi2DoF_small_zeroshrcls"].Fill(track.chi2ndof)
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
                        
                        ### draw all waves
                        rChip = [track.trkcls[det].xmm,track.trkcls[det].ymm,track.trkcls[det].zmm]
                        xwave = get_wave(rChip[2],rChip[0],thetaxmin,thetaxmax)
                        ywave = get_wave(rChip[2],rChip[1],thetaymin,thetaymax)
                        for btheta in range(1,histos["hWaves_zx"].GetNbinsX()+1):
                            theta = histos["hWaves_zx"].GetXaxis().GetBinCenter(btheta)
                            rhox  = xwave.Eval(theta)
                            rhoy  = ywave.Eval(theta)
                            histos["hWaves_zx"].Fill(theta,rhox)
                            histos["hWaves_zy"].Fill(theta,rhoy)
                        del xwave
                        del ywave
                    
                    ### draw only wave intersections
                    fill_pair(0,1,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(0,2,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(0,3,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(0,4,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(1,2,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(1,3,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(1,4,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(2,3,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(2,4,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    fill_pair(3,4,track,histos["hWaves_zx_intersections"],histos["hWaves_zy_intersections"])
                    
                    ### find the tunnel widths
                    thetax = track.hough_coords[0]
                    rhox   = track.hough_coords[1]
                    thetay = track.hough_coords[2]
                    rhoy   = track.hough_coords[3]
                    bthetax = hzx.GetXaxis().FindBin( thetax )
                    brhox   = hzx.GetXaxis().FindBin( rhox   )
                    bthetay = hzy.GetXaxis().FindBin( thetay )
                    brhoy   = hzy.GetXaxis().FindBin( rhoy   )
                    arr_thetax = [ hzx.GetXaxis().GetBinLowEdge(bthetax), hzx.GetXaxis().GetBinUpEdge(bthetax) ]
                    arr_rhox   = [ hzx.GetYaxis().GetBinLowEdge(brhox),   hzx.GetYaxis().GetBinUpEdge(brhox)   ]
                    arr_thetay = [ hzy.GetXaxis().GetBinLowEdge(bthetay), hzy.GetXaxis().GetBinUpEdge(bthetay) ]
                    arr_rhoy   = [ hzy.GetYaxis().GetBinLowEdge(brhoy),   hzy.GetYaxis().GetBinUpEdge(brhoy)   ]
                    for det in cfg["detectors"]:
                        xmin,xmax,ymin,ymax = get_edges_from_theta_rho_corners(det,arr_thetax,arr_rhox,arr_thetay,arr_rhoy)
                        histos[f"h_tunnel_width_x_{det}"].Fill(xmax-xmin)
                        histos[f"h_tunnel_width_y_{det}"].Fill(ymax-ymin)
                
                
                ### at the end of the pkl_event, clean the Hough space histos
                del hzx
                del hzy
                
                ###########################
                ### fill the eudaq tree ###
                tEUDAQout.Fill()
                ###########################
                
                print(f"Event[{nevents-1}], Trigger[{pkl_event.trigger}] --> Good tracks: {len(good_tracks)}, Acceptance tracks: {len(acceptance_tracks)}, Selected tracks: {len(selected_tracks)}")

    # print(f"Events:{nevents}, Tracks:{ntracks}")
    print(f"Tracks:{ntracks}, GoodTriggers:{nevents-nbadtrigs_actual}, Actual triggers: {ntrigs_actual} (with AllTriggers:{nevents} and BadTriggers in the range: {nbadtrigs_actual} (or {nbadtrigs} in the full run))")
    
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
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hD_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hD_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetGridx()
    histos["hD_zoomin_before_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetGridx()
    histos["hD_zoomin_after_cuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")

    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hF_before_cuts"].Draw("colz")
    flange.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hF_after_cuts"].Draw("colz")
    flange.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hW_before_cuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    histos["hW_after_cuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_xz_before_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_xz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_xz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_xz_after_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_xz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_xz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_yz_before_cuts"].Draw("hist")
    if(cfg["isMC"] and cfg["isFakeMC"]):
        histos["hTheta_yz_tru"].SetLineColor(ROOT.kRed)
        histos["hTheta_yz_tru"].Draw("hist same")
    cnv.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["hTheta_yz_after_cuts"].Draw("hist")
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
    histos["hPf_small"].Draw("hist")
    cnv.RedrawAxis()
    cnv.cd(2)
    # ROOT.gPad.SetTicks(1,1)
    # histos["hPd_small"].Draw("hist")
    # cnv.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetTicks(1,1)
    histos["hPr_small"].Draw("hist")
    cnv.RedrawAxis()
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
    
    
    leg = ROOT.TLegend(0.3,0.8,0.7,0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    leg.AddEntry(histos["hChi2DoF_full_alowshrcls"],"Baseline w/shared clusters","l")
    leg.AddEntry(histos["hChi2DoF_full_zeroshrcls"],"Baseline w/o shared clusters","l")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_full_alowshrcls"],histos["hChi2DoF_full_zeroshrcls"])
    histos["hChi2DoF_full_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_full_zeroshrcls"].SetMaximum(1.1*hmax)  
    histos["hChi2DoF_full_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_full_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_full_alowshrcls"].Draw("hist")
    histos["hChi2DoF_full_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_mid_alowshrcls"],histos["hChi2DoF_mid_zeroshrcls"])
    histos["hChi2DoF_mid_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_mid_zeroshrcls"].SetMaximum(1.1*hmax)  
    histos["hChi2DoF_mid_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_mid_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_mid_alowshrcls"].Draw("hist")
    histos["hChi2DoF_mid_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
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
    leg.Draw("same")
    cnv.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",500,500)
    cnv.SetTicks(1,1)
    hmax = h1h2max(histos["hChi2DoF_small_alowshrcls"],histos["hChi2DoF_small_zeroshrcls"])
    histos["hChi2DoF_small_alowshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_small_zeroshrcls"].SetMaximum(1.1*hmax)
    histos["hChi2DoF_small_alowshrcls"].SetLineColor(ROOT.kBlack)
    histos["hChi2DoF_small_zeroshrcls"].SetLineColor(ROOT.kRed)
    histos["hChi2DoF_small_alowshrcls"].Draw("hist")
    histos["hChi2DoF_small_zeroshrcls"].Draw("hist same")
    leg.Draw("same")
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
    leg.Draw("same")
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
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
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
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
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
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
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
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
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
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
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
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",2500,500)
    cnv.Divide(5,1)
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
    cnv.SaveAs(f"{foupdfname}")
    
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,1000)
    cnv.Divide(2,2)
    cnv.cd(1)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zx"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zy"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.cd(3)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zx_intersections"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.cd(4)
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    histos["hWaves_zy_intersections"].Draw("colz")
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs(f"{foupdfname}")
    


    leg = ROOT.TLegend(0.2,0.7,0.55,0.8)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.037)
    leg.SetBorderSize(0)
    cnv = ROOT.TCanvas("cnv_dipole_window","",1500,1000)
    cnv.Divide(3,2)
    for idet,det in enumerate(cfg["detectors"]):
        cnv.cd(idet+1)
        ROOT.gPad.SetTicks(1,1)
        histos[f"h_tunnel_width_x_{det}"].SetMinimum(0)
        histos[f"h_tunnel_width_y_{det}"].SetMinimum(0)
        hbmax = histos[f"h_tunnel_width_x_{det}"].GetMaximum()
        hamax = histos[f"h_tunnel_width_y_{det}"].GetMaximum()
        hmax = hbmax if(hbmax>hamax) else hamax
        hmax *= 1.2
        histos[f"h_tunnel_width_x_{det}"].SetMaximum(hmax)
        histos[f"h_tunnel_width_y_{det}"].SetMaximum(hmax)
        histos[f"h_tunnel_width_x_{det}"].SetLineColor(ROOT.kBlack)
        histos[f"h_tunnel_width_x_{det}"].Draw("hist")
        histos[f"h_tunnel_width_y_{det}"].SetLineColor(ROOT.kRed)
        histos[f"h_tunnel_width_y_{det}"].Draw("hist same")
        if(idet==0):
            leg.AddEntry(histos[f"h_tunnel_width_x_{det}"],"k=x","l")
            leg.AddEntry(histos[f"h_tunnel_width_y_{det}"],"k=y","l")
        leg.Draw("same")
        
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
    
    ########################
    ### write eudaq file ###
    ########################
    fEUDAQout.cd()
    tEUDAQout.Write()
    tEUDAQoutMeta.Write()
    fEUDAQout.Write()
    fEUDAQout.Close()
    ########################
    
    
    ### summary of tracking
    # print(f"\nTracks:{ntracks}, GoodTriggers:{nevents-nbadtrigs}  (with AllTriggers:{nevents} and BadTriggers: {nbadtrigs})")
    print(f"\nTracks:{ntracks}, GoodTriggers:{nevents-nbadtrigs_actual} Actual triggers: {ntrigs_actual} (with AllTriggers:{nevents} and BadTriggers in the range: {nbadtrigs_actual} (or {nbadtrigs} in the full run))")
    
    
    tracks_triggers_dict["all"]["pix"]["all"]  /= tracks_triggers_dict["all"]["trgs"]["all"]
    tracks_triggers_dict["all"]["cls"]["all"]  /= tracks_triggers_dict["all"]["trgs"]["all"]
    tracks_triggers_dict["all"]["pix"]["good"] /= tracks_triggers_dict["all"]["trgs"]["good"]
    tracks_triggers_dict["all"]["cls"]["good"] /= tracks_triggers_dict["all"]["trgs"]["good"]

    tracks_triggers_dict["even"]["pix"]["all"]  /= tracks_triggers_dict["even"]["trgs"]["all"]
    tracks_triggers_dict["even"]["cls"]["all"]  /= tracks_triggers_dict["even"]["trgs"]["all"]
    tracks_triggers_dict["even"]["pix"]["good"] /= tracks_triggers_dict["even"]["trgs"]["good"]
    tracks_triggers_dict["even"]["cls"]["good"] /= tracks_triggers_dict["even"]["trgs"]["good"]

    tracks_triggers_dict["odd"]["pix"]["all"]  /= tracks_triggers_dict["odd"]["trgs"]["all"]
    tracks_triggers_dict["odd"]["cls"]["all"]  /= tracks_triggers_dict["odd"]["trgs"]["all"]
    tracks_triggers_dict["odd"]["pix"]["good"] /= tracks_triggers_dict["odd"]["trgs"]["good"]
    tracks_triggers_dict["odd"]["cls"]["good"] /= tracks_triggers_dict["odd"]["trgs"]["good"]
    
    
    
    print(f"\ncounters before: {tracks_triggers_dict}")
    def convert_all_to_ints(d):
        if isinstance(d, dict):
            return {k: convert_all_to_ints(v) for k, v in d.items()}
        elif isinstance(d, float) or isinstance(d, int):
            return int(d)
        else:
            return d
    tracks_triggers_dict = convert_all_to_ints(tracks_triggers_dict)
    print(f"\ncounters: {tracks_triggers_dict}")
    
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print(f'ֿֿ\nExecution time: {elapsed_time} seconds')