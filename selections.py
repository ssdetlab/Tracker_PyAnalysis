#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

import config
from config import *
import utils
from utils import *


def diamond_cut(xL,xR,yT,x,y,tol=4):
    YBL = yofx([0,0],[xL,yT],x)
    YTL = yofx([xL,0],[0,yT],x)
    YTR = yofx([0,yT],[xR,0],x)
    YBR = yofx([xR,yT],[0,0],x)
    if(y<YBL-tol): return False
    if(y>YTL+tol): return False
    if(y>YTR+tol): return False
    if(y<YBR-tol): return False
    return True
    

def spot_cut(x,y):
    X = x-cfg["cut_spot_xcenter"]
    Y = y-cfg["cut_spot_ycenter"]
    R = cfg["cut_spot_radius"]
    if( math.sqrt(X*X+Y*Y)>R ): return False
    if( math.sqrt(X*X+Y*Y)>R ): return False
    if( math.sqrt(X*X+Y*Y)>R ): return False
    if( math.sqrt(X*X+Y*Y)>R ): return False
    return True
    

def pass_geoacc_selection(track):
    ## r0: first detector, rN: last detector, rW: window, rD: dipole exit
    r0,rN,rW,rF,rD = get_track_point_at_extremes(track)
    xWinL,xWinR,yWinB,yWinT = get_pdc_window_bounds()
    xDipL,xDipR,yDipB,yDipT = get_dipole_exit_bounds()
    xFlgL,xFlgR,yFlgB,yFlgT = get_dipole_flange_bounds()
    pass_inclination_yz  = ( rN[1]>=r0[1]  and r0[1]>=rW[1]  and rN[1]>=rW[1] )
    pass_vertexatpdc     = ( (rW[0]>=xWinL and rW[0]<=xWinR) and (rW[1]>=yWinB and rW[1]<=yWinT) )
    pass_dipole_aperture = ( (rD[0]>=xDipL and rD[0]<=xDipR) and (rD[1]>0 and rD[1]<=yDipT) )
    pass_flange_aperture = ( (rF[0]>=xFlgL and rF[0]<=xFlgR) and (rF[1]>0 and rF[1]<=yFlgT) )
    pass_dipole_spot     = ( spot_cut(rD[0],rD[1]) ) if(cfg["cut_spot"]) else True
    pass_dipole_Eslot    = ( rD[1]>7.9 and rD[1]<15.5 )
    pass_dipole_Xslot    = ( rD[0]>-5  and rD[0]<+5 )
    return (pass_inclination_yz and pass_vertexatpdc and pass_dipole_aperture and pass_flange_aperture and pass_dipole_spot)


def remove_tracks_with_shared_clusters(tracks):
    clsid_to_trackidx = {}
    for det in cfg["detectors"]:
        clsid_to_trackidx.update({det:{}})
        
    for itrk,track in enumerate(tracks):
        for det in cfg["detectors"]:
            CID = track.trkcls[det].CID
            if(CID not in clsid_to_trackidx[det]):
                clsid_to_trackidx[det].update({CID:itrk})
            else:
                itrk0 = clsid_to_trackidx[det][CID]
                # print(f"found shared cluster for CID={CID}: itrk1={itrk}(chi2={track.chi2ndof}), itrk2={itrk0}(chi2={tracks[itrk0].chi2ndof})")
                if(tracks[itrk0].chi2ndof>track.chi2ndof):
                    clsid_to_trackidx[det][CID] = itrk
    
    passing_tracks_idx = []
    passing_tracks = []
    det0 = cfg["detectors"][0]
    for CID,itrk in clsid_to_trackidx[det0].items():
        if(itrk not in passing_tracks_idx):
            noccurancees = 1
            for i in range(1,len(cfg["detectors"])):
                deti = cfg["detectors"][i]
                noccurancees += (itrk in clsid_to_trackidx[deti].values())
            if(noccurancees!=len(cfg["detectors"])): continue
            passing_tracks_idx.append(itrk)
            passing_tracks.append(tracks[itrk])
    
    return passing_tracks