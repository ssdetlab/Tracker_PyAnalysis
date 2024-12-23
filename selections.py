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


def pass_slope_and_window_selection(track):
    ## r0: first detector, rN: last detector, rW: window
    r0,rN,rW = get_track_point_at_extremes(track)
    xWinL,xWinR,yWinB,yWinT = get_pdc_window_bounds()
    pass_inclination_yz = (rN[1]>=r0[1] and r0[1]>=rW[1] and rN[1]>=rW[1])
    pass_vertexatpdc    = ((rW[0]>=xWinL and rW[0]<=xWinR) and (rW[1]>=yWinB and rW[1]<=yWinT))
    return (pass_inclination_yz and pass_vertexatpdc)


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
                print(f"found shared cluster for CID={CID}: itrk1={itrk}(chi2={track.chi2ndof}), itrk2={itrk0}(chi2={tracks[itrk0].chi2ndof})")
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