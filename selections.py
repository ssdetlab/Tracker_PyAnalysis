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
    

def tilted_eliptic_RoI_cut(track):
    X0 = cfg["cut_RoI_spot_xcenter"]
    Y0 = cfg["cut_RoI_spot_ycenter"]
    a = cfg["cut_RoI_spot_radius_x"]
    b = cfg["cut_RoI_spot_radius_y"]
    t = cfg["cut_RoI_spot_theta_deg"]*np.pi/180.
    A = (a*math.sin(t))**2 + (b*math.cos(t))**2
    B = 2*(b**2-a**2)*math.sin(t)*math.cos(t)
    C = (a*math.cos(t))**2 + (b*math.sin(t))**2
    D = -2*A*X0 - B*Y0
    E = -B*X0 - 2*C*Y0
    F = A*(X0**2) + B*X0*Y0 + C*(Y0**2) - (a*b)**2
    for det in cfg["detectors"]:
            x = track.trkcls[det].x ### cluster center measured in pixels in the EUDAQ frame
            y = track.trkcls[det].y ### cluster center measured in pixels in the EUDAQ frame 
            elipse = A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F
            if(elipse>0.): return False
    return True

def spot_cut(x,y):
    CX = cfg["cut_spot_xcenter"]
    CY = cfg["cut_spot_ycenter"]
    RX = cfg["cut_spot_radius_x"]
    RY = cfg["cut_spot_radius_y"]
    X = (x-CX)/RX
    Y = (y-CY)/RY
    X2 = X*X
    Y2 = Y*Y
    if( (X2+Y2)>1. ): return False
    return True

def strip_cut(x,y):
    CX = cfg["cut_spot_xcenter"]
    CY = cfg["cut_spot_ycenter"]
    SX = cfg["cut_strip_x"]
    SY = cfg["cut_strip_y"]
    if( x<CX-SX or x>CX+SX ): return False
    if( y<CY-SY or y>CY+SY ): return False
    return True
    

def pass_dk_at_detector(track,detector,dxMin=-999,dxMax=+999,dyMin=-999,dyMax=+999):
    if(not cfg["use_large_dk_filter"]): return True
    dx,dy = res_track2cluster(detector,track.points,track.direction,track.centroid)
    if(dx<dxMin or dx>dxMax): return False
    if(dy<dyMin or dy>dyMax): return False
    return True
    

def pass_geoacc_selection(track):
    ## r0: first detector, rN: last detector, rW: window, rD: dipole exit
    r0,rN,rW,rF,rD = get_track_point_at_extremes(track)
    xWinL,xWinR,yWinB,yWinT = get_pdc_window_bounds()
    xDipL,xDipR,yDipB,yDipT = get_dipole_exit_bounds()
    xFlgL,xFlgR,yFlgB,yFlgT = get_dipole_flange_bounds()
    
    psss_RoI             = ( tilted_eliptic_RoI_cut(track) ) if(cfg["cut_RoI_spot"]) else True
    pass_inclination_yz  = ( rN[1]>=r0[1]  and r0[1]>=rW[1]  and rN[1]>=rW[1] )
    pass_vertexatpdc     = ( (rW[0]>=xWinL and rW[0]<=xWinR) and (rW[1]>=yWinB and rW[1]<=yWinT) )
    pass_flange_aperture = ( (rF[0]>=xFlgL and rF[0]<=xFlgR) and (rF[1]>0 and rF[1]<=yFlgT) )
    pass_dipole_aperture = ( (rD[0]>=xDipL and rD[0]<=xDipR) and (rD[1]>0 and rD[1]<=yDipT) )
    pass_dipole_spot     = ( spot_cut(rD[0],rD[1])  ) if(cfg["cut_spot"])  else True
    pass_dipole_strip    = ( strip_cut(rD[0],rD[1]) ) if(cfg["cut_strip"]) else True
    pass_dk_at_det       = ( pass_dk_at_detector(track,"ALPIDE_3",dxMax=-0.02,dyMax=-0.02) ) ### TODO(need to make a flag for this) RELEVANT ONLY FOR PRE-ALIGNMENT!!!
    print(f"psss_RoI={psss_RoI}, pass_inclination_yz={pass_inclination_yz}, pass_vertexatpdc={pass_vertexatpdc}, pass_flange_aperture={pass_flange_aperture}, pass_dipole_aperture={pass_dipole_aperture}, pass_dipole_spot={pass_dipole_spot}, pass_dipole_strip={pass_dipole_strip}, pass_dk_at_det={pass_dk_at_det}")
    return (psss_RoI and
            pass_inclination_yz and
            pass_vertexatpdc and
            pass_flange_aperture and
            pass_dipole_aperture and
            pass_dipole_spot and
            pass_dipole_strip and
            pass_dk_at_det)


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