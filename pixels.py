#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

import config
from config import *
import objects
from objects import *


def get_all_pixles_eudaq(evt,hPixMatrix,ROI={}):
    if(cfg["isMC"]):
        print("Cannot call the get_all_pixles_eudaq function if isMC=1.")
        print("Quitting.")
        quit()
    staves = evt.event.st_ev_buffer
    pixels = {}
    raws   = {}
    ids2d  = {}
    for det in cfg["detectors"]:
        pixels.update({det:[]})
        raws.update({det:[]})
        ids2d.update({det:[]})
    n_active_staves = 0
    n_active_chips  = 0
    for istv in range(staves.size()):
        staveid  = staves[istv].stave_id
        chips    = staves[istv].ch_ev_buffer
        isactivestave = True
        for ichp in range(chips.size()):
            chipid   = chips[ichp].chip_id
            detector = cfg["plane2det"][chipid]
            nhits    = chips[ichp].hits.size()
            n_active_chips += (nhits>0)
            for ipix in range(nhits):
                ix,iy = chips[ichp].hits[ipix]
                if(len(ROI)>0):
                    if(("ix" in ROI) and (ix<ROI["ix"]["min"] or ix>ROI["ix"]["max"])): continue
                    if(("iy" in ROI) and (iy<ROI["iy"]["min"] or iy>ROI["iy"]["max"])): continue
                # iy,ix = chips[ichp].hits[ipix] ##TODO: this is swapped!!!
                raw = hPixMatrix[detector].FindBin(ix,iy)
                # if(raw not in raws[detector]): ### This is a bad condition since the raw number can fall in underflow/overflow bin and hence ignored multiple times if e.g. the x-y coordinates are flipped. That's why I moved to the (x,y) tuple insertion instead of the raw pixel number.
                id2d = (ix,iy)
                if(id2d not in ids2d[detector]):
                    ids2d[detector].append(id2d)
                    raws[detector].append(raw)
                    pixels[detector].append( Hit(detector,ix,iy,raw) )
            n_active_staves += (isactivestave)
    return n_active_staves,n_active_chips,pixels

    
# def get_all_pixles_mc(fIn,Event,hPixMatrix,ROI={}):
#     if(not cfg["isMC"]):
#         print("Cannot call the get_all_pixles_eudaq function if isMC=0.")
#         print("Quitting.")
#         quit()
#     pixels = {}
#     raws   = {}
#     ids2d  = {}
#     chipmc2det = {"00":"ALPIDE_0", "03":"ALPIDE_1", "05":"ALPIDE_2", "07":"ALPIDE_3"}
#     for det in cfg["detectors"]:
#         pixels.update({det:[]})
#         raws.update({det:[]})
#         ids2d.update({det:[]})
#     n_active_staves = 0
#     n_active_chips  = 0
#
#     for chipmc,det in chipmc2det.items():
#         tpixels = fIn.Get(f"pixels_{chipmc}_00_{Event}")
#         detector = det
#         isactivestave = True
#         for ichp in range(chips.size()):
#             nhits = tpixels.GetEntries()
#             n_active_chips += (nhits>0)
#             for pix in tpixels:
#                 ix = pix.cellx
#                 iy = pix.celly
#                 if(len(ROI)>0):
#                     if(("ix" in ROI) and (ix<ROI["ix"]["min"] or ix>ROI["ix"]["max"])): continue
#                     if(("iy" in ROI) and (iy<ROI["iy"]["min"] or iy>ROI["iy"]["max"])): continue
#                 raw = hPixMatrix[detector].FindBin(ix,iy)
#                 id2d = (ix,iy)
#                 if(id2d not in ids2d[detector]):
#                     ids2d[detector].append(id2d)
#                     raws[detector].append(raw)
#                     pixels[detector].append( Hit(detector,ix,iy,raw) )
#             n_active_staves += (isactivestave)
#     return n_active_staves,n_active_chips,pixels


def get_all_pixles(evt,hPixMatrix,ROI={}):
    n_active_planes = -1
    n_active_chips  = -1
    pixels = {}
    if(not cfg["isMC"]): n_active_staves, n_active_chips, pixels = get_all_pixles_eudaq(evt,hPixMatrix,ROI)
    # else:                n_active_staves, n_active_chips, pixels = get_all_pixles_mc(evt,hPixMatrix,ROI)
    else:
        print("NEED TO RE-IMPLEMENT THE MC READOUT.")
        print("QUITTING.")
        quit()
    return n_active_staves, n_active_chips, pixels