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


if(cfg["isMC"]):
    # print("Building the classes for MC")
    ### declare the data tree and its classes
    ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; Float_t xFake; Float_t yFake; };" )
    ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
    ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
    ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )

def get_all_pixles(evt,hPixMatrix,ROI={}):
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
            if(chipid not in cfg["planes"]): continue
            detector = cfg["plane2det"][chipid]
            nhits    = chips[ichp].hits.size()
            n_active_chips += (nhits>0)
            for ipix in range(nhits):
                ix = -1
                iy = -1
                xFake = 0
                yFake = 0
                if(not cfg["isMC"]):
                    ### EUDAQ
                    ix,iy = chips[ichp].hits[ipix]
                elif(cfg["isMC"] and not cfg["isFakeMC"]):
                    ### AllPix converted to EUDAQ
                    ix = chips[ichp].hits[ipix].ix 
                    iy = chips[ichp].hits[ipix].iy
                elif(cfg["isMC"] and cfg["isFakeMC"]):
                    ### fake mc converted to EUDAQ
                    ix = chips[ichp].hits[ipix].ix 
                    iy = chips[ichp].hits[ipix].iy
                    xFake = chips[ichp].hits[ipix].xFake
                    yFake = chips[ichp].hits[ipix].yFake
                    # print(f"ix={ix}, iy={iy},   xFake={xFake}, yFake={yFake}")
                    
                if(len(ROI)>0):
                    if(("ix" in ROI) and (ix<ROI["ix"]["min"] or ix>ROI["ix"]["max"])): continue
                    if(("iy" in ROI) and (iy<ROI["iy"]["min"] or iy>ROI["iy"]["max"])): continue
                raw = hPixMatrix[detector].FindBin(ix,iy)
                id2d = (ix,iy)
                if(id2d not in ids2d[detector]):
                    ids2d[detector].append(id2d)
                    raws[detector].append(raw)
                    pixels[detector].append( Hit(detector,ix,iy,raw) if(not cfg["isFakeMC"]) else Hit(detector,ix,iy,raw,xFake,yFake) )
            n_active_staves += (isactivestave)
    return n_active_staves,n_active_chips,pixels
