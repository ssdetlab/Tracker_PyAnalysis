#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )

detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
planes    = [8,6,4,2,0]
plane2det = {8:"ALPIDE_0", 6:"ALPIDE_1", 4:"ALPIDE_2", 2:"ALPIDE_3", 0:"ALPIDE_4"}

tfile = ROOT.TFile("tree_with_HT_selected_tracks_only_Run502.root","READ")
ttree = tfile.Get("MyTree")
nevents = ttree.GetEntries()
print(f"nevents={nevents}")


for ievt in range(nevents):
    ### get the event
    ttree.GetEntry(ievt)
    ### get the staves
    staves = ttree.event.st_ev_buffer
    ### get the chips
    for istv in range(staves.size()):
        staveid  = staves[istv].stave_id
        chips    = staves[istv].ch_ev_buffer
        for ichp in range(chips.size()):
            chipid   = chips[ichp].chip_id
            if(chipid not in planes): continue
            detector = plane2det[chipid]
            nhits    = chips[ichp].hits.size()
            print(f"chipid: {chipid} det: {detector} --> npixels: {nhits}")
            for ipix in range(nhits):
                ix = chips[ichp].hits[ipix].ix
                iy = chips[ichp].hits[ipix].iy
                print(f"   pixel[{ipix}]: ({ix},{iy})")
