#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT
import glob

def transform(ix,iy):
    return (iy,ix)

ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )

fOut = ROOT.TFile.Open("my.root", "RECREATE")
tOut = ROOT.TTree("MyTree","my tree")
event = ROOT.event()
tOut.Branch("event", event)


# fPattern = "dataFile_Signal_E320lp_1perBX_EFieldV18p1N0p5Vpercm_Processed_Stave*_Event501.root"
# files = glob.glob("../../Downloads/"+fPattern)
#
# for fname in files:
#     print(f"file: {fname}")
#     fIn = ROOT.TFile(fname,"READ")
#     tpixels = fIn.Get("pixels")
#     print(f"Npixels: {tpixels.GetEntries()}")
#     for evt in tpixels:
#         print(f"(ix,iy)=({evt.cellx},{evt.celly}) --> (x,y)=({evt.globalposx:.5f},{evt.globalposy:.5f})")
#         ix,iy = transform(evt.cellx,evt.celly)


### treename_J_K_B with J=2 digit stave number, K=2 digit chip number and B=is the bx number
# pixels_00_00_1=ALPIDE_0=slot#8
# pixels_03_00_1=ALPIDE_1=slot#5
# pixels_05_00_1=ALPIDE_2=slot#3
# pixels_07_00_1=ALPIDE_3=slot#1

Event = 1
chipmc2det = {"00":"ALPIDE_0", "03":"ALPIDE_1", "05":"ALPIDE_2", "07":"ALPIDE_3"}
fname = f"dataFile_E320BackgroundFromTrackerPrototype_EFieldV18p1N0p5Vpercm_Processed_Event{Event}_merged.root"

print(f"file: {fname}")
fIn = ROOT.TFile(fname,"READ")

##############################
event.st_ev_buffer.clear() ###
##############################
event.trg_n    = Event
event.ts_begin = -1.
event.ts_end   = -1.
event.st_ev_buffer.push_back( ROOT.stave() )
##############################
for chipmc,det in chipmc2det.items():
    tpixels = fIn.Get(f"pixels_{chipmc}_00_{Event}")
    print(f"Npixels for {det}=(chipmc={chipmc}): {tpixels.GetEntries()}")
    event.st_ev_buffer[0].ch_ev_buffer.push_back( ROOT.chip() )
    ichip = event.st_ev_buffer[0].ch_ev_buffer.size()-1
    event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id = int(chipmc)
    for pix in tpixels:
        print(f"chipmc={chipmc}, int(chipmc)={int(chipmc)} --> (ix,iy)=({pix.cellx},{pix.celly}) --> (x,y)=({pix.globalposx:.5f},{pix.globalposy:.5f})")
        ix,iy = transform(pix.cellx,pix.celly)
        event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.push_back( ROOT.pixel() )
        ihit = event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.size()-1
        event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].ix = ix
        event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].iy = iy
### fill every event
tOut.Fill()

### finish
fOut.cd()
tOut.Write()
fOut.Write()
fOut.Close()

