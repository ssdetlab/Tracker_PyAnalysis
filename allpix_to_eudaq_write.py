#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT
import glob

runname = "AallPix2_mc_prototype_beam_beryllium_window"
runnum  = 0
srunnum = "000"

def transform(ix,iy):
    return (1023-ix,511-iy)

### declare the data tree and its classes
ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
### declare the meta-data tree and its classes
ROOT.gROOT.ProcessLine("struct run_meta_data  { Int_t run_number; Double_t run_start; Double_t run_end; };" )
### the main root gile
fOut = ROOT.TFile.Open(f"{runname}_Run{srunnum}.root", "RECREATE")
### data tree
tOut = ROOT.TTree("MyTree","")
event = ROOT.event()
tOut.Branch("event", event)
### meta-data tree
tOutMeta = ROOT.TTree("MyTreeMeta","")
run_meta_data = ROOT.run_meta_data()
tOutMeta.Branch("run_meta_data", run_meta_data)


### treename_J_K_B with J=2 digit stave number, K=2 digit chip number and B=is the bx number
# pixels_00_00_1=ALPIDE_0=slot#8
# pixels_03_00_1=ALPIDE_1=slot#5
# pixels_05_00_1=ALPIDE_2=slot#3
# pixels_07_00_1=ALPIDE_3=slot#1

### get the files
basedir = "../../Downloads/E320BackgroundFromTrackerPrototype_NewTreeStructure"
files = glob.glob(f"{basedir}/*.root")
files = sorted(files, key = lambda x: x.split("_Event")[1])
event_list = {}
for fname in files:
    fonly = fname.replace("../../Downloads/E320BackgroundFromTrackerPrototype_NewTreeStructure/","")
    print(f"file: {fonly}")
    Event = int(fname.split("_Event")[1].split(".root")[0])
    if(Event not in event_list): event_list.update({Event:[fname]})
    else:                        event_list[Event].append(fname)

### fill meta-data tree
run_meta_data.run_number = runnum ### dummy
run_meta_data.run_start  = -1.    ### dummy
run_meta_data.run_end    = -1.    ### dummy
tOutMeta.Fill()

### fill data tree
chipmc2det = {"00":"ALPIDE_0", "30":"ALPIDE_1", "50":"ALPIDE_2", "70":"ALPIDE_3"}
chipmc2chipid = {"00":"00", "30":"03", "50":"05", "70":"07"} ##TODO: this is due to a bug in the name of the file
for Event,flist in event_list.items():
    ##############################
    ### nicely clear per event
    for s in range(event.st_ev_buffer.size()):
        for c in range(event.st_ev_buffer[s].ch_ev_buffer.size()):
            event.st_ev_buffer[s].ch_ev_buffer[c].hits.clear()
        event.st_ev_buffer[s].ch_ev_buffer.clear()
    event.st_ev_buffer.clear() ###
    ##############################
    event.trg_n    = Event
    event.ts_begin = -1.
    event.ts_end   = -1.
    event.st_ev_buffer.push_back( ROOT.stave() )
    ##############################
    for fname in flist:
        print(f"Event:{Event} --> file: {fname}")
        chipmc = fname.split("_Stave")[1].split("_")[0]
        chipid = chipmc2chipid[chipmc] ##TODO: this is due to a bug in the name of the file
        det = chipmc2det[chipmc]
        fIn = ROOT.TFile(fname,"READ")
        ##############################
        tpixels = fIn.Get(f"pixels_{chipid}_00_{Event}")
        print(f"Npixels for {det}=(chipid={chipid}): {tpixels.GetEntries()}")
        event.st_ev_buffer[0].ch_ev_buffer.push_back( ROOT.chip() )
        ichip = event.st_ev_buffer[0].ch_ev_buffer.size()-1
        event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id = int(chipid)
        for pix in tpixels:
            print(f"Event:{Event} --> chipid={chipid}, int(chipid)={int(chipid)} --> (ix,iy)=({pix.cellx},{pix.celly}) --> (x,y,z)=({pix.globalposx:.5f},{pix.globalposy:.5f},{pix.globalposz:.5f})")
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
tOutMeta.Write()
fOut.Write()
fOut.Close()

