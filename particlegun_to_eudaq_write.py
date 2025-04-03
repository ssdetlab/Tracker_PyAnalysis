#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT
import glob

import argparse
parser = argparse.ArgumentParser(description='particlegun_to_eudaq_write.py...')
parser.add_argument('-conf', metavar='config file',  required=True,   help='full path to config file')
argus = parser.parse_args()
configfile = argus.conf
Nevents    = 10000
Nprtperevt = 1

import config
from config import *
### must be called here (first) and only once!
init_config(configfile,False)
# print config once
show_config()

import gun
from gun import*
import objects
from objects import *
import utils
from utils import *

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


runname = "tree_FakeParticles_particlegun_run"
runnum  = 1
srunnum = "001"

hPixelMatrix = ROOT.TH2D("pixelmatrix",";x_{pix} [mm];y_{pix} [mm]", cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2.)

### declare the data tree and its classes
ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; Float_t xFake; Float_t yFake; };" )
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



### fill meta-data tree
run_meta_data.run_number = runnum ### dummy
run_meta_data.run_start  = -1.    ### dummy
run_meta_data.run_end    = -1.    ### dummy
tOutMeta.Fill()


### init the particle gun
vtxsurf = {"x":[-21,+21], "y":[0,30], "z":cfg["zDipoleExit"]}
slopes  = {"xz":[-5e-3,+5e-3], "yz":[1e-2,3e-2]}    
gun = ParticleGun(vtxsurf,slopes)

Lxmin = gun.layers[0][0][0]*1.2
Lymin = gun.layers[0][0][1]*1.2
Lxmax = gun.layers[0][2][0]*1.2
Lymax = gun.layers[0][2][1]*1.2

histos = {}
ROOT.TH2D()
hname = "dipole_precuts";  histos.update({hname:ROOT.TH2D(hname,"Dipole exit plane (pre-cuts);x [mm];y [mm];Fake Tracks",120,-80,+80, 120,-70,+90)})
hname = "dipole_postcuts"; histos.update({hname:ROOT.TH2D(hname,"Dipole exit plane (post-cuts);x [mm];y [mm];Fake Tracks",120,-80,+80, 120,-70,+90)})
hname = "window_precuts";  histos.update({hname:ROOT.TH2D(hname,"Vacuum window plane (pre-cuts);x [mm];y [mm];Fake Tracks",120,-70,+70, 120,50,+190)})
hname = "window_postcuts"; histos.update({hname:ROOT.TH2D(hname,"Vacuum window plane (pst-cuts);x [mm];y [mm];Fake Tracks",120,-70,+70, 120,50,+190)})
for det in cfg["detectors"]:
    # hname = f"{det}_precuts";  histos.update({hname:ROOT.TH2D(hname,f"{det} (pre-cuts);x [mm];y [mm];Fake Tracks",100,Lxmin,Lxmax, 200,Lymin,Lymax)})
    # hname = f"{det}_postcuts"; histos.update({hname:ROOT.TH2D(hname,f"{det} (post-cuts);x [mm];y [mm];Fake Tracks",100,Lxmin,Lxmax, 200,Lymin,Lymax)})
    hname = f"{det}_precuts";  histos.update({hname:ROOT.TH2D(hname,f"{det} (pre-cuts);x [mm];y [mm];Fake Tracks",500,-25,+25, 1000,30,+130)})
    hname = f"{det}_postcuts"; histos.update({hname:ROOT.TH2D(hname,f"{det} (post-cuts);x [mm];y [mm];Fake Tracks",500,-25,+25, 1000,30,+130)})

hname = "slopexz_precuts";  histos.update({hname:ROOT.TH1D(hname,"XZ Slope (pre-cuts);a_{xz};Fake Tracks",100,slopes["xz"][0]*1.2,slopes["xz"][1]*1.2)})
hname = "slopexz_postcuts"; histos.update({hname:ROOT.TH1D(hname,"XZ Slope (post-cuts);a_{xz};Fake Tracks",100,slopes["xz"][0]*1.2,slopes["xz"][1]*1.2)})
hname = "slopeyz_precuts";  histos.update({hname:ROOT.TH1D(hname,"YZ Slope (pre-cuts);a_{yz};Fake Tracks",100,slopes["yz"][0]*0.8,slopes["yz"][1]*1.2)})
hname = "slopeyz_postcuts"; histos.update({hname:ROOT.TH1D(hname,"YZ Slope (post-cuts);a_{yz};Fake Tracks",100,slopes["yz"][0]*0.8,slopes["yz"][1]*1.2)})
    

    

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

layer = ROOT.TPolyLine()
xMinL = gun.layers[0][0][0]
yMinL = gun.layers[0][0][1]
xMaxL = gun.layers[0][2][0]
yMaxL = gun.layers[0][2][1]
layer.SetNextPoint(xMinL,yMinL)
layer.SetNextPoint(xMinL,yMaxL)
layer.SetNextPoint(xMaxL,yMaxL)
layer.SetNextPoint(xMaxL,yMinL)
layer.SetNextPoint(xMinL,yMinL)    
layer.SetLineColor(ROOT.kBlue)
layer.SetLineWidth(1)

Ngen = 0
Nacc = 0

### fill data tree
for Event in range(Nevents):
    ##############################
    ### generate particles in acceptance per event
    fakeprts = []
    while(len(fakeprts)<Nprtperevt):
        fakeprt = gun.generate()
        Ngen += 1

        histos["dipole_precuts"].Fill(fakeprt.vtx[0],fakeprt.vtx[1])
        histos["window_precuts"].Fill(gun.k_of_z(cfg["zWindow"],fakeprt.slp[0],fakeprt.itp[0]), gun.k_of_z(cfg["zWindow"],fakeprt.slp[1],fakeprt.itp[1]))
        for det in cfg["detectors"]: histos[f"{det}_precuts"].Fill(fakeprt.smrpts[det][0],fakeprt.smrpts[det][1])
        histos["slopexz_precuts"].Fill(fakeprt.slp[0])
        histos["slopeyz_precuts"].Fill(fakeprt.slp[1])
        
        if(not gun.accept(fakeprt)): continue
        Nacc += 1
        
        histos["dipole_postcuts"].Fill(fakeprt.vtx[0],fakeprt.vtx[1])
        histos["window_postcuts"].Fill(gun.k_of_z(cfg["zWindow"],fakeprt.slp[0],fakeprt.itp[0]), gun.k_of_z(cfg["zWindow"],fakeprt.slp[1],fakeprt.itp[1]))
        for det in cfg["detectors"]: histos[f"{det}_postcuts"].Fill(fakeprt.smrpts[det][0],fakeprt.smrpts[det][1])
        histos["slopexz_postcuts"].Fill(fakeprt.slp[0])
        histos["slopeyz_postcuts"].Fill(fakeprt.slp[1])
        
        fakeprts.append(fakeprt)

    hits = {}
    for det in cfg["detectors"]:
        hits.update({det:[]})
    for prt in fakeprts:
        for det in cfg["detectors"]:
            hits[det].append(prt.smrpts[det])
    
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
    for det in cfg["detectors"]:
        chipid = cfg["det2plane"][det]
        event.st_ev_buffer[0].ch_ev_buffer.push_back( ROOT.chip() )
        ichip = event.st_ev_buffer[0].ch_ev_buffer.size()-1
        event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id = int(chipid)
        
        for hit in hits[det]:
            ### in lab space
            x = hit[0]
            y = hit[1]
            z = hit[2]
            ### in chip space
            r = transform_to_chip_space([x,y,z])
            ix = hPixelMatrix.GetXaxis().FindBin(r[0])-1 ### TODO: maybe no need for the -1?
            iy = hPixelMatrix.GetYaxis().FindBin(r[1])-1 ### TODO: maybe no need for the -1?
            event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.push_back( ROOT.pixel() )
            ihit = event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.size()-1
            event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].ix = ix
            event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].iy = iy
            event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].xFake = r[0]
            event.st_ev_buffer[0].ch_ev_buffer[ichip].hits[ihit].yFake = r[1]
        
            # print(f"{det}: xLab={x:.3f}, yLab={y:.3f}, zLab={z:.1f} --> xChp={r[0]:.3f}, yChp={r[1]:.3f}, zChp={r[2]:.1f} --> ix={ix}, iy={iy}")

    ################################
    ### fill every event
    tOut.Fill()

print(f"Acceptance: {(Nacc/Ngen)*100:.1f}%")


cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
cnv.Divide(2,1)
cnv.cd(1)
ROOT.gPad.SetTicks(1,1)
histos["dipole_precuts"].Draw("colz")
dipole.Draw()
ROOT.gPad.RedrawAxis()
cnv.cd(2)
ROOT.gPad.SetTicks(1,1)
histos["dipole_postcuts"].Draw("colz")
dipole.Draw()
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs("particlegun_to_eudaq_write.pdf(")

cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
cnv.Divide(2,1)
cnv.cd(1)
ROOT.gPad.SetTicks(1,1)
histos["window_precuts"].Draw("colz")
window.Draw()
ROOT.gPad.RedrawAxis()
cnv.cd(2)
ROOT.gPad.SetTicks(1,1)
histos["window_postcuts"].Draw("colz")
window.Draw()
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs("particlegun_to_eudaq_write.pdf")

for det in cfg["detectors"]:
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos[f"{det}_precuts"].Draw("colz")
    layer.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos[f"{det}_postcuts"].Draw("colz")
    layer.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("particlegun_to_eudaq_write.pdf")

cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
cnv.Divide(2,1)
cnv.cd(1)
ROOT.gPad.SetTicks(1,1)
histos["slopexz_precuts"].Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.cd(2)
ROOT.gPad.SetTicks(1,1)
histos["slopexz_postcuts"].Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs("particlegun_to_eudaq_write.pdf")

cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
cnv.Divide(2,1)
cnv.cd(1)
ROOT.gPad.SetTicks(1,1)
histos["slopeyz_precuts"].Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.cd(2)
ROOT.gPad.SetTicks(1,1)
histos["slopeyz_postcuts"].Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.Update()
cnv.SaveAs("particlegun_to_eudaq_write.pdf)")


### finish
fOut.cd()
tOut.Write()
tOutMeta.Write()
fOut.Write()
fOut.Close()

