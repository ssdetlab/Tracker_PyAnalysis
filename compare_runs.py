#!/usr/bin/python
import os
import os.path
import math
import time
import subprocess
import array
import numpy as np
import ROOT
# from ROOT import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
from skspatial.objects import Line, Sphere
from skspatial.plotting import plot_3d

import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-run', metavar='run type', required=True,  help='run type [source/cosmics]')
argus = parser.parse_args()
runtype = argus.run

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)

mm2um = 1000

cols = [ROOT.kBlack, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange, ROOT.kBlue]

mrks = [20,          24,        32,         25,            23]

files_cosmics = {
    "run74x":"~/Downloads/data_telescope/eudaq/Jun05/vbb6_dv10_vresetd200_clip70_run74x/tree_vbb6_dv10_vresetd200_clip70_run74x_multiprocess_histograms.root", ##delay=4.0us, strobe=100ns
    "run75x":"~/Downloads/data_telescope/eudaq/Jun12/vbb6_dv10_vresetd200_clip70_run75x/tree_vbb6_dv10_vresetd200_clip70_run75x_multiprocess_histograms.root", ##delay=4.7us, strobe=100ns
    "run75y":"~/Downloads/data_telescope/eudaq/Jun17/vbb6_dv10_vresetd200_clip70_run75y/tree_vbb6_dv10_vresetd200_clip70_run75y_multiprocess_histograms.root", ##delay=165ns, strobe=12us
    "run76x":"~/Downloads/data_telescope/eudaq/Jun27/vbb6_dv10_vresetd200_clip70_run76x/tree_vbb6_dv10_vresetd200_clip70_run76x_multiprocess_histograms.root", ##delay=1.5us, strobe=10us
    # "run760":"~/Downloads/data_telescope/eudaq/Jun22/vbb6_dv10_vresetd200_clip70_run760/tree_vbb6_dv10_vresetd200_clip70_run760_multiprocess_histograms.root", ##delay=165ns, strobe=100ns
    # "run759":"~/Downloads/data_telescope/eudaq/Jun18/vbb6_dv10_vresetd200_clip70_run759/tree_vbb6_dv10_vresetd200_clip70_run759_multiprocess_histograms.root", ##delay=165ns, strobe=100us
}
files_source = {
    "run77x":"~/Downloads/data_telescope/eudaq/Jul08/vbb0_dv10_vresetd147_clip60_run77x/tree_vbb0_dv10_vresetd147_clip60_run77x_multiprocess_histograms.root",
    "run77y":"~/Downloads/data_telescope/eudaq/Jul12/vbb0_dv10_vresetd147_clip60_run77y/tree_vbb0_dv10_vresetd147_clip60_run77y_multiprocess_histograms.root",
}
files = files_cosmics if(runtype=="cosmics") else files_source

dely = {
    "run74x":"4.0 #mus",
    "run75x":"4.7 #mus",
    "run75y":"165 ns",
    "run76x":"1.5 #mus",
    "run760":"165 ns",
    "run77x":"150 ns",
    "run77y":"150 ns",
}
strb = {
    "run74x":"100 ns",
    "run75x":"100 ns",
    "run75y":"12 #mus",
    "run76x":"10 #mus",
    "run760":"100 ns",
    "run77x":"10 #mus",
    "run77y":"10 #mus",
}

detectors = ["ALPIDE_0", "ALPIDE_1", "ALPIDE_2", "ALPIDE_3"] if(runtype=="cosmics") else ["ALPIDE_0", "ALPIDE_1", "ALPIDE_2"]
voltages  = ["V_{bb}=-6V", "V_{bb}=-6V", "V_{bb}=-6V", "V_{bb}=0V"] if(runtype=="cosmics") else ["V_{bb}=0V", "V_{bb}=0V", "V_{bb}=0V"]

histprefx = ["h_cls_size", "h_Chi2fit_res_trk2cls_x", "h_Chi2fit_res_trk2cls_y", ]
histos = {}
runs = []
runscol = {}
runsmrk = {}
for run,fname in files.items():
    runs.append(run)
    runscol.update({run:cols[runs.index(run)]})
    runsmrk.update({run:mrks[runs.index(run)]})
run2fit = "run75y"


def gethmax(h,norm=True):
    hmax = 0
    hint = h.Integral()
    for b in range(h.GetNbinsX()+1):
        y = h.GetBinContent(b) if(norm==False) else h.GetBinContent(b)/hint
        hmax = y if(y>hmax) else hmax
    return hmax


def book_histos(tfo):
    tfo.cd()
    for run,fname in files.items():
        for prefx in histprefx:
            for idet,det in enumerate(detectors):
                hname = prefx+"_"+det
                hist = det+"/"+hname
                name = run+"_"+hname
                tfi = ROOT.TFile(fname,"READ")
                print("From file:",fname,"getting histogram named:",hist)
                histos.update({name:tfi.Get(hist).Clone(name)})
                if(det in histos[name].GetTitle()): histos[name].SetTitle( det+", "+voltages[idet] )
                histos[name].SetDirectory(0)


def alice_histos(scale0,scale6):
    print("setting alice histos")
    h_V0_20deg = ROOT.TH1D("h_cls_size_alice_V0",";Cluster size;Events",10,0.5,10.5)
    h_V3_20deg = ROOT.TH1D("h_cls_size_alice_V3",";Cluster size;Events",10,0.5,10.5)
    
    h_V0_20deg.SetFillColorAlpha(ROOT.kBlue,0.04)
    h_V3_20deg.SetFillColorAlpha(ROOT.kBlue,0.04)

    h_V0_20deg.SetLineColorAlpha(ROOT.kBlue,0.04)
    h_V3_20deg.SetLineColorAlpha(ROOT.kBlue,0.04)
    
    h_V0_20deg.SetLineWidth(1)
    h_V3_20deg.SetLineWidth(1)
    
    h_V0_20deg.Fill(1, 0.113)
    h_V0_20deg.Fill(2, 0.122)
    h_V0_20deg.Fill(3, 0.0245)
    h_V0_20deg.Fill(4, 0.0111)
    h_V0_20deg.Fill(5, 9.9E-07)
    scale = scale0/gethmax(h_V0_20deg,False)
    print("h_V0_20deg scale=",scale)
    h_V0_20deg.Scale(scale)
    
    h_V3_20deg.Fill(1, 0.118)
    h_V3_20deg.Fill(2, 0.118)
    h_V3_20deg.Fill(3, 0.0274)
    h_V3_20deg.Fill(4, 0.0215)
    h_V3_20deg.Fill(5, 0.00000914)
    h_V3_20deg.Fill(6, 9.91E-07)
    scale = scale6/gethmax(h_V3_20deg,False)
    print("h_V3_20deg scale=",scale)
    h_V3_20deg.Scale(scale)
    
    return h_V0_20deg,h_V3_20deg


def write_histos(tfo):
    tfo.cd()
    for hname,hist in histos.items():
        hist.Write()

def fit1(h,col,xmin,xmax):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    # f1 = TF1("f1", "gaus(0)", xmin,xmax)
    g1.SetLineColor(col)
    # f1.SetLineColor(col)
    h.Fit(g1,"EMRS")
    # f1.SetParameter(0,g1.GetParameter(0))
    # f1.SetParameter(1,g1.GetParameter(1))
    # f1.SetParameter(2,g1.GetParameter(2))
    # chi2dof = f1.GetChisquare()/f1.GetNDF() if(f1.GetNDF()>0) else -1
    chi2dof = g1.GetChisquare()/g1.GetNDF() if(g1.GetNDF()>0) else -1
    print("g1 chi2/Ndof=",chi2dof)
    return g1

def fit2(h,col):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g2 = ROOT.TF1("g2", "gaus", xmin,xmax)
    f1 = ROOT.TF1("f2", "gaus(0)+gaus(3)", xmin,xmax)
    g1.SetLineColor(col)
    g2.SetLineColor(col)
    f2.SetLineColor(col)
    h.Fit(g1,"EMRS")
    h.Fit(g2,"EMRS")
    f2.SetParameter(0,g1.GetParameter(0))
    f2.SetParameter(1,g1.GetParameter(1))
    f2.SetParameter(2,g1.GetParameter(2))
    f2.SetParameter(3,g2.GetParameter(0))
    f2.SetParameter(4,g2.GetParameter(1))
    f2.SetParameter(5,g2.GetParameter(2))
    chi2dof = f2.GetChisquare()/f2.GetNDF() if(f2.GetNDF()>0) else -1
    print("f2 chi2/Ndof=",chi2dof)
    return f2

def fit3(h,col):
    g1 = ROOT.TF1("g1", "gaus", xmin,xmax)
    g2 = ROOT.TF1("g2", "gaus", xmin,xmax)
    g3 = ROOT.TF1("g3", "gaus", xmin,xmax)
    f3 = ROOT.TF1("f3", "gaus(0)+gaus(3)+gaus(6)", xmin,xmax)
    g1.SetLineColor(col)
    g2.SetLineColor(col)
    g3.SetLineColor(col)
    f3.SetLineColor(col)
    h.Fit(g1,"EMRS")
    h.Fit(g2,"EMRS")
    h.Fit(g3,"EMRS")
    f3.SetParameter(0,g1.GetParameter(0))
    f3.SetParameter(1,g1.GetParameter(1))
    f3.SetParameter(2,g1.GetParameter(2))
    f3.SetParameter(3,g2.GetParameter(0))
    f3.SetParameter(4,g2.GetParameter(1))
    f3.SetParameter(5,g2.GetParameter(2))
    f3.SetParameter(6,g3.GetParameter(0))
    f3.SetParameter(7,g3.GetParameter(1))
    f3.SetParameter(8,g3.GetParameter(2))
    chi2dof = f3.GetChisquare()/f3.GetNDF() if(f3.GetNDF()>0) else -1
    print("f3 chi2/Ndof=",chi2dof)
    return f3


def plot_2x2_histos(pdf,prefix,alice=None):
    ymax = -1e10
    for run in runs:
        for idet,det in enumerate(detectors):
            hname = prefix+"_"+det
            histos[run+"_"+hname].SetLineColor(runscol[run])
            histos[run+"_"+hname].SetMarkerColor(runscol[run])
            histos[run+"_"+hname].SetMarkerSize(1)
            histos[run+"_"+hname].SetMarkerStyle(runsmrk[run])
            histos[run+"_"+hname].Scale(1./histos[run+"_"+hname].Integral())
            histos[run+"_"+hname].SetTitle(det+", "+voltages[idet])
            histos[run+"_"+hname].GetYaxis().SetTitle("Normalized")
            tmax = histos[run+"_"+hname].GetMaximum()
            ymax = tmax if(tmax>ymax) else ymax

    # factor = 2 if("cls_size" in prefix) else 1.2
    factor = 1.2
    for run in runs:
        for det in detectors:
            hname = prefix+"_"+det
            histos[run+"_"+hname].SetMaximum(ymax*factor)
    
    leg = ROOT.TLegend(0.50,0.63,0.89,0.87)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    for run in runs:
        label = run+": Trg="+dely[run]+", Stb="+strb[run]
        leg.AddEntry(histos[run+"_"+prefix+"_ALPIDE_0"],label,"lp")
    
    cnv = ROOT.TCanvas("cnv","",1200,1000)
    cnv.Divide(2,2)
    for count1,det in enumerate(detectors):
        p = cnv.cd(count1+1)
        p.SetTicks(1,1)
        # if("cls_size" in prefix): p.SetLogy()
        
        hname = prefix+"_"+det
        for count2,run in enumerate(runs):
            if(count2==0): histos[run+"_"+hname].Draw("e1p")
            else:          histos[run+"_"+hname].Draw("e1p same")
            ### overlay ALICE digitized plots
            if("cls_size" in prefix and alice!=None):
                if(det=="ALPIDE_3"):
                    # alice["h_V0_20deg"].SetFillColorAlpha(ROOT.kBlue,0.1)
                    alice["h_V0_20deg"].Draw("hist same")
                else:
                    # alice["h_V3_20deg"].SetFillColorAlpha(ROOT.kBlue,0.1)
                    alice["h_V3_20deg"].Draw("hist same")
            ### fit
            if(run==run2fit and "h_Chi2fit_res_trk2cls" in prefix):
                func = fit1(histos[run+"_"+hname],runscol[run],-0.01,+0.01)
                s = ROOT.TLatex()
                s.SetNDC(1);
                s.SetTextAlign(13);
                s.SetTextColor(ROOT.kBlack)
                s.SetTextFont(22)
                s.SetTextSize(0.045)
                s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
                s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
                s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
                    
                
            
        leg.Draw("same")
    cnv.SaveAs(pdf)



#####################################################################################
#####################################################################################
#####################################################################################

tfilenameout = "compare.root"
tfo = ROOT.File(tfilenameout,"RECREATE")
book_histos(tfo)

scale6 = gethmax(histos[run2fit+"_h_cls_size_ALPIDE_0"])
scale0 = gethmax(histos[run2fit+"_h_cls_size_ALPIDE_3"]) if(runtype=="cosmics") else gethmax(histos[run2fit+"_h_cls_size_ALPIDE_0"])
print("scale0=",scale0,"  scale6=",scale6)

h_V0_20deg,h_V3_20deg = alice_histos(scale0,scale6)
alice = {"h_V0_20deg":h_V0_20deg, "h_V3_20deg":h_V3_20deg}
plot_2x2_histos(tfilenameout.replace("root","pdf("),"h_cls_size",alice)
plot_2x2_histos(tfilenameout.replace("root","pdf"),"h_Chi2fit_res_trk2cls_x")
plot_2x2_histos(tfilenameout.replace("root","pdf)"),"h_Chi2fit_res_trk2cls_y")

write_histos(tfo)
tfo.Write()
tfo.Close()