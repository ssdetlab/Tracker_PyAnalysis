#!/usr/bin/python
import os
import os.path
import math
import time
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
# from skspatial.objects import Line, Sphere
# from skspatial.plotting import plot_3d

import argparse
parser = argparse.ArgumentParser(description='postptrocess_plots.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
argus = parser.parse_args()
configfile = argus.conf

import config
from config import *
### must be called here (first) and only once!
init_config(configfile,True)

import utils
from utils import *
import svd_fit
from svd_fit import *
import chi2_fit
from chi2_fit import *
import hists
from hists import *

import objects
from objects import *
import pixels
from pixels import *
import clusters
from clusters import *
import truth
from truth import *
import noise
from noise import *
import candidate
from candidate import *



ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)

mm2um = 1000

tfilenamein = make_run_dirs(cfg["inputfile"])
tfilenamein = tfilenamein.replace(".root","_multiprocess_histograms.root")
# tfilenamein = cfg["inputfile"].replace(".root","_multiprocess_histograms.root")
detectors = cfg["detectors"]

histprefx = ["h_cls_size", "h_Chi2fit_res_trk2cls_x", "h_Chi2fit_res_trk2cls_y", ]
histos = {}

def gethmax(h,norm=True):
    hmax = 0
    hint = h.Integral()
    for b in range(h.GetNbinsX()+1):
        y = h.GetBinContent(b) if(norm==False) else h.GetBinContent(b)/hint
        hmax = y if(y>hmax) else hmax
    return hmax

def book_histos(tfo):
    tfo.cd()
    for prefx in histprefx:
        for idet,det in enumerate(detectors):
            hname = prefx+"_"+det
            hist = det+"/"+hname
            name = hname
            tfi = ROOT.TFile( tfilenamein,"READ" )
            print("From file:",tfilenamein,"getting histogram named:",hist)
            histos.update({name:tfi.Get(hist).Clone(name)})
            if(det in histos[name].GetTitle()): histos[name].SetTitle( det )
            histos[name].SetDirectory(0)


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


def plot_2x2_histos(pdf,prefix):
    for idet,det in enumerate(detectors):
        hname = prefix+"_"+det
        histos[hname].SetLineColor(ROOT.kBlack)
        histos[hname].SetMarkerColor(ROOT.kBlack)
        histos[hname].SetMarkerSize(1)
        histos[hname].SetMarkerStyle(20)
        histos[hname].Scale(1./histos[hname].Integral())
        histos[hname].SetTitle(det)
        histos[hname].GetYaxis().SetTitle("Normalized")
    
    cnv = ROOT.TCanvas("cnv","",1200,1000)
    cnv.Divide(2,2)
    for count1,det in enumerate(detectors):
        p = cnv.cd(count1+1)
        p.SetTicks(1,1)
        # if("cls_size" in prefix): p.SetLogy()
        
        hname = prefix+"_"+det
        histos[hname].Draw("e1p")
        ### fit
        if("h_Chi2fit_res_trk2cls" in prefix):
            func = fit1(histos[hname],ROOT.kRed,-0.012,+0.012)
            s = ROOT.TLatex()
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextFont(22)
            s.SetTextSize(0.045)
            s.DrawLatex(0.17,0.85,ROOT.Form("Mean: %.2f #mum" % (mm2um*func.GetParameter(1))))
            s.DrawLatex(0.17,0.78,ROOT.Form("Sigma: %.2f #mum" % (mm2um*func.GetParameter(2))))
            s.DrawLatex(0.2,0.71,ROOT.Form("#chi^{2}/N_{DOF}: %.2f" % (func.GetChisquare()/func.GetNDF())))
    cnv.SaveAs(pdf)



#####################################################################################
#####################################################################################
#####################################################################################
if __name__ == "__main__":
    # get the start time
    tfilenameout = tfilenamein.replace(".root","_postprocessplots.root")
    tfo = ROOT.TFile(tfilenameout,"RECREATE")
    book_histos(tfo)
    
    # scale6 = gethmax(histos[run2fit+"_h_cls_size_ALPIDE_0"])
    # scale0 = gethmax(histos[run2fit+"_h_cls_size_ALPIDE_3"]) if(runtype=="cosmics") else gethmax(histos[run2fit+"_h_cls_size_ALPIDE_0"])
    # print("scale0=",scale0,"  scale6=",scale6)
    
    plot_2x2_histos(tfilenameout.replace("root","pdf("),"h_cls_size")
    plot_2x2_histos(tfilenameout.replace("root","pdf"),"h_Chi2fit_res_trk2cls_x")
    plot_2x2_histos(tfilenameout.replace("root","pdf)"),"h_Chi2fit_res_trk2cls_y")
    
    write_histos(tfo)
    tfo.Write()
    tfo.Close()