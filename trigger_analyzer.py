import ROOT
import math
import array
import numpy as np

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

import argparse
parser = argparse.ArgumentParser(description='analyze_triggers.py...')
parser.add_argument('-file', metavar='input file', required=True,  help='full path to input file')
parser.add_argument('-imin', metavar='first entry', required=False,  help='first entry')
parser.add_argument('-imax', metavar='last entry', required=False,  help='last entry')
argus = parser.parse_args()
infile = argus.file
tfile = ROOT.TFile(infile,"READ")
ttree = tfile.Get("MyTree")
nentries = ttree.GetEntries()
print(f"Entries in tree: {nentries}")
imin = int(argus.imin) if(argus.imin is not None) else 0
imax = int(argus.imax) if(argus.imax is not None) else nentries
nentries = imax-imin
print(f"Reading from entry {imin} to {imax} --> corrected entries: {nentries}")


detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
detectorids = [8,6,4,2,0]
detcol = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kOrange+1]

x_trg = np.zeros(nentries)
x_ent = np.zeros(nentries)
x_tim = []
hits_vs_trg = {}
hits_vs_ent = {}
for det in detectors:
    hits_vs_trg.update({ det:np.zeros(nentries) })
    hits_vs_ent.update({ det:np.zeros(nentries) })
y_yag = np.zeros(nentries)


y_q0act    = np.zeros(nentries)
y_q1act    = np.zeros(nentries)
y_q2act    = np.zeros(nentries)
y_m12      = np.zeros(nentries)
y_m34      = np.zeros(nentries)
y_rad      = np.zeros(nentries)
y_toro3163 = np.zeros(nentries)
y_toro3255 = np.zeros(nentries)
y_foilm1   = np.zeros(nentries)
y_foilm2   = np.zeros(nentries)

ranges = []
rng = []
counter = 0
for ientry,entry in enumerate(ttree):
    if(ientry<imin): continue
    if(ientry>=imax): break

    trgn   = entry.event.trg_n
    ts_bgn = entry.event.ts_begin
    ts_end = entry.event.ts_end
    
    x_trg[counter] = trgn
    x_ent[counter] = ientry
    x_tim.append( get_human_timestamp_ns(ts_bgn) )
    
    y_q0act[counter]    = entry.event.epics_frame.espec_quad0_bact
    y_q1act[counter]    = entry.event.epics_frame.espec_quad1_bact
    y_q2act[counter]    = entry.event.epics_frame.espec_quad2_bact
    y_m12[counter]      = entry.event.epics_frame.mcalc_m12
    y_m34[counter]      = entry.event.epics_frame.mcalc_m34
    y_rad[counter]      = entry.event.epics_frame.radm_li20_1_ch01_meas
    y_toro3163[counter] = entry.event.epics_frame.toro_li20_3163_tmit_pc
    y_toro3255[counter] = entry.event.epics_frame.toro_li20_3255_tmit_pc
    y_foilm1[counter]   = entry.event.epics_frame.xps_li20_mc05_m1_rbv
    y_foilm2[counter]   = entry.event.epics_frame.xps_li20_mc05_m2_rbv
    y_yag[counter]      = entry.event.epics_frame.yag_hm_rbv
    
    allhits = 0
    for ichip in range(entry.event.st_ev_buffer[0].ch_ev_buffer.size()):
        detid = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id
        detix = detectorids.index(detid)
        det   = detectors[detix]
        nhits = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.size()
        hits_vs_trg[det][counter] = nhits
        hits_vs_ent[det][counter] = nhits
        allhits += nhits

    if(trgn==12750):
        print(f"trgn={trgn}, ientry={ientry}, counter={counter}, allhits={allhits}")

    ### important!!
    counter += 1

print(f"beginning:  {x_tim[0]}")
print(f"end of run: {x_tim[-1]}")


graphs = {}
maxhits = -1
for det in detectors:
    maxhitsdet = max(hits_vs_trg[det])
    if(maxhitsdet>maxhits): maxhits = maxhitsdet

    gname = f"hits_vs_trg_{det}"
    graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,hits_vs_trg[det])} )
    graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
    print(f"{gname}: avg={np.mean(hits_vs_trg[det])}, std={np.std(hits_vs_trg[det])}")

    gname = f"hits_vs_ent_{det}"
    graphs.update( {gname:ROOT.TGraph(len(x_ent),x_ent,hits_vs_ent[det])} )
    graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
    print(f"{gname}: avg={np.mean(hits_vs_trg[det])}, std={np.std(hits_vs_trg[det])}")



gname = "q0act"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_q0act)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kBlack)
gname = "q1act"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_q1act)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kRed)
gname = "q2act"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_q2act)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kGreen+1)
gname = "m12"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_m12)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kBlack)
gname = "m34"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_m34)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kRed)
gname = "rad"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_rad)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kBlack)
gname = "foilm1"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_foilm1)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kBlack)
gname = "foilm2"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_foilm2)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kRed)
gname = "foilm2"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_foilm2)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kRed)
gname = "toro3163"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_toro3163)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kBlack)
gname = "toro3255"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_toro3255)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kRed)
gname = "yag"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_yag)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)
graphs[gname].SetLineColor(ROOT.kBlack)


for i,det in enumerate(detectors):
    gname = f"hits_vs_ent_{det}"
    graphs[gname].SetTitle(f"{det}: fired pixels per tree entry;Tree entry;Fired pixels")
    graphs[gname].SetMaximum(maxhits*5)
    graphs[gname].SetMinimum(0.9)
    graphs[gname].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    graphs[gname].SetLineColor(detcol[i])
for i,det in enumerate(detectors):
    gname = f"hits_vs_trg_{det}"
    graphs[gname].SetTitle(f"{det}: fired pixels per trigger;Trigger number;Fired pixels")
    graphs[gname].SetMaximum(maxhits*5)
    graphs[gname].SetMinimum(0.9)
    graphs[gname].GetXaxis().SetLimits(x_ent[0],x_ent[-1])
    graphs[gname].SetLineColor(detcol[i])


cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
cnv.SetTicks(1,1)
cnv.SetLogy()
leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
mg = ROOT.TMultiGraph()
for i,det in enumerate(detectors):
    gname = f"hits_vs_trg_{det}"
    leg.AddEntry(graphs[gname],f"{det}","l")
    mg.Add(graphs[gname])
mg.Draw("al")
leg.Draw("same")
mg.SetTitle(f";Trigger number;Fired pixels")
mg.SetMaximum(maxhits*5)
mg.SetMinimum(0.9)
mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf(")

cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
cnv.SetTicks(1,1)
# cnv.SetLogy()
leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
mg = ROOT.TMultiGraph()
for q in range(3):
    gname = f"q{q}act"
    leg.AddEntry(graphs[gname],f"Quad{q}","l")
    mg.Add(graphs[gname])
mg.Draw("al")
leg.Draw("same")
mg.SetTitle(f";Trigger number;Quad [kG/m]")
mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf")

cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
cnv.SetTicks(1,1)
# cnv.SetLogy()
leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
mg = ROOT.TMultiGraph()
for mij in ["12","34"]:
    gname = f"m{mij}"
    leg.AddEntry(graphs[gname],f"M{mij}","l")
    mg.Add(graphs[gname])
mg.Draw("al")
leg.Draw("same")
mg.SetTitle(";Trigger number;M_{ij} [??]")
mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf")

cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
cnv.SetTicks(1,1)
# cnv.SetLogy()
leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
mg = ROOT.TMultiGraph()
for t in ["3163","3255"]:
    gname = f"toro{t}"
    leg.AddEntry(graphs[gname],f"Toroid {t}","l")
    mg.Add(graphs[gname])
mg.Draw("al")
leg.Draw("same")
mg.SetTitle(";Trigger number;Toroid charge [pC]")
mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf")


# cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
# cnv.SetTicks(1,1)
# # cnv.SetLogy()
# leg = ROOT.TLegend(0.4,0.2,0.7,0.5)
# leg.SetFillStyle(4000) # will be transparent
# leg.SetFillColor(0)
# leg.SetTextFont(42)
# leg.SetBorderSize(0)
# mg = ROOT.TMultiGraph()
# for m in ["m1","m2"]:
#     gname = f"foil{m}"
#     leg.AddEntry(graphs[gname],f"Foil {m}","l")
#     mg.Add(graphs[gname])
# mg.Draw("al")
# leg.Draw("same")
# mg.SetTitle(";Trigger number;Foil position [mm]")
# mg.GetXaxis().SetLimits(x_trg[0],x_trg[-1])
# cnv.RedrawAxis()
# cnv.Update()
# cnv.SaveAs("hits_vs_triggers.pdf")

cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
cnv.SetTicks(1,1)
# cnv.SetLogy()
graphs["rad"].Draw("al")
graphs["rad"].SetTitle(";Trigger number;RadMon [mRem/??]")
graphs["rad"].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf")

cnv = ROOT.TCanvas("cnv_hits_vs_trg_all","",1200,500)
cnv.SetTicks(1,1)
# cnv.SetLogy()
graphs["foilm2"].Draw("al")
graphs["foilm2"].SetTitle(";Trigger number;Foil x [mm]")
graphs["foilm2"].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf)")

