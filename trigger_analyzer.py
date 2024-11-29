import ROOT
import math
import array
import numpy as np

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


# # tfile = ROOT.TFile("test_data/e320_prototype_beam_2024/runs/tree_11_05_2024_Run446.root","READ")
# # tfile = ROOT.TFile("test_data/e320_prototype_beam_2024/runs/tree_11_03_2024_Run405.root","READ")
# # tfile = ROOT.TFile("test_data/e320_prototype_beam_2024/runs/tree_11_04_2024_Run409.root","READ")
# tfile = ROOT.TFile("../../Downloads/tree_11_04_2024_06_50_48_Run417_1.root","READ")


detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3"]
detectorids = [8,5,3,1]
detcol = [ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2]

x_trg = np.zeros(nentries)
x_ent = np.zeros(nentries)
hits_vs_trg = {}
hits_vs_ent = {}
for det in detectors:
    hits_vs_trg.update({ det:np.zeros(nentries) })
    hits_vs_ent.update({ det:np.zeros(nentries) })
y_yag = np.zeros(nentries)


ranges = []
rng = []
for ientry,entry in enumerate(ttree):
    if(ientry<imin): continue
    if(ientry>=imax): break
    
    trgn = entry.event.trg_n
    x_trg[ientry] = trgn
    x_ent[ientry] = ientry
    
    y_yag[ientry] = entry.event.epics_frame.yag_hm_rbv
    
    allhits = 0
    for ichip in range(entry.event.st_ev_buffer[0].ch_ev_buffer.size()):
        detid = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].chip_id
        detix = detectorids.index(detid)
        det   = detectors[detix]
        nhits = entry.event.st_ev_buffer[0].ch_ev_buffer[ichip].hits.size()
        hits_vs_trg[det][ientry] = nhits
        hits_vs_ent[det][ientry] = nhits
        allhits += nhits

    if(trgn==12750):
        print(f"trgn={trgn}, ientry={ientry}, allhits={allhits}")

#     if(nhits<10000000):
#         if(len(rng)==0): rng = [i]
#         else:            continue
#     else:
#         if(len(rng)>0):
#             rng.append(i-1)
#             nrng = rng[1]-rng[0]
#             if(nrng>0):
#                 x = {(rng[0],rng[1]):nrng}
#                 ranges.append(x)
#             rng = []
#
# print(ranges)


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

gname = "yag"
graphs.update( {gname:ROOT.TGraph(len(x_trg),x_trg,y_yag)} )
graphs[gname].SetBit(ROOT.TGraph.kIsSortedX)

    

cnv = ROOT.TCanvas("cnv_hits_vs_ent","",1200,1000)
cnv.Divide(2,2)
for i,det in enumerate(detectors):
    p = cnv.cd(i+1)
    p.SetTicks(1,1)
    p.SetLogy()
    gname = f"hits_vs_ent_{det}"
    graphs[gname].SetTitle(f"{det}: fired pixels per tree entry;Tree entry;Fired pixels")
    graphs[gname].SetMaximum(maxhits*5)
    graphs[gname].SetMinimum(0.9)
    graphs[gname].GetXaxis().SetLimits(x_trg[0],x_trg[-1])
    graphs[gname].SetLineColor(detcol[i])
    graphs[gname].Draw("AC")
    p.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf(")

cnv = ROOT.TCanvas("cnv_hits_vs_trg","",1200,1000)
cnv.Divide(2,2)
for i,det in enumerate(detectors):
    p = cnv.cd(i+1)
    p.SetTicks(1,1)
    p.SetLogy()
    gname = f"hits_vs_trg_{det}"
    graphs[gname].SetTitle(f"{det}: fired pixels per trigger;Trigger number;Fired pixels")
    graphs[gname].SetMaximum(maxhits*5)
    graphs[gname].SetMinimum(0.9)
    graphs[gname].GetXaxis().SetLimits(x_ent[0],x_ent[-1])
    graphs[gname].SetLineColor(detcol[i])
    graphs[gname].Draw("AC")
    p.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf")

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
cnv.SaveAs("hits_vs_triggers.pdf")

cnv = ROOT.TCanvas("cnv_yag_vs_trg","",1200,500)
cnv.SetTicks(1,1)
gname = "yag"
graphs[gname].SetTitle(f";Trigger number;YAG position")
# graphs[gname].SetMaximum()
# graphs[gname].SetMinimum(0)
graphs[gname].GetXaxis().SetLimits(x_ent[0],x_ent[-1])
graphs[gname].SetLineColor(ROOT.kBlack)
graphs[gname].Draw("AC")
cnv.RedrawAxis()
cnv.Update()
cnv.SaveAs("hits_vs_triggers.pdf)")