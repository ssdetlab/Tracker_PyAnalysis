import ROOT
import os
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


def get_h1(h):
    npix = h.GetNbinsX()*h.GetNbinsY()
    h1 = ROOT.TH1D(f"{h.GetName()}_1D","",npix,0,npix)
    for bx in range(1,h.GetNbinsX()+1):
        for by in range(1,h.GetNbinsY()+1):
            y = h.GetBinContent(bx,by)
            x = h.GetBin(bx,by)
            h1.SetBinContent(x,y)
    return h1


detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]
sufxs = [0,1,2,3,4,5]
quad1 = [46.421, 44.98254, 40.425, 30.05477, 26.718, 29.86015]
quad0 = [-30.677, -27.9938, -20.38, -11.56, -11.56, -6.659]
quad2 = [-30.6775, -27.994, -20.3813, -11.56075, -3.371, -6.659]
m34   = [1, 3, 10, 28, 30, 26]

files = []
prefix = "test_data/e320_prototype_beam_Feb2025/runs/run_0000490/beam_quality"
filename = "tree_Run490_trigger_analysis.root"
for sfx in sufxs:
    path = f"{prefix}_{sfx}/{filename}"
    if(not os.path.isfile(path)): 
        files.append( None )
        continue
    files.append( ROOT.TFile(path,"READ")  )

maxima = list(range(6*6))

cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0):
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}")        
        h1 = get_h1(h)
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h1.Scale(1./n)
        h1.SetTitle(det)
        h1.GetYaxis().SetTitle("Pixels/Trigger w/o masking")
        h1.DrawCopy("hist")
        maxima[ipad-1] = h1.GetMaximumBin()
        ROOT.gPad.RedrawAxis()
        ipad += 1
cnv.Update()
cnv.SaveAs("quads_impact.pdf(")

cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0): 
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}")
        ####################
        ### "cheap mask" ###
        bmax = maxima[ipad-1]
        h.SetBinContent(bmax,0)
        ####################
        h1 = get_h1(h)
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h1.Scale(1./n)
        h1.SetTitle(det)
        h1.GetYaxis().SetTitle("Pixels/Trigger w/masking")
        h1.DrawCopy("hist")
        ROOT.gPad.RedrawAxis()
        ipad += 1
cnv.Update()
cnv.SaveAs("quads_impact.pdf")

cnv = ROOT.TCanvas("cnv","",3000,1800)
cnv.Divide(6,6)
ipad = 1
for isfx,sfx in enumerate(sufxs):
    if(files[sfx] is None):
        ipad += (len(detectors)+1)
        continue
    if((ipad-1)%6==0):
        cnv.cd(ipad)
        s = ROOT.TLatex()
        s.SetNDC(1)
        s.SetTextAlign(13)
        s.SetTextColor(ROOT.kBlack)
        s.SetTextFont(22)
        s.SetTextSize(0.17)
        s.DrawLatex(0.2,0.91,ROOT.Form("Q_{0}: %.2f kG/m" % (quad0[isfx])))
        s.DrawLatex(0.2,0.69,ROOT.Form("Q_{1}: %.2f kG/m" % (quad1[isfx])))
        s.DrawLatex(0.2,0.47,ROOT.Form("Q_{2}: %.2f kG/m" % (quad2[isfx])))
        s.DrawLatex(0.2,0.25,ROOT.Form("M_{34}: %.2f m"   % (m34[isfx])))
        ipad += 1
    for det in detectors:
        cnv.cd(ipad)
        ROOT.gPad.SetTicks(1,1)
        print(f"file:{sfx}, detector:{det} pad:{ipad}")
        h = files[sfx].Get(f"h_pix_occ_2D_{det}").Clone(f"h_{sfx}_pix_occ_2D_{det}")
        h.SetTitle(det)
        h.GetZaxis().SetTitle("Pixels/Trigger")
        ####################
        ### "cheap mask" ###
        bmax = maxima[ipad-1]
        h.SetBinContent(bmax,0)
        ####################
        n = files[sfx].Get("h_ntrgs").GetBinContent(1)
        h.Scale(1./n)
        h.DrawCopy("colz")
        ROOT.gPad.RedrawAxis()
        ipad += 1
cnv.Update()
cnv.SaveAs("quads_impact.pdf)")



