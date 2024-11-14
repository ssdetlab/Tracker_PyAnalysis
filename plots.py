import ROOT


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

detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3"]

def h2max(hname,norm):
    hmax = -1
    for det in detectors:
        y = f.Get(f"{det}/{hname}_{det}").GetMaximum()/norm
        if(y>=hmax): hmax = y
    return hmax


# f = ROOT.TFile("tree_11_03_2024_Run404_multiprocess_histograms.root","READ")
# f = ROOT.TFile("tree_11_05_2024_Run446_multiprocess_histograms_250triggers.root","READ")
# f = ROOT.TFile("tree_11_03_2024_Run405_multiprocess_histograms_250triggers.root","READ")
f = ROOT.TFile("tree_11_04_2024_Run409_multiprocess_histograms_250triggers.root","READ")

### get the number of triggers
nTriggers = int(f.Get(f"h_events").GetBinContent(1))

cnv = ROOT.TCanvas("cnv_pix_occ","",1200,1000)
cnv.Divide(2,2)
hmax = h2max("h_pix_occ_2D",nTriggers)
for i,det in enumerate(detectors):
    p = cnv.cd(i+1)
    p.SetTicks(1,1)
    h = f.Get(f"{det}/h_pix_occ_2D_{det}")
    h.SetTitle(f"{det}: pixels occupancy for {nTriggers} triggers")
    h.GetZaxis().SetTitle(f"Pixels per trigger")
    h.Scale(1./float(nTriggers))
    h.SetMaximum(hmax*1.1)
    h.GetZaxis().SetTitleOffset(1.3)
    h.Draw("colz")
cnv.SaveAs("plots.pdf(")

cnv = ROOT.TCanvas("cnv_cls_occ","",1200,1000)
cnv.Divide(2,2)
hmax = h2max("h_cls_occ_2D",nTriggers)
for i,det in enumerate(detectors):
    p = cnv.cd(i+1)
    p.SetTicks(1,1)
    h = f.Get(f"{det}/h_cls_occ_2D_{det}")
    h.SetTitle(f"{det}: clusters occupancy for {nTriggers} triggers")
    h.GetZaxis().SetTitle(f"Clusters per trigger")
    h.Scale(1./float(nTriggers))
    h.SetMaximum(hmax*1.1)
    h.GetZaxis().SetTitleOffset(1.3)
    h.Draw("colz")
cnv.SaveAs("plots.pdf")

cnv = ROOT.TCanvas("cnv_cls_sze","",1200,1000)
cnv.Divide(2,2)
hmax = h2max("h_csize_vs_y",nTriggers)
for i,det in enumerate(detectors):
    p = cnv.cd(i+1)
    p.SetTicks(1,1)
    p.SetLogz()
    h = f.Get(f"{det}/h_csize_vs_y_{det}")
    h.SetTitle(f"{det}: cluster size vs y for {nTriggers} triggers")
    h.GetZaxis().SetTitle(f"Clusters per trigger")
    h.Scale(1./float(nTriggers))
    h.SetMaximum(hmax*1.5)
    h.Draw("colz")
cnv.SaveAs("plots.pdf)")