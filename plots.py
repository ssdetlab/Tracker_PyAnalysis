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

detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3","ALPIDE_4"]

def h2max(hname,norm):
    hmax = -1
    for det in detectors:
        y = f.Get(f"{det}/{hname}_{det}").GetMaximum()/norm
        if(y>=hmax): hmax = y
    return hmax


f = ROOT.TFile("test_data/e320_prototype_beam_2024/runs/run_0000405/tree_11_03_2024_Run405_multiprocess_histograms.root","READ")

# f = ROOT.TFile("test_data/e320_prototype_beam_Feb2025/runs/run_0000490/tree_2_Run490_multiprocess_histograms_notrk.root","READ")
# f = ROOT.TFile("test_data/e320_prototype_beam_Feb2025/runs/run_0000490/tree_Run490_multiprocess_histograms_notrk.root","READ")

cnvx = 500
cnvy = 1500
dividey = 5
if("2024" in f.GetName()):
    detectors = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3"]
    cnvx = 500
    cnvy = 1000
    dividey = 4

### get the number of triggers
nTriggers = int(f.Get(f"h_events").GetBinContent(1))


cnv = ROOT.TCanvas("cnv_pix_occ_0","",500,250)
cnv.SetTicks(1,1)
h = f.Get("ALPIDE_0/h_pix_occ_2D_ALPIDE_0")
h.SetTitle("ALPIDE_0: average pixel occupancy over "+str(nTriggers)+" BXs;#it{x} (pixel number);#it{y} (pixel number);Average number of fired pixels/BX")
h.Scale(1./float(nTriggers))
# h.SetMaximum(hmax*1.1)
h.GetZaxis().SetTitleOffset(0.95)
h.Draw("colz")
cnv.SaveAs("nov_occ.pdf")
cnv.SaveAs("nov_occ.png")



cnv = ROOT.TCanvas("cnv_pix_occ","",cnvx,cnvy)
cnv.Divide(1,dividey)
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

cnv = ROOT.TCanvas("cnv_cls_occ","",cnvx,cnvy)
cnv.Divide(1,dividey)
hmax = h2max("h_cls_occ_2D",nTriggers)
for i,det in enumerate(detectors):
    p = cnv.cd(i+1)
    p.SetTicks(1,1)
    p.SetLogz()
    h = f.Get(f"{det}/h_cls_occ_2D_{det}")
    h.SetTitle(f"{det}: clusters occupancy for {nTriggers} triggers")
    h.GetZaxis().SetTitle(f"Clusters per trigger")
    h.Scale(1./float(nTriggers))
    h.SetMaximum(hmax*1.1)
    h.GetZaxis().SetTitleOffset(1.3)
    h.Draw("colz")
cnv.SaveAs("plots.pdf")

cnv = ROOT.TCanvas("cnv_cls_sze","",cnvx,cnvy)
cnv.Divide(1,dividey)
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