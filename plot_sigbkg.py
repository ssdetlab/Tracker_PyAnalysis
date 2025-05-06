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


def h1h2max(h1,h2):
    hmax = -1
    y1 = h1.GetMaximum()
    y2 = h2.GetMaximum()
    hmax = y1 if(y1>y2) else y2
    return hmax

def h1h2min(h1,h2):
    hmax = 1e10
    for b in range(1,h1.GetNbinsX()+1):
        y1 = h1.GetBinContent(b)
        y2 = h2.GetBinContent(b)
        hmax = y1 if(y1<hmax and y1>0) else hmax
        hmax = y2 if(y2<hmax and y2>0) else hmax
    return hmax


fS = ROOT.TFile("test_data/e320_prototype_beam_Feb2025/runs/run_0000502/tree_Run502_dipole_window.root","READ")
fB = ROOT.TFile("test_data/e320_prototype_beam_Feb2025/runs/run_0000503/tree_Run503_dipole_window.root","READ")


hTrgS = fS.Get("hTriggers").Clone("TriggerS")
hTrgB = fB.Get("hTriggers").Clone("TriggerB")
nTrgS = hTrgS.GetBinContent(2)-361 ### TODO!!! since the run starts at trig=362 and not in 0, and 0-361 is in the list of bad triggers but not in the root file.
nTrgB = hTrgB.GetBinContent(2)
print(f"nTrgS={nTrgS}, nTrgB={nTrgB}")


leg = ROOT.TLegend(0.2,0.8,0.6,0.87)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetTextSize(0.037)
leg.SetBorderSize(0)
hS0 = fS.Get("hPf_zoom").Clone("P_0_S")
hB0 = fB.Get("hPf_zoom").Clone("P_0_B")
hS0.GetXaxis().SetTitle("p_{z} [GeV]")
hB0.GetXaxis().SetTitle("p_{z} [GeV]")
hS0.SetLineColor(ROOT.kBlue)
hS0.SetFillColorAlpha(ROOT.kBlue,0.35)
hB0.SetLineColor(ROOT.kRed)
hB0.SetFillColorAlpha(ROOT.kRed,0.35)
leg.AddEntry(hS0,"Run502: Beam+Beryllium","f")
leg.AddEntry(hB0,"Run503: Beam-only","f")



cnv = ROOT.TCanvas("cnv","",500,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
hS = fS.Get("hPf_zoom").Clone("P_S")
hB = fB.Get("hPf_zoom").Clone("P_B")
hS.Scale(1./nTrgS)
hB.Scale(1./nTrgB)
hmax = h1h2max(hS,hB)
hmin = h1h2min(hS,hB)
hS.GetYaxis().SetTitle("Tracks/Trigger")
hB.GetYaxis().SetTitle("Tracks/Trigger")
hS.GetXaxis().SetTitle("p_{z} [GeV]")
hB.GetXaxis().SetTitle("p_{z} [GeV]")
hS.SetLineColor(ROOT.kBlue)
hS.SetFillColorAlpha(ROOT.kBlue,0.35)
hB.SetLineColor(ROOT.kRed)
hB.SetFillColorAlpha(ROOT.kRed,0.35)
hS.SetMaximum(5*hmax)
hB.SetMaximum(5*hmax)
hS.SetMinimum(0.5*hmin)
hB.SetMinimum(0.5*hmin)
hS.Draw("hist")
hB.Draw("hist same")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs("plot_sigbkg.pdf(")

cnv = ROOT.TCanvas("cnv","",500,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
hS = fS.Get("hTheta_yz_after_cuts").Clone("Theta_yz_S")
hB = fB.Get("hTheta_yz_after_cuts").Clone("Theta_yz_B")
hS.Scale(1./nTrgS)
hB.Scale(1./nTrgB)
hmax = h1h2max(hS,hB)
hmin = h1h2min(hS,hB)
hS.GetYaxis().SetTitle("Tracks/Trigger")
hB.GetYaxis().SetTitle("Tracks/Trigger")
hS.GetXaxis().SetTitle("#theta_{yz} [rad]")
hB.GetXaxis().SetTitle("#theta_{yz} [rad]")
hS.SetLineColor(ROOT.kBlue)
hS.SetFillColorAlpha(ROOT.kBlue,0.35)
hB.SetLineColor(ROOT.kRed)
hB.SetFillColorAlpha(ROOT.kRed,0.35)
hS.SetMaximum(5*hmax)
hB.SetMaximum(5*hmax)
hS.SetMinimum(0.5*hmin)
hB.SetMinimum(0.5*hmin)
hS.Draw("hist")
hB.Draw("hist same")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs("plot_sigbkg.pdf")

cnv = ROOT.TCanvas("cnv","",500,500)
cnv.cd()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
hS = fS.Get("hTheta_xz_after_cuts").Clone("Theta_yz_S")
hB = fB.Get("hTheta_xz_after_cuts").Clone("Theta_yz_B")
hS.Scale(1./nTrgS)
hB.Scale(1./nTrgB)
hmax = h1h2max(hS,hB)
hS.GetYaxis().SetTitle("Tracks/Trigger")
hB.GetYaxis().SetTitle("Tracks/Trigger")
hS.GetXaxis().SetTitle("#theta_{xz} [rad]")
hB.GetXaxis().SetTitle("#theta_{xz} [rad]")
hS.SetLineColor(ROOT.kBlue)
hS.SetFillColorAlpha(ROOT.kBlue,0.35)
hB.SetLineColor(ROOT.kRed)
hB.SetFillColorAlpha(ROOT.kRed,0.35)
hS.SetMaximum(5*hmax)
hB.SetMaximum(5*hmax)
hS.SetMinimum(0.5*hmin)
hB.SetMinimum(0.5*hmin)
hS.Draw("hist")
hB.Draw("hist same")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs("plot_sigbkg.pdf)")


