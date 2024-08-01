#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT

import config
from config import *

### based largely on this: https://www.cs.ubc.ca/~lsigal/425_2018W2/Lecture17.pdf

class HoughSeeder:
    def __init__(self,clusters,eventid=0):
        ### for not having memory leaks with the TH2D
        self.eventid = eventid
        ### the clusters per detector
        self.x0 = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.y0 = []
        self.y1 = []
        self.y2 = []
        self.y3 = []
        self.z0 = []
        self.z1 = []
        self.z2 = []
        self.z3 = []
        self.clsid0 = []
        self.clsid1 = []
        self.clsid2 = []
        self.clsid3 = []
        self.detid0 = []
        self.detid1 = []
        self.detid2 = []
        self.detid3 = []
        ### all the clusters together
        self.clsid = []
        self.detid = []
        self.x = []
        self.y = []
        self.z = []
        ### the seed
        self.clsid_seed = []
        self.detid_seed = []
        self.x_seed = []
        self.y_seed = []
        self.z_seed = []
        ### the seed clusters per detector
        self.seed_clusters = {}
        for det in cfg["detectors"]: self.seed_clusters.update({det:[]})
        self.seed_clusters_per_detector = [0]*len(cfg["detectors"])
        ### other constants
        self.npix_x = cfg["npix_x"]
        self.npix_y = cfg["npix_y"]
        self.pix_x  = cfg["pix_x"]
        self.pix_y  = cfg["pix_y"]
        self.xepsilon = 1e-6
        self.fepsilon = 1e-6
        #TODO: this has to be optimized!!!!
        nclusters = 1
        for det in cfg["detectors"]: nclusters *= len(clusters[det])
        self.nbins = -1
        if  (nclusters<=20):                   self.nbins = 50
        elif(nclusters>20 and nclusters<=200): self.nbins = 250
        elif(nclusters>200):                   self.nbins = 500
        self.minintersections = math.comb(4,2)
        
        ### set the clusters
        self.set_clusters(clusters)
        self.zmin = self.z0[0]
        self.zmax = self.z3[0]
        self.zmid = (self.zmax-self.zmin)/2.
        self.zmin = self.zmin-self.zmid
        self.zmax = self.zmax+self.zmid
        ### set the lines
        self.zx_functions,self.zx_fmin,self.zx_fmax = self.set_lines("zx",self.z,self.x)
        self.zy_functions,self.zy_fmin,self.zy_fmax = self.set_lines("zy",self.z,self.y)
        ### set the 2D hist and map
        self.h2Freq_zx,self.imap_zx = self.set_histmap("zx",self.zx_functions,[self.zx_fmin,self.zx_fmax])
        self.h2Freq_zy,self.imap_zy = self.set_histmap("zy",self.zy_functions,[self.zy_fmin,self.zy_fmax])
        ### the hough candidates
        self.hough_points_zx,self.zx_cls = self.candidates("zx",self.h2Freq_zx,self.imap_zx)
        self.hough_points_zy,self.zy_cls = self.candidates("zy",self.h2Freq_zy,self.imap_zy)
        ### set the seeds and summary statistics
        self.summary = self.set_seeds(clusters)

    def __str__(self):
        return f"Seeder"

    def set_clusters(self,clusters):
        for det in cfg["detectors"]:
            for c in clusters[det]:
                if(det=="ALPIDE_0"):
                    self.clsid0.append( c.CID )
                    self.detid0.append( c.DID )
                    self.x0.append( c.xmm )
                    self.y0.append( c.ymm )
                    self.z0.append( c.zmm )
                if(det=="ALPIDE_1"):
                    self.clsid1.append( c.CID )
                    self.detid1.append( c.DID )
                    self.x1.append( c.xmm )
                    self.y1.append( c.ymm )
                    self.z1.append( c.zmm )
                if(det=="ALPIDE_2"):
                    self.clsid2.append( c.CID )
                    self.detid2.append( c.DID )
                    self.x2.append( c.xmm )
                    self.y2.append( c.ymm )
                    self.z2.append( c.zmm )
                if(det=="ALPIDE_3"):
                    self.clsid3.append( c.CID )
                    self.detid3.append( c.DID )
                    self.x3.append( c.xmm )
                    self.y3.append( c.ymm )
                    self.z3.append( c.zmm )
        self.clsid = self.clsid0+self.clsid1+self.clsid2+self.clsid3
        self.detid = self.detid0+self.detid1+self.detid2+self.detid3
        self.x = self.x0+self.x1+self.x2+self.x3
        self.y = self.y0+self.y1+self.y2+self.y3
        self.z = self.z0+self.z1+self.z2+self.z3

    def color(self,zref):
        col = -1
        if  (zref==self.z0[0]): col = ROOT.kRed
        elif(zref==self.z1[0]): col = ROOT.kBlack
        elif(zref==self.z2[0]): col = ROOT.kGreen-2
        elif(zref==self.z3[0]): col = ROOT.kMagenta
        else:
            print("zref does not match any of z0-z4. Quitting")
            quit()
        return col
    
    def graph(self,name,x,y,col,xlim,ylim):
        g = ROOT.TGraph(len(x),array.array("d",x),array.array("d",y))
        g.SetTitle(name+";x [mm];y [mm]")
        g.SetMarkerColor(col)
        g.SetMarkerStyle(20)
        g.GetXaxis().SetLimits(xlim[0],xlim[1])
        g.GetYaxis().SetLimits(ylim[0],ylim[1])
        return g

    def multigraph(self,name,title,z,k,zlim,klim):
        mg = ROOT.TMultiGraph()
        for i in range(len(z)):
            g = ROOT.TGraph(1,array.array("d",[z[i]]),array.array("d",[k[i]]))
            col = self.color(z[i])
            g.SetMarkerColor(col)
            g.SetMarkerStyle(20)
            g.GetXaxis().SetLimits(zlim[0],zlim[1])
            g.GetYaxis().SetLimits(klim[0],klim[1])
            mg.Add(g)
        mg.SetName(name)
        mg.SetTitle(title)
        mg.GetXaxis().SetLimits(zlim[0],zlim[1])
        mg.GetYaxis().SetLimits(klim[0],klim[1])
        return mg

    def intersect(self,f1,f2):
        name1 = f1.GetName()+"_flat"
        name2 = f2.GetName()+"_flat"
        flat1 = ROOT.TF1(name1,f"{f1.GetParameter(1)}*sin(x)+{f1.GetParameter(0)}*cos(x)",f1.GetXmin(),f1.GetXmax())
        flat2 = ROOT.TF1(name2,f"{f2.GetParameter(1)}*sin(x)+{f2.GetParameter(0)}*cos(x)",f2.GetXmin(),f2.GetXmax())
        diff = ROOT.TF1(name1+"-"+name2,"abs("+name1+"-"+name2+")",f1.GetXmin(),f1.GetXmax())
        mindiff  = diff.GetMinimum()
        xminimum = diff.GetMinimumX()
        yminimum = flat1.Eval(xminimum)
        return mindiff,xminimum,yminimum

    def candidates(self,name,h2,imap,minintersections=6):
        hough_points = []
        arr_clusters = []
        for bx in range(1,h2.GetNbinsX()+1):
            for by in range(1,h2.GetNbinsY()+1):
                if(h2.GetBinContent(bx,by)>=self.minintersections):
                    theta = h2.GetXaxis().GetBinCenter(bx)
                    rho   = h2.GetYaxis().GetBinCenter(by)
                    hough_points.append({"theta":theta,"rho":rho,"imap":imap[(bx,by)]})
                    for k in imap[(bx,by)]:
                        for l in k:
                            if(l in arr_clusters): continue
                            arr_clusters.append(l) 
        return hough_points,arr_clusters

    def set_lines(self,name,z,k):
        functions = {}
        fmin = +1e-20
        fmax = -1e-20
        for i in range(len(z)):
            fname = f"{name}_{i}"
            col = self.color(z[i])
            functions.update( { fname : ROOT.TF1(fname,"[1]*sin(x)+[0]*cos(x)",0,+np.pi,2) } )
            functions[fname].SetParameter(0,z[i])
            functions[fname].SetParameter(1,k[i])
            functions[fname].SetLineWidth(1)
            functions[fname].SetLineColor(col)
            if(functions[fname].GetMinimum()<fmin): fmin = functions[fname].GetMinimum()
            if(functions[fname].GetMaximum()>fmax): fmax = functions[fname].GetMaximum()
        return functions,fmin,fmax
    
    def set_histmap(self,name,functions,flim):
        h2Freq = ROOT.TH2D("h2Freq_"+name+"_"+str(self.eventid),";#theta;#rho;Frequency",self.nbins,0,+np.pi,self.nbins,flim[0]*1.2,flim[1]*1.2)
        imap = {}
        for i,namei in enumerate(functions):
            for j,namej in enumerate(functions):
                if(j<=i): continue
                if(functions[namei].GetLineColor()==functions[namej].GetLineColor()): continue
                mindiff,xmindiff,ymindiff = self.intersect(functions[namei],functions[namej])
                # if(abs(xmindiff-np.pi/2)<self.xepsilon): continue
                if(abs(xmindiff-0)<self.xepsilon):       continue
                if(abs(xmindiff-np.pi)<self.xepsilon):   continue
                if(mindiff>self.fepsilon):               continue
                h2Freq.Fill(xmindiff,ymindiff)
                bx = h2Freq.GetXaxis().FindBin(xmindiff)
                by = h2Freq.GetYaxis().FindBin(ymindiff)
                if((bx,by) not in imap): imap.update({(bx,by):[[i,j]]})
                else:                    imap[(bx,by)].append([i,j])
        return h2Freq,imap
    
    def set_seeds(self,clusters):
        n_common_clusters = 0
        for k in range(len(self.x)):
            if(k not in self.zx_cls and k not in self.zy_cls): continue
            if(k in self.zx_cls and k in self.zy_cls): n_common_clusters += 1
            det = cfg["detectors"][self.detid[k]]
            self.seed_clusters_per_detector[self.detid[k]] += 1
            self.seed_clusters[det].append( clusters[det][self.clsid[k]] )
            self.clsid_seed.append( self.clsid[k] )
            self.detid_seed.append( self.detid[k] )
            self.x_seed.append( self.x[k] )
            self.y_seed.append( self.y[k] )
            self.z_seed.append( self.z[k] )
        nplanes = 0
        nseeds = 1
        for n in self.seed_clusters_per_detector:
            if(n>0): nplanes+=1
            nseeds *= n
        found_intrscts_zx = (len(self.hough_points_zx)>0)
        found_intrscts_zy = (len(self.hough_points_zy)>0)
        found_intrscts_zx_and_zy = (found_intrscts_zx and found_intrscts_zy)
        summary = {"nplanes":nplanes,
                   "nseeds":nseeds,
                   "found_intrscts_zx":found_intrscts_zx,
                   "found_intrscts_zy":found_intrscts_zy,
                   "found_intrscts_zx_and_zy": found_intrscts_zx_and_zy,
                   "n_common_clusters":n_common_clusters}
        return summary
    
    def clear_h2Freq(self):
        del self.h2Freq_zx
        del self.h2Freq_zy
    
    def plot(self,plotsuff):
        ROOT.gROOT.SetBatch(1)
        ROOT.gStyle.SetOptFit(0)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPadBottomMargin(0.15)
        ROOT.gStyle.SetPadLeftMargin(0.13)
        ROOT.gStyle.SetPadRightMargin(0.15)
        
        plotname = "hough_transform_"+plotsuff+".pdf"
        
        gxy0 = self.graph("ALPIDE_0",self.x0,self.y0,ROOT.kRed,     [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )
        gxy1 = self.graph("ALPIDE_1",self.x1,self.y1,ROOT.kBlack,   [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )
        gxy2 = self.graph("ALPIDE_2",self.x2,self.y2,ROOT.kGreen-2, [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )
        gxy3 = self.graph("ALPIDE_3",self.x3,self.y3,ROOT.kMagenta, [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )

        cnv = ROOT.TCanvas("cnv","",1000,1000)
        cnv.Divide(2,2)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        gxy0.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        gxy1.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.cd(3)
        ROOT.gPad.SetTicks(1,1)
        gxy2.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.cd(4)
        ROOT.gPad.SetTicks(1,1)
        gxy3.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.SaveAs(plotname+"(")
        
        zx_mg = self.multigraph("zx","Telescope x vs z;z [mm];x [mm]",self.z,self.x,[self.zmin,self.zmax],[-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2])
        zy_mg = self.multigraph("zy","Telescope x vs y;z [mm];y [mm]",self.z,self.y,[self.zmin,self.zmax],[-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2])
        cnv = ROOT.TCanvas("cnv","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        zx_mg.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        zy_mg.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.SaveAs(plotname)
        
        arrows_zx = []
        for point in self.hough_points_zx:
            arw = ROOT.TArrow(2.5,50,point["theta"],point["rho"],0.01)
            arrows_zx.append( arw )
        arrows_zy = []
        for point in self.hough_points_zy:
            arw = ROOT.TArrow(2.5,50,point["theta"],point["rho"],0.01)
            arrows_zy.append( arw )
    
        cnv = ROOT.TCanvas("cnv","",1500,1200)
        cnv.Divide(2,2)
        cnv.cd(1)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        ROOT.gPad.SetTicks(1,1)
        first = True    
        for name,f in self.zx_functions.items():
            f.SetMinimum(self.zx_fmin*1.2)
            f.SetMaximum(self.zx_fmax*1.2)
            if(first):
                f.Draw("l")
                f.SetTitle("Hough transform in z-x")
                first = False
            else: f.Draw("l same")
            for arw in arrows_zx: arw.Draw()
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        first = True
        for name,f in self.zy_functions.items():
            f.SetMinimum(self.zy_fmin*1.2)
            f.SetMaximum(self.zy_fmax*1.2)
            if(first):
                f.Draw("l")
                f.SetTitle("Hough transform in z-y")
                first = False
            else: f.Draw("l same")
            for arw in arrows_zy: arw.Draw()
        ROOT.gPad.RedrawAxis()
        cnv.cd(3)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        ROOT.gPad.SetTicks(1,1)
        self.h2Freq_zx.Draw("colz")
        for arw in arrows_zx: arw.Draw()
        ROOT.gPad.RedrawAxis()
        cnv.cd(4)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        ROOT.gPad.SetTicks(1,1)
        self.h2Freq_zy.Draw("colz")
        for arw in arrows_zy: arw.Draw()
        ROOT.gPad.RedrawAxis()
        cnv.SaveAs(plotname)
    
        zx_seed_mg = self.multigraph("zx_seed","Post-seeding Telescope x vs z;z [mm];x [mm]",self.z_seed,self.x_seed,[self.zmin,self.zmax],[-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2])
        zy_seed_mg = self.multigraph("zy_seed","Post-seeding Telescope x vs y;z [mm];y [mm]",self.z_seed,self.y_seed,[self.zmin,self.zmax],[-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2])
        cnv = ROOT.TCanvas("cnv","",1000,500)
        cnv.Divide(2,1)
        cnv.cd(1)
        ROOT.gPad.SetTicks(1,1)
        zx_seed_mg.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        zy_seed_mg.Draw("AP")
        ROOT.gPad.RedrawAxis()
        cnv.SaveAs(plotname+")")

###############################################
###############################################
###############################################

if __name__ == "__main__":
    ROOT.gROOT.SetBatch(1)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadBottomMargin(0.15)
    ROOT.gStyle.SetPadLeftMargin(0.13)
    ROOT.gStyle.SetPadRightMargin(0.15)
    
    # ### must be called here (first) and only once!
    # configfile = "conf/config_cosmics_data_E320_full_prototype_tracker_JulX.txt"
    # init_config(configfile,True)
    
    '''
    ALPIDE_0:
      [0]: Cluster: x=20.5, y=208.5, r=((-14.371459999999999, -1.2767999999999997, 0.0)) [mm], size=4
      [1]: Cluster: x=487.0833333333333, y=509.75, r=((-0.7285633333333337, 6.8208, 0.0)) [mm], size=12
      [2]: Cluster: x=823.7142857142857, y=360.0, r=((9.114525714285714, 2.79552, 0.0)) [mm], size=7
    ALPIDE_1:
      [0]: Cluster: x=317.6666666666667, y=322.6666666666667, r=((-6.203108427456671, 1.3208135161224561, 36.03)) [mm], size=3
      [1]: Cluster: x=374.5, y=79.5, r=((-4.535280427996441, -5.2139726551301475, 36.03)) [mm], size=4
      [2]: Cluster: x=483.6666666666667, y=150.66666666666666, r=((-1.3450108964715013, -3.2980725849451806, 36.03)) [mm], size=3
      [3]: Cluster: x=990.0, y=320.0, r=((13.455975935799831, 1.267245788199794, 36.03)) [mm], size=1
    ALPIDE_2:
      [0]: Cluster: x=74.5, y=161.5, r=((-13.500444390743365, -3.182909549105556, 60.05)) [mm], size=4
      [1]: Cluster: x=101.5, y=165.5, r=((-12.710752666409318, -3.0769557793834688, 60.05)) [mm], size=4
      [2]: Cluster: x=370.64285714285717, y=91.92857142857143, r=((-4.844953789036907, -5.070162361349382, 60.05)) [mm], size=14
      [3]: Cluster: x=510.0, y=395.0, r=((-0.7539993678639819, 3.0682988004468266, 60.05)) [mm], size=1
      [4]: Cluster: x=733.5, y=505.5, r=((5.787019566681159, 6.025569801745483, 60.05)) [mm], size=4
    ALPIDE_3:
      [0]: Cluster: x=19.333333333333332, y=441.3333333333333, r=((-15.21916880313665, 4.239481072259748, 84.07)) [mm], size=3
      [1]: Cluster: x=228.0, y=482.5, r=((-9.11748735949526, 5.34456172908987, 84.07)) [mm], size=2
      [2]: Cluster: x=251.0, y=229.0, r=((-8.446619478625736, -1.4696811256385718, 84.07)) [mm], size=1
      [3]: Cluster: x=285.0, y=76.0, r=((-7.453456632905826, -5.582562042597089, 84.07)) [mm], size=1
      [4]: Cluster: x=382.5, y=304.0, r=((-4.6010708048453814, 0.5453865658264474, 84.07)) [mm], size=2
      [5]: Cluster: x=700.5, y=464.5, r=((4.698294925598414, 4.857372026313013, 84.07)) [mm], size=4
    '''
    ###
    class DummyCls:
        def __init__(self,x,y,z,det,CID):
            self.CID  = CID
            self.DID  = ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3"].index(det)
            self.det  = det
            self.xmm  = x
            self.ymm  = y
            self.zmm  = z
    ###
    clusters = {"ALPIDE_0":[], "ALPIDE_1":[], "ALPIDE_2":[], "ALPIDE_3":[]}
    ###
    clusters["ALPIDE_0"].append( DummyCls(-14.371459999999999,-1.2767999999999997,0.0, "ALPIDE_0",0) )
    clusters["ALPIDE_0"].append( DummyCls(-0.7285633333333337,6.8208,0.0, "ALPIDE_0",1) )
    clusters["ALPIDE_0"].append( DummyCls(9.114525714285714,2.79552,0.0, "ALPIDE_0",2) )
    ###
    clusters["ALPIDE_1"].append( DummyCls(-6.203108427456671,1.3208135161224561,36.03, "ALPIDE_1",0) )
    clusters["ALPIDE_1"].append( DummyCls(-4.535280427996441,-5.2139726551301475,36.03, "ALPIDE_1",1) )
    clusters["ALPIDE_1"].append( DummyCls(-1.3450108964715013,-3.2980725849451806,36.03, "ALPIDE_1",2) )
    clusters["ALPIDE_1"].append( DummyCls(13.455975935799831,1.267245788199794,36.03, "ALPIDE_1",3) )
    ###
    clusters["ALPIDE_2"].append( DummyCls(-13.500444390743365,-3.182909549105556,60.05, "ALPIDE_2",0) )
    clusters["ALPIDE_2"].append( DummyCls(-12.710752666409318,-3.0769557793834688,60.05, "ALPIDE_2",1) )
    clusters["ALPIDE_2"].append( DummyCls(-4.844953789036907,-5.070162361349382,60.05, "ALPIDE_2",2) )
    clusters["ALPIDE_2"].append( DummyCls(-0.7539993678639819,3.0682988004468266,60.05, "ALPIDE_2",3) )
    clusters["ALPIDE_2"].append( DummyCls(5.787019566681159,6.025569801745483,60.05, "ALPIDE_2",4) )
    ###
    clusters["ALPIDE_3"].append( DummyCls(-15.21916880313665,4.239481072259748,84.07, "ALPIDE_3",0) )
    clusters["ALPIDE_3"].append( DummyCls(-9.11748735949526,5.34456172908987,84.07, "ALPIDE_3",1) )
    clusters["ALPIDE_3"].append( DummyCls(-8.446619478625736,-1.4696811256385718,84.07, "ALPIDE_3",2) )
    clusters["ALPIDE_3"].append( DummyCls(-7.453456632905826,-5.582562042597089,84.07, "ALPIDE_3",3) )
    clusters["ALPIDE_3"].append( DummyCls(-4.6010708048453814,0.5453865658264474,84.07, "ALPIDE_3",4) )
    clusters["ALPIDE_3"].append( DummyCls(4.698294925598414,4.857372026313013,84.07, "ALPIDE_3",5) )
    ###
    seeder = HoughSeeder(clusters,True)
    for det in ["ALPIDE_0","ALPIDE_1","ALPIDE_2","ALPIDE_3"]:
        for seed_cluster in seeder.seed_clusters[det]:
            print(f"seed_cluster {seed_cluster.det}:{seed_cluster.CID}")
    
    seeder.plot()
        
        