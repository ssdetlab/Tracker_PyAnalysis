#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT

import config
from config import *
import lookup_table
from lookup_table import *

### based largely on this: https://www.cs.ubc.ca/~lsigal/425_2018W2/Lecture17.pdf
### see also https://www.sciencedirect.com/science/article/pii/S0167865500000441?via%3Dihub

def fwave(theta,k,z):
    rho = k*math.sin(theta) + z*math.cos(theta)
    return rho
    
def fdiff(theta,k1,z1,k2,z2):
    return fwave(theta,k1,z1)-fwave(theta,k2,z2)


class HoughSeeder:
    def __init__(self,clusters,eventid=0):
        ### for not having memory leaks with the TH2D
        self.eventid = eventid
        nclusters = 0
        for det in cfg["detectors"]: nclusters += len(clusters[det])
        ### colors
        self.zcols = [ROOT.kRed, ROOT.kBlack, ROOT.kGreen-2, ROOT.kMagenta]
        ### the clusters per detector
        n0 = len(clusters[cfg["detectors"][0]])
        n1 = len(clusters[cfg["detectors"][1]])
        n2 = len(clusters[cfg["detectors"][2]])
        n3 = len(clusters[cfg["detectors"][3]])
        self.x0 = np.zeros(n0)
        self.x1 = np.zeros(n1)
        self.x2 = np.zeros(n2)
        self.x3 = np.zeros(n3)
        self.y0 = np.zeros(n0)
        self.y1 = np.zeros(n1)
        self.y2 = np.zeros(n2)
        self.y3 = np.zeros(n3)
        self.z0 = np.zeros(n0)
        self.z1 = np.zeros(n1)
        self.z2 = np.zeros(n2)
        self.z3 = np.zeros(n3)
        ### all the clusters together
        self.x = None
        self.y = None
        self.z = None
        ### all seed clusters
        ### other constants
        self.npix_x = cfg["npix_x"]
        self.npix_y = cfg["npix_y"]
        self.pix_x  = cfg["pix_x"]
        self.pix_y  = cfg["pix_y"]
        self.xepsilon = 1e-15
        self.fepsilon = 1e-15
        #TODO: this has to be optimized!!!!
        self.theta_x_scale = cfg["seed_thetax_scale"]
        self.rho_x_scale   = cfg["seed_rhox_scale"]
        self.theta_y_scale = cfg["seed_thetay_scale"]
        self.rho_y_scale   = cfg["seed_rhoy_scale"]
        self.thetamin_x = np.pi/2-self.theta_x_scale*np.pi/2.
        self.thetamax_x = np.pi/2+self.theta_x_scale*np.pi/2.
        self.thetamin_y = np.pi/2-self.theta_y_scale*np.pi/2.
        self.thetamax_y = np.pi/2+self.theta_y_scale*np.pi/2.
        self.nbins_theta = -1
        if(nclusters<=20):                     self.nbins_theta = cfg["seed_nbins_theta_020"]
        elif(nclusters>20 and nclusters<=200): self.nbins_theta = cfg["seed_nbins_theta_200"]
        elif(nclusters>200):                   self.nbins_theta = cfg["seed_nbins_theta_inf"]
        self.nbins_rho = -1
        if(nclusters<=20):                     self.nbins_rho = cfg["seed_nbins_rho_020"]
        elif(nclusters>20 and nclusters<=200): self.nbins_rho = cfg["seed_nbins_rho_200"]
        elif(nclusters>200):                   self.nbins_rho = cfg["seed_nbins_rho_inf"]
        self.minintersections = math.comb(len(cfg["detectors"]),2) ### all pairs out of for detectors w/o repetitions
        ### set the clusters
        self.set_clusters(clusters)
        self.zmin = self.z0[0]
        self.zmax = self.z3[0]
        self.zmid = (self.zmax-self.zmin)/2.
        self.zmin = self.zmin-self.zmid
        self.zmax = self.zmax+self.zmid
        ### set the waves
        self.zx_wave_fmin,self.zx_wave_fmax = self.get_waves_limits("zx",self.z,self.x,self.thetamin_x,self.thetamax_x)
        self.zy_wave_fmin,self.zy_wave_fmax = self.get_waves_limits("zy",self.z,self.y,self.thetamin_y,self.thetamax_y)
        self.rhomin_x = self.zx_wave_fmin*self.rho_x_scale
        self.rhomax_x = self.zx_wave_fmax*self.rho_x_scale
        self.rhomin_y = self.zy_wave_fmin*self.rho_y_scale
        self.rhomax_y = self.zy_wave_fmax*self.rho_y_scale
        ### define the wave parameter space
        self.h2waves_zx = self.define_theta_rho_axes("zx",self.thetamin_x,self.thetamax_x,self.rhomin_x,self.rhomax_x)
        self.h2waves_zy = self.define_theta_rho_axes("zy",self.thetamin_y,self.thetamax_y,self.rhomin_y,self.rhomax_y)
        ### the data 4D structure 
        self.accumulator = {}
        ### fill the accumulator
        self.fill_4d_wave_intersections(clusters)
        ### get the 4D bin numbers of the good coordinates
        self.cells = self.get_seed_coordinates()
        del self.accumulator
        ### 
        self.LUT = LookupTable(clusters,eventid)
        self.tunnels,self.hough_coord = self.get_tunnels()
        self.tunnel_nsseds, self.seeds = self.set_seeds(clusters)
        self.nseeds = len(self.seeds)
        print(f"Got {len(self.tunnels)} valid tunnels out of {len(self.cells)} tunnels.")
        print(f"N seeds per tunnel:\n{self.tunnel_nsseds}.")
        print(f"Total of {len(self.seeds)} seeds.")
        
        '''
        ### clear memory if there are no seeds
        if(self.summary["nseeds"]==0):
            self.clear_h2Freq()
            self.clear_functions()
        '''

    def __str__(self):
        return f"Seeder"

    def set_clusters(self,clusters):
        for det in cfg["detectors"]:
            for i,c in enumerate(clusters[det]):
                if(det=="ALPIDE_0"):
                    self.x0[i] = c.xmm
                    self.y0[i] = c.ymm
                    self.z0[i] = c.zmm
                if(det=="ALPIDE_1"):
                    self.x1[i] = c.xmm
                    self.y1[i] = c.ymm
                    self.z1[i] = c.zmm
                if(det=="ALPIDE_2"):
                    self.x2[i] = c.xmm
                    self.y2[i] = c.ymm
                    self.z2[i] = c.zmm
                if(det=="ALPIDE_3"):
                    self.x3[i] = c.xmm
                    self.y3[i] = c.ymm
                    self.z3[i] = c.zmm
        self.x = np.concatenate((self.x0,self.x1,self.x2,self.x3),axis=0)
        self.y = np.concatenate((self.y0,self.y1,self.y2,self.y3),axis=0)
        self.z = np.concatenate((self.z0,self.z1,self.z2,self.z3),axis=0)

    def color(self,zref):
        col = -1
        if  (zref==self.z0[0]): col = self.zcols[0]
        elif(zref==self.z1[0]): col = self.zcols[1]
        elif(zref==self.z2[0]): col = self.zcols[2]
        elif(zref==self.z3[0]): col = self.zcols[3]
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

    def set_function(self,name,z,k,thetamin,thetamax):
        ### rho = k*sin(theta) + z*cos(theta)
        func = ROOT.TF1(f"func_{name}","[1]*sin(x)+[0]*cos(x)",thetamin,thetamax,2)
        func.SetParameter(0,z)
        func.SetParameter(1,k)
        return func

    def get_waves_limits(self,name,z,k,thetamin,thetamax):
        fmin = +1e-20
        fmax = -1e-20
        for i in range(len(z)):
            func = self.set_function(name,z[i],k[i],thetamin,thetamax)
            if(func.GetMinimum()<fmin): fmin = func.GetMinimum()
            if(func.GetMaximum()>fmax): fmax = func.GetMaximum()
            del func
        return fmin,fmax
    
    def define_theta_rho_axes(self,name,tmin,tmax,fmin,fmax):
        h2 = ROOT.TH2D("h2waves_map_"+name,";#theta;#rho;",self.nbins_theta,tmin,tmax,self.nbins_rho,fmin,fmax)
        return h2
    
    def find_waves_intersect(self,k1,z1,k2,z2):
        dk = (k1-k2) if(abs(k1-k2)>self.xepsilon) else 1e15*np.sign(k1-k2)
        theta = math.atan2((z2-z1),dk) # the arc tangent of (y/x) in radians
        rho   = k1*math.sin(theta) + z1*math.cos(theta)
        return theta,rho
    
    def find_functions_intersect(self,f1,f2):
        name1 = f1.GetName()+"_flat"
        name2 = f2.GetName()+"_flat"
        flat1 = ROOT.TF1(name1,f"{f1.GetParameter(1)}*sin(x)+{f1.GetParameter(0)}*cos(x)",f1.GetXmin(),f1.GetXmax())
        flat2 = ROOT.TF1(name2,f"{f2.GetParameter(1)}*sin(x)+{f2.GetParameter(0)}*cos(x)",f2.GetXmin(),f2.GetXmax())
        diff  = ROOT.TF1(name1+"-"+name2,"abs("+name1+"-"+name2+")",f1.GetXmin(),f1.GetXmax())
        mindiff  = diff.GetMinimum()
        theta = diff.GetMinimumX()
        rho   = flat1.Eval(theta)
        del flat1,flat2,diff
        return mindiff,theta,rho

    def getbin(self,thetax,rhox,thetay,rhoy):
        bin_thetax = self.h2waves_zx.GetXaxis().FindBin(thetax) if(thetax>=self.thetamin_x and thetax<self.thetamax_x) else -1
        bin_rhox   = self.h2waves_zx.GetYaxis().FindBin(rhox)   if(rhox>=self.rhomin_x     and rhox<self.rhomax_x)     else -1 
        bin_thetay = self.h2waves_zy.GetXaxis().FindBin(thetay) if(thetay>=self.thetamin_y and thetay<self.thetamax_y) else -1
        bin_rhoy   = self.h2waves_zy.GetYaxis().FindBin(rhoy)   if(rhoy>=self.rhomin_y     and rhoy<self.rhomax_y)     else -1
        valid = (bin_thetax>=0 and bin_rhox>=0 and bin_thetay>=0 and bin_rhoy>=0)
        return valid,bin_thetax,bin_rhox,bin_thetay,bin_rhoy

    def fill_accumulator(self,brhox,bthetax,brhoy,bthetay):
        key = (brhox,bthetax,brhoy,bthetay)
        self.accumulator[key] = self.accumulator.get(key,0)+1
    
    def get_pair(self,CA,CB):
        thetax,rhox = self.find_waves_intersect(CA.xmm,CA.zmm,CB.xmm,CB.zmm)
        thetay,rhoy = self.find_waves_intersect(CA.ymm,CA.zmm,CB.ymm,CB.zmm)            
        valid,bthetax,brhox,bthetay,brhoy = self.getbin(thetax,rhox,thetay,rhoy)
        # print(f"thetax={thetax}, rhox={rhox}, thetay={thetay}, rhoy={rhoy}")
        if(valid): self.fill_accumulator(brhox,bthetax,brhoy,bthetay)
        self.h2waves_zx.Fill(thetax,rhox)
        self.h2waves_zy.Fill(thetay,rhoy)

    def fill_4d_wave_intersections(self,clusters):
        print(f"ievt={self.eventid}: Starting 0-1")
        for c0 in clusters["ALPIDE_0"]:
            for c1 in clusters["ALPIDE_1"]:
                self.get_pair(c0,c1)
        print(f"ievt={self.eventid}: Starting 0-2")
        for c0 in clusters["ALPIDE_0"]:
            for c2 in clusters["ALPIDE_2"]:
                self.get_pair(c0,c2)
        print(f"ievt={self.eventid}: Starting 0-3")
        for c0 in clusters["ALPIDE_0"]:
            for c3 in clusters["ALPIDE_3"]:
                self.get_pair(c0,c3)
        print(f"ievt={self.eventid}: Starting 1-2")
        for c1 in clusters["ALPIDE_1"]:
            for c2 in clusters["ALPIDE_2"]:
                self.get_pair(c1,c2)
        print(f"ievt={self.eventid}: Starting 1-3")
        for c1 in clusters["ALPIDE_1"]:
            for c3 in clusters["ALPIDE_3"]:
                self.get_pair(c1,c3)
        print(f"ievt={self.eventid}: Starting 2-3")
        for c2 in clusters["ALPIDE_2"]:
            for c3 in clusters["ALPIDE_3"]:
                self.get_pair(c2,c3)
    
    def search_in_neighbours(self,key):
        neigbours_vals = 0
        neighbours = [-1,0,+1]
        neigbours_avg = [0,0,0,0]
        for d0 in neighbours:
            for d1 in neighbours:
                for d2 in neighbours:
                    for d3 in neighbours:
                        nighbourkey = (key[0]+d0, key[1]+d1, key[2]+d2, key[3]+d3)
                        neigbours_val = self.accumulator.get(nighbourkey,0)
                        neigbours_vals += neigbours_val
                        if(neigbours_val>0):
                            neigbours_avg[0] += (key[0]+d0)*neigbours_val
                            neigbours_avg[1] += (key[1]+d1)*neigbours_val
                            neigbours_avg[2] += (key[2]+d2)*neigbours_val
                            neigbours_avg[3] += (key[3]+d3)*neigbours_val
        for i in range(len(neigbours_avg)):
            ### get the average
            bin_avg = int(round(float(neigbours_avg[i])/float(neigbours_vals)))
            ### correct if out of bounds
            if((i==0 or i==2) and bin_avg<1):                bin_avg = 1 ### rho x/y
            if((i==0 or i==2) and bin_avg>self.nbins_rho):   bin_avg = self.nbins_rho ### rho x/y
            if((i==1 or i==3) and bin_avg<1):                bin_avg = 1 ### theta x/y
            if((i==1 or i==3) and bin_avg>self.nbins_theta): bin_avg = self.nbins_theta ### theta x/y
            ### assign corrected value
            neigbours_avg[i] = bin_avg
        neigbours_avg_key = (neigbours_avg[0],neigbours_avg[1],neigbours_avg[2],neigbours_avg[3])
        return neigbours_vals, neigbours_avg_key
    
    # def get_seed_coordinates(self):
    #     cells = [x for x,y in self.accumulator.items() if(y>=self.minintersections)]
    #     print(f"cumulator size: {len(self.accumulator)}, good cells: {len(cells)}")
    #     return cells

    def get_seed_coordinates(self):
        cells = []
        for key,val in self.accumulator.items():
            ### all good: have all 6 intersections
            if(val>=self.minintersections):
                cells.append(key)
            ### if a bit too low: have only 4 intersectiosn
            if(cfg["seed_allow_neigbours"] and (val<self.minintersections and val>=(self.minintersections-2))):
                neigbours_vals,neigbours_avg_key = self.search_in_neighbours(key)
                if(neigbours_vals>=self.minintersections):
                    # cells.append(key)
                    cells.append(neigbours_avg_key)
                    if(key!=neigbours_avg_key): print(f"key={key} --> corrected key={neigbours_avg_key}")
            ### otherwise don't bother
        print(f"cumulator size: {len(self.accumulator)}, good cells: {len(cells)}")
        return cells
    
    def get_tunnels(self):
        tunnels = []
        hough_coord = []
        for icell,cell in enumerate(self.cells):
            brhox   = cell[0]
            bthetax = cell[1]
            brhoy   = cell[2]
            bthetay = cell[3]
            thetax = self.h2waves_zx.GetXaxis().GetBinCenter(bthetax)
            rhox   = self.h2waves_zx.GetYaxis().GetBinCenter(brhox)
            thetay = self.h2waves_zy.GetXaxis().GetBinCenter(bthetay)
            rhoy   = self.h2waves_zy.GetYaxis().GetBinCenter(brhoy)
            valid,tunnel = self.LUT.clusters_in_tunnel(thetax,rhox,thetay,rhoy)
            if(valid):
                tunnels.append( tunnel )
                hough_coord.append( (thetax,rhox,thetay,rhoy) )
            # print(f"Cell[{icell}]: valid?{valid} --> tunnel={tunnel}")
        return tunnels,hough_coord
    
    def set_seeds(self,clusters):
        tunnel_nsseds = [1]*len(self.tunnels)
        seeds = []
        det0 = cfg["detectors"][0]
        det1 = cfg["detectors"][1]
        det2 = cfg["detectors"][2]
        det3 = cfg["detectors"][3]
        for itnl,tunnel in enumerate(self.tunnels):
            candidate = []
            n0 = len(tunnel[det0])
            n1 = len(tunnel[det1])
            n2 = len(tunnel[det2])
            n3 = len(tunnel[det3])
            tunnel_nsseds[itnl] = n0*n1*n2*n3
            for c0 in tunnel[det0]:
                for c1 in tunnel[det1]:
                    for c2 in tunnel[det2]:
                        for c3 in tunnel[det3]:
                            seeds.append( [c0,c1,c2,c3] )
        return tunnel_nsseds, seeds
    
    def plot_seeder(self,name):
        ROOT.gErrorIgnoreLevel = ROOT.kError
        # ROOT.gErrorIgnoreLevel = ROOT.kWarning
        ROOT.gROOT.SetBatch(1)
        ROOT.gStyle.SetOptFit(0)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPadBottomMargin(0.15)
        ROOT.gStyle.SetPadLeftMargin(0.13)
        ROOT.gStyle.SetPadRightMargin(0.15)
        
        plotname = name.replace(".pdf","_hough_transform.pdf")
        
        gxy0 = self.graph("ALPIDE_0",self.x0,self.y0,ROOT.kRed,     [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )
        gxy1 = self.graph("ALPIDE_1",self.x1,self.y1,ROOT.kBlack,   [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )
        gxy2 = self.graph("ALPIDE_2",self.x2,self.y2,ROOT.kGreen-2, [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )
        gxy3 = self.graph("ALPIDE_3",self.x3,self.y3,ROOT.kMagenta, [-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2], [-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2] )

        cnv = ROOT.TCanvas("cnv_hits_xy","",1000,1000)
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
        cnv = ROOT.TCanvas("cnv_hits_vs_z","",1000,500)
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

        cnv = ROOT.TCanvas("cnv_transform","",1500,1200)
        cnv.Divide(2,2)
        cnv.cd(1)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        ROOT.gPad.SetTicks(1,1)
        first = True
        for name,f in self.zx_wave_functions.items():
            f.SetTitle("Hough transform in z-x;#theta;#rho")
            f.GetXaxis().SetTitle("#theta")
            f.GetYaxis().SetTitle("#rho")
            f.SetMinimum(self.zx_wave_fmin*1.2)
            f.SetMaximum(self.zx_wave_fmax*1.2)
            if(first):
                f.Draw("l")
                first = False
            else: f.Draw("l same")
            for arw in arrows_zx: arw.Draw()
        ROOT.gPad.RedrawAxis()
        cnv.cd(2)
        ROOT.gPad.SetTicks(1,1)
        ROOT.gPad.SetGridx()
        ROOT.gPad.SetGridy()
        first = True
        for name,f in self.zy_wave_functions.items():
            f.SetTitle("Hough transform in z-y")
            f.GetXaxis().SetTitle("#theta")
            f.GetYaxis().SetTitle("#rho")
            f.SetMinimum(self.zy_wave_fmin*1.2)
            f.SetMaximum(self.zy_wave_fmax*1.2)
            if(first):
                f.Draw("l")
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

        zx_seed_mg = self.multigraph("zx_seed","Post-seeding Telescope x vs z;z [mm];x [mm]",self.seed_z,self.seed_x,[self.zmin,self.zmax],[-self.npix_x*self.pix_x/2,+self.npix_x*self.pix_x/2])
        zy_seed_mg = self.multigraph("zy_seed","Post-seeding Telescope y vs z;z [mm];y [mm]",self.seed_z,self.seed_y,[self.zmin,self.zmax],[-self.npix_y*self.pix_y/2,+self.npix_y*self.pix_y/2])
        cnv = ROOT.TCanvas("cnv_frequency","",1000,500)
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

        ### important!!!
        self.clear_h2Freq()
        self.clear_functions()
