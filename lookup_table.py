#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT

import config
from config import *
import utils
from utils import *

class LookupTable:
    def __init__(self,clusters,eventid=0):
        self.eventid = eventid
        self.LUT = {}
        self.AXS = {}
        ncls = len(clusters)
        self.nbinsx = -1
        self.nbinsy = -1
        if(ncls<20):
            self.nbinsx = cfg["lut_nbinsx_020"]
            self.nbinsy = cfg["lut_nbinsy_020"]
        elif(ncls>=20 and ncls<200):
            self.nbinsx = cfg["lut_nbinsx_200"]
            self.nbinsy = cfg["lut_nbinsy_200"]
        else:
            self.nbinsx = cfg["lut_nbinsx_inf"]
            self.nbinsy = cfg["lut_nbinsy_inf"]
        self.chipXmin = -( cfg["chipX"]*(1.+cfg["lut_scaleX"]) )/2.
        self.chipXmax = +( cfg["chipX"]*(1.+cfg["lut_scaleX"]) )/2.
        self.chipYmin = -( cfg["chipY"]*(1.+cfg["lut_scaleY"]) )/2.
        self.chipYmax = +( cfg["chipY"]*(1.+cfg["lut_scaleY"]) )/2.
        # print(f"xlim: [{self.chipXmin},{self.chipXmax}], ylim: [{self.chipYmin},{self.chipYmax}]")
        ### call in the constructor:
        self.init_axs()
        self.init_lut()
        self.fill_lut(clusters)

    def init_axs(self):
        for det in cfg["detectors"]:
            self.AXS.update({ det:ROOT.TH2D("lut_"+det+"_"+str(self.eventid),";x;y;Clusters",self.nbinsx,self.chipXmin,self.chipXmax, self.nbinsy,self.chipYmin,self.chipYmax) })
        
    def init_lut(self):
        for det in cfg["detectors"]:
            self.LUT.update({ det:{} })
            for bx in range(1,self.AXS[det].GetNbinsX()+1):
                for by in range(1,self.AXS[det].GetNbinsY()+1):
                    axsbin = self.AXS[det].GetBin(bx,by)
                    self.LUT[det].update( {axsbin:[]} )
    
    def fill_lut(self,clusters):
        for det in cfg["detectors"]:
            for clsidx,cluster in enumerate(clusters[det]):
                validx = (cluster.xmm>=self.chipXmin and cluster.xmm<self.chipXmax)
                validy = (cluster.ymm>=self.chipYmin and cluster.ymm<self.chipYmax)
                if(not validx or not validy):
                    print(f"in full_lut: validx={validx}, validy={validy} with x={cluster.xmm}, y={cluster.ymm}")
                    print("please increase the lut scale. quitting")
                    quit()
                # print(f"{det}: cluster x={cluster.xmm}, y={cluster.ymm}")
                axsbin = self.AXS[det].FindBin(cluster.xmm,cluster.ymm)
                self.LUT[det][axsbin].append(clsidx)
    
    def remove_from_lut(self,det,x,y,clsidx):
        axsbin = self.AXS[det].FindBin(x,y)
        self.LUT[det][axsbin].remove(clsidx)
    
    def get_par_lin(self,theta_k,rho_k): ### theta and rho from Hough transform
        if(math.sin(theta_k)==0):
            print(f"in get_par_lin, sin(theta)=0: quitting.")
            quit()
        if(math.tan(theta_k)==0):
            print(f"in get_par_lin, 1/tan(theta)=0: quitting.")
            quit()
        AK = -1./math.tan(theta_k)
        BK = rho_k/math.sin(theta_k)
        # print(f"theta_k={theta_k}, rho_k={rho_k} --> AK={AK}, BK={BK}")
        return AK,BK
    
    def k_of_z(self,z,AK,BK):
        k = AK*z + BK
        # print(f"AK={AK}, BK={BK}, z={z} --> k={k}")
        return k
    
    def clusters_in_tunnel(self,theta_x,rho_x,theta_y,rho_y):
        # print(f"From bin centers: theta_x={theta_x}, rho_x={rho_x}, theta_y={theta_y}, rho_y={rho_y}")
        tunnel = {}
        planes = 0
        AX,BX = self.get_par_lin(theta_x,rho_x)
        AY,BY = self.get_par_lin(theta_y,rho_y)
        # print(f"Ax={AX}, BX={BX}, Ax={AY}, BY={BY}")
        for det in cfg["detectors"]:
            zdet = cfg["rdetectors"][det][2]
            XX = self.k_of_z(zdet,AX,BX)
            YY = self.k_of_z(zdet,AY,BY)
            # print(f"{det} prediction: x={XX}, y={YY}, z={zdet}")
            xmin = XX-cfg["lut_widthx"]
            xmax = XX+cfg["lut_widthx"]
            ymin = YY-cfg["lut_widthy"]
            ymax = YY+cfg["lut_widthy"]
            xbinmin = self.AXS[det].GetXaxis().FindBin(xmin) if(xmin>=self.chipXmin) else 1
            xbinmax = self.AXS[det].GetXaxis().FindBin(xmax) if(xmax<self.chipXmax)  else self.AXS[det].GetNbinsX()
            ybinmin = self.AXS[det].GetYaxis().FindBin(ymin) if(ymin>=self.chipYmin) else 1
            ybinmax = self.AXS[det].GetYaxis().FindBin(ymax) if(ymax<self.chipYmax)  else self.AXS[det].GetNbinsY()
            ### add neighbour bins:
            if(cfg["seed_allow_neigbours"]):
                xbinmin = xbinmin-1 if(xbinmin>1) else xbinmin
                xbinmax = xbinmax+1 if(xbinmax<self.AXS[det].GetNbinsX()) else xbinmax
                ybinmin = ybinmin-1 if(ybinmin>1) else ybinmin
                ybinmax = ybinmax+1 if(ybinmax<self.AXS[det].GetNbinsY()) else ybinmax
            # print(f"{det}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
            clsidx_in_tnl = []
            for bx in range(xbinmin,xbinmax+1):
                for by in range(ybinmin,ybinmax+1):
                    axsbin = self.AXS[det].GetBin(bx,by)
                    for c in self.LUT[det][axsbin]:
                        clsidx_in_tnl.append(c)
            tunnel.update( {det:clsidx_in_tnl} )
            planes += (len(clsidx_in_tnl)>0)
        valid = (planes==len(cfg["detectors"]))
        return valid,tunnel
        
    def clear_all(self):
        del self.LUT
        del self.AXS
    