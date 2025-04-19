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
        ncls = 0
        for det in cfg["detectors"]: ncls += len(clusters[det])
        ncls = int(ncls/len(cfg["detectors"]))
        self.nbinsx = -1
        self.nbinsy = -1
        self.tunnel_width_x = 0.
        self.tunnel_width_y = 0.
        if(ncls<cfg["cls_mult_low"]):
            self.nbinsx = cfg["lut_nbinsx_low"]
            self.nbinsy = cfg["lut_nbinsy_low"]
            self.tunnel_width_x = cfg["lut_widthx_low"]
            self.tunnel_width_y = cfg["lut_widthy_low"]
        elif(ncls>=cfg["cls_mult_low"] and ncls<cfg["cls_mult_mid"]):
            self.nbinsx = cfg["lut_nbinsx_mid"]
            self.nbinsy = cfg["lut_nbinsy_mid"]
            self.tunnel_width_x = cfg["lut_widthx_mid"]
            self.tunnel_width_y = cfg["lut_widthy_mid"]
        elif(ncls>=cfg["cls_mult_mid"] and ncls<cfg["cls_mult_hgh"]):
            self.nbinsx = cfg["lut_nbinsx_hgh"]
            self.nbinsy = cfg["lut_nbinsy_hgh"]
            self.tunnel_width_x = cfg["lut_widthx_hgh"]
            self.tunnel_width_y = cfg["lut_widthy_hgh"]
        elif(ncls>=cfg["cls_mult_hgh"] and ncls<cfg["cls_mult_inf"]):
            self.nbinsx = cfg["lut_nbinsx_inf"]
            self.nbinsy = cfg["lut_nbinsy_inf"]
            self.tunnel_width_x = cfg["lut_widthx_inf"]
            self.tunnel_width_y = cfg["lut_widthy_inf"]
        else:
            sys.exit(f"In lookup_table ncls:ncls>cls_mult_inf, not implemented. exitting")
        
        xalgnmax,yalgnmax = self.find_alignment_bounds()
        self.chipXmin = -( (cfg["chipX"]+xalgnmax)*(1.+cfg["lut_scaleX"]) )/2.
        self.chipXmax = +( (cfg["chipX"]+xalgnmax)*(1.+cfg["lut_scaleX"]) )/2.
        self.chipYmin = -( (cfg["chipY"]+yalgnmax)*(1.+cfg["lut_scaleY"]) )/2.
        self.chipYmax = +( (cfg["chipY"]+yalgnmax)*(1.+cfg["lut_scaleY"]) )/2.
        # print(f"LUT: ncls={ncls}, xlim[{self.chipXmin:.3f},{self.chipXmax:.3f}], ylim[{self.chipYmin:.3f},{self.chipYmax:.3f}], nx={self.nbinsx}, ny={self.nbinsy}, tunnel_wx={self.tunnel_width_x}, tunnel_wy={self.tunnel_width_y}")
        ### call in the constructor:
        self.init_axs()
        self.init_lut()
        # self.fill_lut(clusters)
    
    # def __del__(self):
        # print(f"eventid={self.eventid}: deleted LookupTable class")

    def init_axs(self):
        for det in cfg["detectors"]:
            self.AXS.update({ det:ROOT.TH2D("lut_"+det+"_"+str(self.eventid),";x;y;Clusters",self.nbinsx,self.chipXmin,self.chipXmax, self.nbinsy,self.chipYmin,self.chipYmax) })
        
    def init_lut(self):
        for det in cfg["detectors"]:
            self.LUT.update({ det:{} })

    def find_alignment_bounds(self):
        xmax = 0
        ymax = 0
        for key1 in cfg["misalignment"]:
            for key2 in cfg["misalignment"][key1]:
                d = abs(cfg["misalignment"][key1][key2])
                xmax = d if(key2=="dx" and d>xmax) else xmax
                ymax = d if(key2=="dy" and d>ymax) else ymax
        # print(f"In lookup table: alignment modifier to x is {xmax} and to y is {ymax}")
        return xmax,ymax
    
    def fill_lut(self,clusters):
        for det in cfg["detectors"]:
            for clsidx,cluster in enumerate(clusters[det]):
                bx = self.AXS[det].GetXaxis().FindBin(cluster.xmm)
                by = self.AXS[det].GetYaxis().FindBin(cluster.ymm)
                # print(f"In LUT: eventid={self.eventid}  {det}  --> cluster x={cluster.xmm}, y={cluster.ymm}  -->  bx={bx}, by={by}")
                validx = (cluster.xmm>=self.chipXmin and cluster.xmm<self.chipXmax)
                validy = (cluster.ymm>=self.chipYmin and cluster.ymm<self.chipYmax)
                if(not validx or not validy):
                    print(f"in full_lut: validx={validx}, validy={validy} with x={cluster.xmm}, y={cluster.ymm}")
                    print("please increase the lut scale. quitting")
                    quit()
                axsbin = self.AXS[det].FindBin(cluster.xmm,cluster.ymm)
                # print(f"In LUT: eventid={self.eventid}  {det}  -->  axsbin={axsbin}")
                if(axsbin in self.LUT[det]): self.LUT[det][axsbin].append(clsidx)
                else:                        self.LUT[det].update( {axsbin:[clsidx]} )
    
    def remove_from_lut(self,det,x,y,clsidx):
        axsbin = self.AXS[det].FindBin(x,y)
        if(axsbin in self.LUT[det]): self.LUT[det][axsbin].remove(clsidx)
    
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
    
    # def clusters_in_tunnel(self,theta_x,rho_x,theta_y,rho_y):
    #     print(f"clusters_in_tunnel: eventid={self.eventid}  -->  theta_x={theta_x}, rho_x={rho_x}, theta_y={theta_y}, rho_y={rho_y}")
    #     tunnel = {}
    #     planes = 0
    #     AX,BX = self.get_par_lin(theta_x,rho_x)
    #     AY,BY = self.get_par_lin(theta_y,rho_y)
    #     print(f"clusters_in_tunnel: eventid={self.eventid}  -->  Ax={AX}, Bx={BX}, Ay={AY}, By={BY}")
    #     for det in cfg["detectors"]:
    #         zdet = cfg["rdetectors"][det][2]
    #         XX = self.k_of_z(zdet,AX,BX)
    #         YY = self.k_of_z(zdet,AY,BY)
    #         print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det} prediction: x={XX}, y={YY}, z={zdet}")
    #         xmin = XX-self.tunnel_width_x
    #         xmax = XX+self.tunnel_width_x
    #         ymin = YY-self.tunnel_width_y
    #         ymax = YY+self.tunnel_width_y
    #         xbinmin = self.AXS[det].GetXaxis().FindBin(xmin) if(xmin>=self.chipXmin) else 1
    #         xbinmax = self.AXS[det].GetXaxis().FindBin(xmax) if(xmax<self.chipXmax)  else self.nbinsx
    #         ybinmin = self.AXS[det].GetYaxis().FindBin(ymin) if(ymin>=self.chipYmin) else 1
    #         ybinmax = self.AXS[det].GetYaxis().FindBin(ymax) if(ymax<self.chipYmax)  else self.nbinsy
    #         ### add neighbour bins:
    #         if(cfg["seed_allow_neigbours"]):
    #             nNeigbourBinsX = 1 #if(abs(AX)>0.03) else int(5+2*cfg["detectors"].index(det))
    #             nNeigbourBinsY = 1 #if(abs(AY)>0.03) else int(5+2*cfg["detectors"].index(det))
    #             xbinmin = xbinmin-nNeigbourBinsX if(xbinmin-nNeigbourBinsX>0) else 1
    #             xbinmax = xbinmax+nNeigbourBinsX if(xbinmax+nNeigbourBinsX<self.nbinsx+1) else self.nbinsx
    #             ybinmin = ybinmin-nNeigbourBinsY if(ybinmin-nNeigbourBinsY>0) else 1
    #             ybinmax = ybinmax+nNeigbourBinsY if(ybinmax+nNeigbourBinsY<self.nbinsy+1) else self.nbinsy
    #         print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    #         print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det} xbinmin/max={xbinmin,xbinmax}, ybinmin/max={ybinmin,ybinmax}")
    #         clsidx_in_tnl = []
    #         for bx in range(xbinmin,xbinmax+1):
    #             for by in range(ybinmin,ybinmax+1):
    #                 axsbin = self.AXS[det].GetBin(bx,by)
    #                 # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}:  bx/y={bx,by}  axsbin={axsbin}")
    #                 if(axsbin in self.LUT[det]):
    #                     for c in self.LUT[det][axsbin]:
    #                         clsidx_in_tnl.append(c)
    #         tunnel.update( {det:clsidx_in_tnl} )
    #         print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}: tunnel={tunnel[det]}")
    #         planes += (len(clsidx_in_tnl)>0)
    #     valid = (planes==len(cfg["detectors"]))
    #     return valid,tunnel

    def get_edges_from_theta_rho_corners(self,det,theta_x,rho_x,theta_y,rho_y):
        xmin = +1e20
        xmax = -1e20
        ymin = +1e20
        ymax = -1e20
        for i in range(2):
            AX,BX = self.get_par_lin(theta_x[i],rho_x[i])
            AY,BY = self.get_par_lin(theta_y[i],rho_y[i])
            zdet = cfg["rdetectors"][det][2]
            XX = self.k_of_z(zdet,AX,BX)
            YY = self.k_of_z(zdet,AY,BY)
            # print(f"get_edges_from_theta_rho_corners cornere[i]: eventid={self.eventid}  -->  {det} prediction: x={XX}, y={YY}, z={zdet}")
            xmin = XX if(XX<xmin) else xmin
            xmax = XX if(XX>xmax) else xmax
            ymin = YY if(YY<ymin) else ymin
            ymax = YY if(YY>ymax) else ymax
        xmin = xmin-self.tunnel_width_x
        xmax = xmax+self.tunnel_width_x
        ymin = ymin-self.tunnel_width_y
        ymax = ymax+self.tunnel_width_y
        return xmin,xmax,ymin,ymax

    def clusters_in_tunnel(self,theta_x,rho_x,theta_y,rho_y):
        # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  theta_x={theta_x}, rho_x={rho_x}, theta_y={theta_y}, rho_y={rho_y}")
        tunnel = {}
        planes = 0
        for det in cfg["detectors"]:
            xmin,xmax,ymin,ymax = self.get_edges_from_theta_rho_corners(det,theta_x,rho_x,theta_y,rho_y)
            xbinmin = self.AXS[det].GetXaxis().FindBin(xmin) if(xmin>=self.chipXmin) else 1
            xbinmax = self.AXS[det].GetXaxis().FindBin(xmax) if(xmax<self.chipXmax)  else self.nbinsx
            ybinmin = self.AXS[det].GetYaxis().FindBin(ymin) if(ymin>=self.chipYmin) else 1
            ybinmax = self.AXS[det].GetYaxis().FindBin(ymax) if(ymax<self.chipYmax)  else self.nbinsy
            # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
            # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det} xbinmin/max={xbinmin,xbinmax}, ybinmin/max={ybinmin,ybinmax}")
            clsidx_in_tnl = []
            for bx in range(xbinmin,xbinmax+1):
                for by in range(ybinmin,ybinmax+1):
                    axsbin = self.AXS[det].GetBin(bx,by)
                    # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}:  bx/y={bx,by}  axsbin={axsbin}")
                    if(axsbin in self.LUT[det]):
                        for c in self.LUT[det][axsbin]:
                            clsidx_in_tnl.append(c)
            tunnel.update( {det:clsidx_in_tnl} )
            # print(f"clusters_in_tunnel: eventid={self.eventid}  -->  {det}: tunnel={tunnel[det]}")
            planes += (len(clsidx_in_tnl)>0)
        valid = (planes==len(cfg["detectors"]))
        return valid,tunnel
    
        
    def clear_all(self):
        del self.LUT
        del self.AXS
    