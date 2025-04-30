#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
from collections import defaultdict
import ROOT

import config
from config import *
import objects
from objects import *
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
        self.is5lyr = (len(cfg["detectors"])>4)
        nclusters = 0
        for det in cfg["detectors"]: nclusters += len(clusters[det])
        nclusters = int(nclusters/len(cfg["detectors"]))
        ### the clusters per detector
        n0 = len(clusters[cfg["detectors"][0]])
        n1 = len(clusters[cfg["detectors"][1]])
        n2 = len(clusters[cfg["detectors"][2]])
        n3 = len(clusters[cfg["detectors"][3]])
        n4 = len(clusters[cfg["detectors"][4]]) if(self.is5lyr) else 0
        self.x0 = np.zeros(n0)
        self.x1 = np.zeros(n1)
        self.x2 = np.zeros(n2)
        self.x3 = np.zeros(n3)
        self.x4 = np.zeros(n4) if(self.is5lyr) else None
        self.y0 = np.zeros(n0)
        self.y1 = np.zeros(n1)
        self.y2 = np.zeros(n2)
        self.y3 = np.zeros(n3)
        self.y4 = np.zeros(n4) if(self.is5lyr) else None
        self.z0 = np.zeros(n0)
        self.z1 = np.zeros(n1)
        self.z2 = np.zeros(n2)
        self.z3 = np.zeros(n3)
        self.z4 = np.zeros(n4) if(self.is5lyr) else None
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
        self.theta_x_scale = 1
        self.rho_x_scale   = 1
        self.theta_y_scale = 1
        self.rho_y_scale   = 1
        if(nclusters<=cfg["cls_mult_low"]):
            self.theta_x_scale = cfg["seed_thetax_scale_low"]
            self.rho_x_scale   = cfg["seed_rhox_scale_low"] 
            self.theta_y_scale = cfg["seed_thetay_scale_low"]
            self.rho_y_scale   = cfg["seed_rhoy_scale_low"]
        elif(nclusters>cfg["cls_mult_low"]  and nclusters<=cfg["cls_mult_mid"]):
            self.theta_x_scale = cfg["seed_thetax_scale_mid"]
            self.rho_x_scale   = cfg["seed_rhox_scale_mid"] 
            self.theta_y_scale = cfg["seed_thetay_scale_mid"]
            self.rho_y_scale   = cfg["seed_rhoy_scale_mid"]
        elif(nclusters>cfg["cls_mult_mid"] and nclusters<=cfg["cls_mult_hgh"]):
            self.theta_x_scale = cfg["seed_thetax_scale_hgh"]
            self.rho_x_scale   = cfg["seed_rhox_scale_hgh"]
            self.theta_y_scale = cfg["seed_thetay_scale_hgh"]
            self.rho_y_scale   = cfg["seed_rhoy_scale_hgh"]
        elif(nclusters>cfg["cls_mult_hgh"] and nclusters<=cfg["cls_mult_inf"]):
            self.theta_x_scale = cfg["seed_thetax_scale_inf"]
            self.rho_x_scale   = cfg["seed_rhox_scale_inf"] 
            self.theta_y_scale = cfg["seed_thetay_scale_inf"]
            self.rho_y_scale   = cfg["seed_rhoy_scale_inf"]
        else: 
            sys.exit(f"In hough_seeder nclusters:{nclusters}>cls_mult_inf, not implemented. exitting")
        self.thetamin_x = np.pi/2-self.theta_x_scale*np.pi/2.
        self.thetamax_x = np.pi/2+self.theta_x_scale*np.pi/2.
        self.thetamin_y = np.pi/2-self.theta_y_scale*np.pi/2.
        self.thetamax_y = np.pi/2+self.theta_y_scale*np.pi/2.
        self.nbins_thetarho = -1
        if(nclusters<=cfg["cls_mult_low"]):                                     self.nbins_thetarho = cfg["seed_nbins_thetarho_low"]
        elif(nclusters>cfg["cls_mult_low"] and nclusters<=cfg["cls_mult_mid"]): self.nbins_thetarho = cfg["seed_nbins_thetarho_mid"]
        elif(nclusters>cfg["cls_mult_mid"] and nclusters<=cfg["cls_mult_hgh"]): self.nbins_thetarho = cfg["seed_nbins_thetarho_hgh"]
        elif(nclusters>cfg["cls_mult_hgh"] and nclusters<=cfg["cls_mult_inf"]): self.nbins_thetarho = cfg["seed_nbins_thetarho_inf"]
        else:
            sys.exit(f"In hough_seeder nclusters:{nclusters}>cls_mult_inf, not implemented. exitting")
        self.minintersections = math.comb(len(cfg["detectors"]),2) ### all pairs out of for detectors w/o repetitions
        self.nmissintersections = cfg["seed_nmiss_neigbours"] ## how many intersectians we are allowed to miss before searching in the neighbouring cells
        # self.neighbourslist = [ i for i in range(-cfg["seed_nmax_neigbours"],cfg["seed_nmax_neigbours"]+1) if(i!=0) ] ### this will be e.g. [-3,-2,-1,+1,+2,+3] if seed_nmax_neigbours=3
        self.neighbourslist = [ i for i in range(-cfg["seed_nmax_neigbours"],cfg["seed_nmax_neigbours"]+1) ] ### this will be e.g. [-3,-2,-1,0,+1,+2,+3] if seed_nmax_neigbours=3
        ### set the clusters
        self.set_clusters(clusters)
        self.zmin = self.z0[0]
        self.zmax = self.z4[0] if(self.is5lyr) else self.z3[0]
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
        ### allow only positive y-z seeds:
        self.LUT = LookupTable(clusters,eventid)
        ### the data structure
        self.accumulator = []
        ### accumulator = [0-1{key:val}, 0-2{key:val}, 0-3{key:val}, 0-4{key:val}, 1-2{key:val}, 1-3{key:val}, 1-4{key:val}, 2-3{key:val}, 2-4{key:val}, 3-4{key:val}]
        ### key   = ecoded(brhox,bthetax,brhoy,bthetay)
        ### value = number of times the 4D key in theta-rho-x/y appears
        for ncomb in range(self.minintersections): self.accumulator.append({})
        self.naccumulators = len(self.accumulator)
        # print(f"naccumulators={self.naccumulators}")
        ### fill the accumulator
        self.fill_4d_wave_intersections(clusters)
        ### get the 4D bin numbers of the good coordinates
        self.cells = self.get_seed_coordinates()
        # print(f"cells={self.cells}")
        ######################
        ##### cleanup!!! #####
        del self.accumulator
        ######################
        ### check the accumulator against the LookupTable
        # self.LUT = LookupTable(clusters,eventid)
        self.LUT.fill_lut(clusters)
        self.tunnels,self.hough_coords,self.hough_bounds,self.hough_space = self.get_tunnels()
        self.tunnel_nsseds, self.tnlid, self.coord, self.seeds = self.set_seeds(clusters)
        self.nseeds = len(self.seeds)
        ######################
        ##### cleanup!!! #####
        del self.h2waves_zx
        del self.h2waves_zy
        self.LUT.clear_all()
        del self.LUT
        ######################
        minSeedsPerTnl = min(self.tunnel_nsseds) if(len(self.tunnel_nsseds)>0)     else -1
        maxSeedsPerTnl = max(self.tunnel_nsseds) if(len(self.tunnel_nsseds)>0)     else -1
        avgSeedsPerTnl = np.mean(self.tunnel_nsseds) if(len(self.tunnel_nsseds)>0) else -1
        stdSeedsPerTnl = np.std(self.tunnel_nsseds)  if(len(self.tunnel_nsseds)>0) else -1
        print(f"eventid={self.eventid}: got {len(self.tunnels)} valid tunnels out of {len(self.cells)} tunnels and a total of {len(self.seeds)} seeds. N seeds per tunnel: min={minSeedsPerTnl}, max={maxSeedsPerTnl}, mean={avgSeedsPerTnl:.3f}+/-{stdSeedsPerTnl:.3f}.")
        
    # def __del__(self):
        # print(f"eventid={self.eventid}: deleted HoughSeeder class")

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
                if(det=="ALPIDE_4" and self.is5lyr):
                    self.x4[i] = c.xmm
                    self.y4[i] = c.ymm
                    self.z4[i] = c.zmm
        self.x = np.concatenate((self.x0,self.x1,self.x2,self.x3,self.x4),axis=0) if(self.is5lyr) else np.concatenate((self.x0,self.x1,self.x2,self.x3),axis=0)
        self.y = np.concatenate((self.y0,self.y1,self.y2,self.y3,self.y4),axis=0) if(self.is5lyr) else np.concatenate((self.y0,self.y1,self.y2,self.y3),axis=0)
        self.z = np.concatenate((self.z0,self.z1,self.z2,self.z3,self.z4),axis=0) if(self.is5lyr) else np.concatenate((self.z0,self.z1,self.z2,self.z3),axis=0)

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
        h2 = ROOT.TH2D("h2waves_map_"+name,";#theta;#rho;",self.nbins_thetarho,tmin,tmax,self.nbins_thetarho,fmin,fmax)
        h2.SetDirectory(0)
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
        mindiff = diff.GetMinimum()
        theta = diff.GetMinimumX()
        rho   = flat1.Eval(theta)
        del flat1,flat2,diff
        return mindiff,theta,rho

    def get_detpair(self,CA,CB):
        if(self.is5lyr):
            if(CA.DID==0 and CB.DID==1): return 0
            if(CA.DID==0 and CB.DID==2): return 1
            if(CA.DID==0 and CB.DID==3): return 2
            if(CA.DID==0 and CB.DID==4): return 3
            if(CA.DID==1 and CB.DID==2): return 4
            if(CA.DID==1 and CB.DID==3): return 5
            if(CA.DID==1 and CB.DID==4): return 6
            if(CA.DID==2 and CB.DID==3): return 7
            if(CA.DID==2 and CB.DID==4): return 8
            if(CA.DID==3 and CB.DID==4): return 9
        else:
            if(CA.DID==0 and CB.DID==1): return 0
            if(CA.DID==0 and CB.DID==2): return 1
            if(CA.DID==0 and CB.DID==3): return 2
            if(CA.DID==1 and CB.DID==2): return 3
            if(CA.DID==1 and CB.DID==3): return 4
            if(CA.DID==2 and CB.DID==3): return 5
        
        print(f"unknown combination for CA.DID={CA.DID} and CB.DID={CB.DID} - quitting.")
        quit()
        return -1

    def encode_key(self,brhox, bthetax, brhoy, bthetay):
        return (brhox * self.nbins_thetarho**3 + bthetax * self.nbins_thetarho**2 + brhoy * self.nbins_thetarho + bthetay)

    def decode_key(self,encoded_key):
        bthetay = encoded_key % self.nbins_thetarho
        encoded_key //= self.nbins_thetarho
        brhoy = encoded_key % self.nbins_thetarho
        encoded_key //= self.nbins_thetarho
        bthetax = encoded_key % self.nbins_thetarho
        encoded_key //= self.nbins_thetarho
        brhox = encoded_key
        return (brhox, bthetax, brhoy, bthetay)

    def getbin(self,thetax,rhox,thetay,rhoy):
        bin_thetax = self.h2waves_zx.GetXaxis().FindBin(thetax) if(thetax>=self.thetamin_x and thetax<self.thetamax_x) else -1
        bin_rhox   = self.h2waves_zx.GetYaxis().FindBin(rhox)   if(rhox>=self.rhomin_x     and rhox<self.rhomax_x)     else -1 
        bin_thetay = self.h2waves_zy.GetXaxis().FindBin(thetay) if(thetay>=self.thetamin_y and thetay<self.thetamax_y) else -1
        bin_rhoy   = self.h2waves_zy.GetYaxis().FindBin(rhoy)   if(rhoy>=self.rhomin_y     and rhoy<self.rhomax_y)     else -1
        valid = (bin_thetax>=0 and bin_rhox>=0 and bin_thetay>=0 and bin_rhoy>=0)
        return valid,bin_thetax,bin_rhox,bin_thetay,bin_rhoy

    def fill_accumulator(self,bdetpair,brhox,bthetax,brhoy,bthetay):
        key = self.encode_key(brhox,bthetax,brhoy,bthetay)
        self.accumulator[bdetpair][key] = self.accumulator[bdetpair].get(key,0)+1
    
    def get_pair(self,CA,CB):
        thetax,rhox = self.find_waves_intersect(CA.xmm,CA.zmm,CB.xmm,CB.zmm)
        thetay,rhoy = self.find_waves_intersect(CA.ymm,CA.zmm,CB.ymm,CB.zmm)
        valid,bthetax,brhox,bthetay,brhoy = self.getbin(thetax,rhox,thetay,rhoy)
        if(not cfg["seed_allow_negative_vertical_inclination"]):
            AX,BX = self.LUT.get_par_lin(thetax,rhox)
            if(AX<0.): return
        detpair = self.get_detpair(CA,CB)
        # print(f"eventid={self.eventid}  detpair={detpair}  valid={valid}  -->  bthetax={bthetax}, brhox={brhox}, bthetay={bthetay}, brhoy={brhoy}")
        # print(f"detpair={detpair}: thetax={thetax}, rhox={rhox}, thetay={thetay}, rhoy={rhoy}")
        if(valid): self.fill_accumulator(detpair,brhox,bthetax,brhoy,bthetay)
        self.h2waves_zx.Fill(thetax,rhox)
        self.h2waves_zy.Fill(thetay,rhoy)

    def fill_4d_wave_intersections(self,clusters):
        # print(f"ievt={self.eventid}: Starting pair search")
        for c0 in clusters["ALPIDE_0"]:
            for c1 in clusters["ALPIDE_1"]:
                self.get_pair(c0,c1)
        for c0 in clusters["ALPIDE_0"]:
            for c2 in clusters["ALPIDE_2"]:
                self.get_pair(c0,c2)
        for c0 in clusters["ALPIDE_0"]:
            for c3 in clusters["ALPIDE_3"]:
                self.get_pair(c0,c3)
        for c1 in clusters["ALPIDE_1"]:
            for c2 in clusters["ALPIDE_2"]:
                self.get_pair(c1,c2)
        for c1 in clusters["ALPIDE_1"]:
            for c3 in clusters["ALPIDE_3"]:
                self.get_pair(c1,c3)
        for c2 in clusters["ALPIDE_2"]:
            for c3 in clusters["ALPIDE_3"]:
                self.get_pair(c2,c3)
        if(self.is5lyr):
            for c0 in clusters["ALPIDE_0"]:
                for c4 in clusters["ALPIDE_4"]:
                    self.get_pair(c0,c4)
            for c1 in clusters["ALPIDE_1"]:
                for c4 in clusters["ALPIDE_4"]:
                    self.get_pair(c1,c4)
            for c2 in clusters["ALPIDE_2"]:
                for c4 in clusters["ALPIDE_4"]:
                    self.get_pair(c2,c4)
            for c3 in clusters["ALPIDE_3"]:
                for c4 in clusters["ALPIDE_4"]:
                    self.get_pair(c3,c4)
        # print(f"ievt={self.eventid}: Finished pair search")
        
    
    def search_in_neighbours(self,encoded_key):
        neigbours_vals = 0
        # neighbours for example: [-5,-4,-3,-2,-1,0,+1,+2,+3,+4,+5]
        key = self.decode_key(encoded_key)
        # print(f"in search_in_neighbours: key={key}")
        ### d0,d1,d2,,d3 are the brhox,bthetax,brhoy,bthetay
        for d0 in self.neighbourslist:
            for d1 in self.neighbourslist:
                for d2 in self.neighbourslist:
                    for d3 in self.neighbourslist:
                        if(d0==0 and d1==0 and d2==0 and d3==0): continue
                        nighbourkey = self.encode_key(key[0]+d0, key[1]+d1, key[2]+d2, key[3]+d3)
                        # print(f"d0={d0}, d1={d1}, d2={d2}, d3={d3} --> nighbourkey={nighbourkey} --> decodednegkey={ self.decode_key(nighbourkey) }")
                        for detpair in range(self.naccumulators): ### loop over all detector-pairs
                            neigbours_vals += (self.accumulator[detpair].get(nighbourkey,0)>0)
        return neigbours_vals

    def get_seed_coordinates(self):
        cells = []
        
        ### accumulator = [0-1{key:val}, 0-2{key:val}, 0-3{key:val}, 0-4{key:val}, 1-2{key:val}, 1-3{key:val}, 1-4{key:val}, 2-3{key:val}, 2-4{key:val}, 3-4{key:val}]
        ### key   = ecoded(brhox,bthetax,brhoy,bthetay)
        ### value = number of times the 4D key in theta-rho-x/y appears
        
        # print(f"accumulator: {self.accumulator}")
        
        ### check the index with the most occurances
        index_of_most_frequent_key = -1
        # First pass: count occurrences
        key_counts = defaultdict(int)
        for d in self.accumulator:
            for key in d:  # only one key per dict
                key_counts[key] += 1
        # Find the key with the highest count
        most_common_key = max(key_counts, key=key_counts.get)
        # Second pass: find first index of most common key
        for idx, d in enumerate(self.accumulator):
            if most_common_key in d:
                index_of_most_frequent_key = idx
                break
        
        for key,val in self.accumulator[index_of_most_frequent_key].items(): ### start by looping on all keys of the detector pair with the most repetitions
            nintersections = (val>0)
            # print(f"key={key}, val={val} --> nintersections={nintersections}")
            
            for detpair in range(1,self.naccumulators):
                nintersections += (self.accumulator[detpair].get(key,0)>0)
                # print(f"detpair={detpair}: nintersections={nintersections}")
            # print(f"Final: nintersections={nintersections}, self.minintersections={self.minintersections}")
            if(nintersections>=self.minintersections):
                cells.append(key)
            
            ### if too low:
            if(cfg["seed_allow_neigbours"] and (nintersections<self.minintersections and nintersections>=(self.minintersections-self.nmissintersections))):
                # print(f"Trying to recover: ")
                nintersections += self.search_in_neighbours(key)
                if(nintersections>=self.minintersections):
                    cells.append(key)
            # print(f"Final nintersections={nintersections}")
            ### otherwise don't bother
        # print(f"cumulator sizes: {len(self.accumulator[0]),len(self.accumulator[1]),len(self.accumulator[2]),len(self.accumulator[3]),len(self.accumulator[4]),len(self.accumulator[5]),len(self.accumulator[6]),len(self.accumulator[7]),len(self.accumulator[8]),len(self.accumulator[9])}, good cells: {len(cells)}")
        return cells
    
    
    def get_tunnels(self):
        # print(f"in get tunnels with {len(self.cells)}"))
        tunnels      = []
        hough_coords = []
        hough_bounds = []
        hough_space  = {
            "zx_xbins":self.h2waves_zx.GetNbinsX(), "zx_xmin":self.h2waves_zx.GetXaxis().GetXmin(), "zx_xmax":self.h2waves_zx.GetXaxis().GetXmax(),
            "zx_ybins":self.h2waves_zx.GetNbinsY(), "zx_ymin":self.h2waves_zx.GetYaxis().GetXmin(), "zx_ymax":self.h2waves_zx.GetYaxis().GetXmax(),
            "zy_xbins":self.h2waves_zy.GetNbinsX(), "zy_xmin":self.h2waves_zy.GetXaxis().GetXmin(), "zy_xmax":self.h2waves_zy.GetXaxis().GetXmax(),
            "zy_ybins":self.h2waves_zy.GetNbinsY(), "zy_ymin":self.h2waves_zy.GetYaxis().GetXmin(), "zy_ymax":self.h2waves_zy.GetYaxis().GetXmax()
        }
        
        for icell,cell in enumerate(self.cells):
            (brhox,bthetax,brhoy,bthetay) = self.decode_key(cell)
            
            central_thetax = self.h2waves_zx.GetXaxis().GetBinCenter(bthetax)
            central_rhox   = self.h2waves_zx.GetYaxis().GetBinCenter(brhox) 
            central_thetay = self.h2waves_zy.GetXaxis().GetBinCenter(bthetay)
            central_rhoy   = self.h2waves_zy.GetYaxis().GetBinCenter(brhoy)
            
            thetax = [ self.h2waves_zx.GetXaxis().GetBinLowEdge(bthetax), self.h2waves_zx.GetXaxis().GetBinUpEdge(bthetax) ]
            rhox   = [ self.h2waves_zx.GetYaxis().GetBinLowEdge(brhox),   self.h2waves_zx.GetYaxis().GetBinUpEdge(brhox)   ]
            thetay = [ self.h2waves_zy.GetXaxis().GetBinLowEdge(bthetay), self.h2waves_zy.GetXaxis().GetBinUpEdge(bthetay) ]
            rhoy   = [ self.h2waves_zy.GetYaxis().GetBinLowEdge(brhoy),   self.h2waves_zy.GetYaxis().GetBinUpEdge(brhoy)   ]
            
            valid,tunnel = self.LUT.clusters_in_tunnel(thetax,rhox,thetay,rhoy)
            
            if(valid):
                tunnels.append( tunnel )
                hough_coords.append( (central_thetax,central_rhox,central_thetay,central_rhoy) )
                hough_bounds.append( (thetax,rhox,thetay,rhoy) )
            # print(f"Cell[{icell}]: valid?{valid} --> tunnel={tunnel}")
        return tunnels,hough_coords,hough_bounds,hough_space
    
    def set_seeds(self,clusters):
        tunnel_nsseds = [1]*len(self.tunnels)
        seeds = []
        tnlid = []
        coord = []
        det0 = cfg["detectors"][0]
        det1 = cfg["detectors"][1]
        det2 = cfg["detectors"][2]
        det3 = cfg["detectors"][3]
        det4 = cfg["detectors"][4] if(self.is5lyr) else ""
        for itnl,tunnel in enumerate(self.tunnels):
            candidate = []
            n0 = len(tunnel[det0])
            n1 = len(tunnel[det1])
            n2 = len(tunnel[det2])
            n3 = len(tunnel[det3])
            n4 = len(tunnel[det4]) if(self.is5lyr) else 0
            tunnel_nsseds[itnl] = n0*n1*n2*n3*n4 if(self.is5lyr) else n0*n1*n2*n3
            for c0 in tunnel[det0]:
                for c1 in tunnel[det1]:
                    for c2 in tunnel[det2]:
                        for c3 in tunnel[det3]:
                            if(self.is5lyr):
                                for c4 in tunnel[det4]:
                                    seeds.append( [c0,c1,c2,c3,c4] )
                                    tnlid.append( itnl )
                                    coord.append( self.hough_coords[itnl] )
                            else:
                                seeds.append( [c0,c1,c2,c3] )
                                tnlid.append( itnl )
                                coord.append( self.hough_coords[itnl] )
        return tunnel_nsseds,tnlid,coord,seeds
