#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit

import argparse
parser = argparse.ArgumentParser(description='gun.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
argus = parser.parse_args()
configfile = argus.conf

import config
from config import *
### must be called here (first) and only once!
# init_config(configfile,False)

import objects
from objects import *
import utils
from utils import *


class FakeParticle:
    def __init__(self,vtx,slp,itp,orgpts,smrpts,msalgnpts):
        self.vtx       = vtx
        self.slp       = slp
        self.itp       = itp
        self.orgpts    = orgpts
        self.smrpts    = smrpts
        self.msalgnpts = msalgnpts

    # def __del__(self):
        # print(f"deleting")

    def __str__(self):
        return f"FakeParticle"


class ParticleGun:
    def __init__(self,vtxsurf,slopes,xyres=0):
        self.rnd = ROOT.TRandom()
        self.rnd.SetSeed()
        self.vtxsurf = vtxsurf
        self.slopes  = slopes
        self.xyres   = xyres
        self.layers  = getChips()
        self.misalgn = {}

    # def __del__(self):
        # print(f"deleting")

    def __str__(self):
        return f"ParticleGun"

    def set_misalgnment(self,misalignment):
        print(f"Fake misalignment in lab frame: {misalignment}")
        self.misalgn = misalignment

    def get_slopes(self):
        xz = self.rnd.Uniform(self.slopes["xz"][0],self.slopes["xz"][1])
        yz = self.rnd.Uniform(self.slopes["yz"][0],self.slopes["yz"][1])
        return [xz,yz]
    
    def get_vertex(self):
        vz = self.vtxsurf["z"]
        Rx = (self.vtxsurf["x"][1]-self.vtxsurf["x"][0])/2.
        Ry = (self.vtxsurf["y"][1]-self.vtxsurf["y"][0])/2.
        ctrx = self.vtxsurf["x"][0] + Rx
        ctry = self.vtxsurf["y"][0] + Ry
        vx = ctrx+self.rnd.Gaus(0,Rx/2)
        vy = ctry+self.rnd.Gaus(-1*vx,Ry/4)
        return [vx,vy,vz]
        
    def get_intercepts(self,vertex,slopes):
        bxz    = vertex[0]-slopes[0]*vertex[2]
        byz    = vertex[1]-slopes[1]*vertex[2]
        itcpts = [bxz,byz]
        return itcpts
    
    def k_of_z(self,z,slope,intercept):
        k = slope*z+intercept
        return k
    
    def produce(self):
        vertex = self.get_vertex()
        slopes = self.get_slopes()
        itcpts = self.get_intercepts(vertex,slopes)
        return vertex,slopes,itcpts
    
    def propagate(self,vertex,slopes,itcpts):
        axz = slopes[0]
        ayz = slopes[1]
        vx  = vertex[0]
        vy  = vertex[1]
        vz  = vertex[2]
        bxz = itcpts[0]
        byz = itcpts[1]
        points = {}
        for det in cfg["detectors"]:
            z = cfg["rdetectors"][det][2]+cfg["zOffset"]
            x = self.k_of_z(z,axz,bxz)
            y = self.k_of_z(z,ayz,byz)
            points.update({det:[x,y,z]})
        return points
            
    def smear(self,points):
        smeared = {}
        for det in points:
            x = points[det][0]+self.rnd.Gaus(0,self.xyres) if(self.xyres>0) else points[det][0]
            y = points[det][1]+self.rnd.Gaus(0,self.xyres) if(self.xyres>0) else points[det][1]
            z = points[det][2]
            smeared.update({det:[x,y,z]})
        return smeared
    
    def misalign(self,smrpts):
        msalgnpts = {}
        for det in smrpts:
            x = smrpts[det][0]
            y = smrpts[det][1]
            z = smrpts[det][2]
            ### apply misalignment:
            x,y = rotate(self.misalgn[det]["theta"],x,y)
            x = x+self.misalgn[det]["dx"]
            y = y+self.misalgn[det]["dy"]
            msalgnpts.update({det:[x,y,z]})
        return msalgnpts
    
    def generate(self):
        vtx,slp,itp = self.produce()
        orgpts  = self.propagate(vtx,slp,itp)
        smrpts  = self.smear(orgpts)
        msalgnpts = self.misalign(smrpts)
        fkprt = FakeParticle(vtx,slp,itp,orgpts,smrpts,msalgnpts)
        return fkprt
        
    def accept(self,fkprt):
        for det in cfg["detectors"]:
            idet = cfg["detectors"].index(det)
            xmin = self.layers[idet][0][0]
            ymin = self.layers[idet][0][1]
            xmax = self.layers[idet][2][0]
            ymax = self.layers[idet][2][1]
            x = fkprt.smrpts[det][0] 
            y = fkprt.smrpts[det][1] 
            if(x<xmin or x>xmax): return False
            if(y<ymin or y>ymax): return False
        return True
    
        
if __name__ == "__main__":
    
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
    
    # print config once
    show_config()
    
    vtxsurf = {"x":[-21,+21], "y":[0,30], "z":cfg["zDipoleExit"]}
    slopes  = {"xz":[-5e-3,+5e-3], "yz":[1e-2,3e-2]}
    Ngen = 10000
    
    histos = {}
    ROOT.TH2D()
    hname = "dipole_precuts";  histos.update({hname:ROOT.TH2D(hname,"Dipole exit plane (pre-cuts);x [mm];y [mm];Fake Tracks",120,-80,+80, 120,-70,+90)})
    hname = "dipole_postcuts"; histos.update({hname:ROOT.TH2D(hname,"Dipole exit plane (post-cuts);x [mm];y [mm];Fake Tracks",120,-80,+80, 120,-70,+90)})
    hname = "window_precuts";  histos.update({hname:ROOT.TH2D(hname,"Vacuum window plane (pre-cuts);x [mm];y [mm];Fake Tracks",120,-70,+70, 120,50,+190)})
    hname = "window_postcuts"; histos.update({hname:ROOT.TH2D(hname,"Vacuum window plane (pst-cuts);x [mm];y [mm];Fake Tracks",120,-70,+70, 120,50,+190)})
    
    gun = ParticleGun(vtxsurf,slopes)
    clusters = {}
    for det in cfg["detectors"]:
        clusters.update({det:[]})
    
    Nacc = 0
    for i in range(Ngen):
        fakeprt = gun.generate()
        histos["dipole_precuts"].Fill(fakeprt.vtx[0],fakeprt.vtx[1])
        histos["window_precuts"].Fill(gun.k_of_z(cfg["zWindow"],fakeprt.slp[0],fakeprt.itp[0]), gun.k_of_z(cfg["zWindow"],fakeprt.slp[1],fakeprt.itp[1]))
        if(not gun.accept(fakeprt)): continue
        Nacc += 1
        histos["dipole_postcuts"].Fill(fakeprt.vtx[0],fakeprt.vtx[1])
        histos["window_postcuts"].Fill(gun.k_of_z(cfg["zWindow"],fakeprt.slp[0],fakeprt.itp[0]), gun.k_of_z(cfg["zWindow"],fakeprt.slp[1],fakeprt.itp[1]))
        for det in cfg["detectors"]:
            clusters[det].append(fakeprt.smrpts[det])
    
    print(f"Acceptance: {(Nacc/Ngen)*100:.1f}%")
    
    dipole = ROOT.TPolyLine()
    xMinD = cfg["xDipoleExitMin"]
    xMaxD = cfg["xDipoleExitMax"]
    yMinD = cfg["yDipoleExitMin"]
    yMaxD = cfg["yDipoleExitMax"]    
    dipole.SetNextPoint(xMinD,yMinD)
    dipole.SetNextPoint(xMinD,yMaxD)
    dipole.SetNextPoint(xMaxD,yMaxD)
    dipole.SetNextPoint(xMaxD,yMinD)
    dipole.SetNextPoint(xMinD,yMinD)
    dipole.SetLineColor(ROOT.kBlue)
    dipole.SetLineWidth(1)
    
    window = ROOT.TPolyLine()
    xMinW = -cfg["xWindowWidth"]/2.
    xMaxW = +cfg["xWindowWidth"]/2.
    yMinW = cfg["yWindowMin"]
    yMaxW = cfg["yWindowMin"]+cfg["yWindowHeight"]
    window.SetNextPoint(xMinW,yMinW)
    window.SetNextPoint(xMinW,yMaxW)
    window.SetNextPoint(xMaxW,yMaxW)
    window.SetNextPoint(xMaxW,yMinW)
    window.SetNextPoint(xMinW,yMinW)    
    window.SetLineColor(ROOT.kBlue)
    window.SetLineWidth(1)
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["dipole_precuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["dipole_postcuts"].Draw("colz")
    dipole.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("gun.pdf(")
    
    cnv = ROOT.TCanvas("cnv_dipole_window","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    histos["window_precuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    histos["window_postcuts"].Draw("colz")
    window.Draw()
    ROOT.gPad.RedrawAxis()
    cnv.Update()
    cnv.SaveAs("gun.pdf)")
    
    ### turn interactive plotting off
    fname = "gun_evtdisp.pdf"
    plt.ioff()
    matplotlib.use('Agg')
    ### define the plot
    # fig = plt.figure(figsize=(15,15),frameon=False,constrained_layout=True)
    fig = plt.figure(figsize=(15,15),frameon=False)
    plt.title(f"Generator", fontdict=None, loc='center', pad=None)
    plt.box(False)
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=-0.01)
    
    ## the views
    ax1 = fig.add_subplot(221, projection='3d', facecolor='none')
    ax2 = fig.add_subplot(222, projection='3d', facecolor='none')
    ax3 = fig.add_subplot(223, projection='3d', facecolor='none')
    ax4 = fig.add_subplot(224, projection='3d', facecolor='none')
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_zlabel("z [mm]")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_zlabel("z [mm]")
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]")
    ax4.set_xlabel("x [mm]")
    ax4.set_ylabel("y [mm]")
    ax4.set_zlabel("z [mm]")
    
    ### avoid ticks and lables for projections
    ax2.zaxis.set_label_position('none')
    ax2.zaxis.set_ticks_position('none')
    ax3.xaxis.set_label_position('none')
    ax3.xaxis.set_ticks_position('none')
    ax4.yaxis.set_label_position('none')
    ax4.yaxis.set_ticks_position('none')
        
    ### the chips
    L1verts = getChips()
    ax1.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax2.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax3.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax4.add_collection3d(Poly3DCollection(L1verts, facecolors='green', linewidths=0.5, edgecolors='g', alpha=.20))
    ax1.set_box_aspect((1, 1, 1))
    ax2.set_box_aspect((1, 1, 1))
    ax3.set_box_aspect((1, 1, 1))
    ax4.set_box_aspect((1, 1, 1))
    
    ### the window
    window = getWindowRealSpace()
    ax1.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    ax2.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    ax3.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    ax4.add_collection3d(Poly3DCollection(window, facecolors='gray', linewidths=0.5, edgecolors='k', alpha=.20))
    
    ### print ALL clusters
    clsx = []
    clsy = []
    clsz = []
    for det in cfg["detectors"]:
        for cluster in clusters[det]:
            clsx.append( cluster[0] )
            clsy.append( cluster[1] )
            clsz.append( cluster[2] )
    ax1.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ax2.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ax3.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    ax4.scatter(clsx,clsy,clsz,s=0.9,c='k',marker='o',alpha=0.3)
    
    ### add beampipe
    us = np.linspace(0, 2.*np.pi, 100)
    zs = np.linspace(cfg["world"]["z"][0],cfg["world"]["z"][1], 100)
    us, zs = np.meshgrid(us,zs)
    Radius = cfg["Rpipe"]
    xs = Radius * np.cos(us)
    ys = Radius * np.sin(us)
    ys = ys-cfg["Rpipe"]+cfg["yWindowMin"]
    ax2.plot_surface(xs, ys, zs, color='b',alpha=0.3)
    ax3.plot_surface(xs, ys, zs, color='b',alpha=0.3)
    
    ## world limits
    ax1.set_xlim(cfg["world"]["x"])
    ax1.set_ylim(cfg["world"]["y"])
    ax1.set_zlim(cfg["world"]["z"])
    
    ax2.set_xlim(cfg["world"]["x"])
    ax2.set_ylim(cfg["world"]["y"])
    ax2.set_zlim(cfg["world"]["z"])
    
    ax3.set_xlim(cfg["world"]["x"])
    ax3.set_ylim(cfg["world"]["y"])
    ax3.set_zlim(cfg["world"]["z"])
    
    ax4.set_xlim(cfg["world"]["x"])
    ax4.set_ylim(cfg["world"]["y"])
    ax4.set_zlim(cfg["world"]["z"])

    ### change view of the 2nd plot: 270 is xz view, 0 is yz view, and -90 is xy view
    ax1.elev = 40
    ax1.azim = 230
    ### x-y view:
    ax2.elev = 90
    ax2.azim = 270
    ### y-z view:
    ax3.elev = 0
    ax3.azim = 0
    ### x-z view:
    ax4.elev = 0
    ax4.azim = 270

    ### finish
    plt.savefig(fname)
    plt.close(fig)
    
    
    
    