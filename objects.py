#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT

import config
from config import *
import utils
from utils import *

_s12_  = math.sqrt(12.)
_1s12_ = 1./_s12_

class Hit:
    def __init__(self,det,x,y,q=-1,xOrig=0,yOrig=0,xFake=0,yFake=0,Azx=0,Bzx=0,Azy=0,Bzy=0,Vx=0,Vy=0,Vz=0):
        self.x = x
        self.y = y
        self.q = q
        self.xmm = self.x*cfg["pix_x"]-cfg["chipX"]/2.
        self.ymm = self.y*cfg["pix_y"]-cfg["chipY"]/2.
        self.zmm = cfg["rdetectors"][det][2]
        if(cfg["isMC"] and cfg["isFakeMC"]):
            self.xOrig = xOrig
            self.yOrig = yOrig
            self.xFake = xFake
            self.yFake = yFake
            self.Azx   = Azx
            self.Bzx   = Bzx
            self.Azy   = Azy
            self.Bzy   = Bzy
            self.Vx    = Vx
            self.Vy    = Vy
            self.Vz    = Vz
        
    def __str__(self):
        return f"Pixel: x={self.x}, y={self.y}, q={self.q}, r=({self.xmm,self.ymm,self.zmm}) [mm]"

class Cls:
    def __init__(self,det,pixels,CID):
        self.det = det
        self.CID = CID
        self.DID = cfg["detectors"].index(det)
        self.pixels = pixels
        self.n = len(pixels)
        self.x,self.y,self.dx,self.dy,self.nx,self.ny = self.build(pixels) 
        self.dxmm = self.dx*cfg["pix_x"]
        self.dymm = self.dy*cfg["pix_y"]
        self.xsizemm = self.nx*cfg["pix_x"]/2.
        self.ysizemm = self.ny*cfg["pix_y"]/2.
        self.xmm0 = self.x*cfg["pix_x"]-cfg["chipX"]/2. ### original x (with misalignment)
        self.ymm0 = self.y*cfg["pix_y"]-cfg["chipY"]/2. ### original y (with misalignment)
        if(cfg["isFakeMC"]):
            self.xmm0 = pixels[0].xFake 
            self.ymm0 = pixels[0].yFake
        self.xmm,self.ymm = align(det,self.xmm0,self.ymm0) ### aligned x,y
        self.zmm  = cfg["rdetectors"][det][2]
        ### add known offset in x-y if any
        self.xmm0 += cfg["offsets_x"][det]
        self.ymm0 += cfg["offsets_y"][det]
        self.xmm  += cfg["offsets_x"][det]
        self.ymm  += cfg["offsets_y"][det]
    def build(self,pixels):
        if(self.n<1):
            print(f"cannot build a cluster from n={self.n} pixels. quitting.")
            quit()
        mu_x = 0
        mu_y = 0
        mu_x2 = 0
        mu_y2 = 0
        xmin = +1e10
        xmax = -1e10
        ymin = +1e10
        ymax = -1e10
        for pixel in pixels:
            mu_x  += pixel.x
            mu_y  += pixel.y
            mu_x2 += pixel.x**2
            mu_y2 += pixel.y**2
            xmin = pixel.x if(pixel.x<xmin) else xmin
            ymin = pixel.y if(pixel.y<ymin) else ymin
            xmax = pixel.x if(pixel.x>xmax) else xmax
            ymax = pixel.y if(pixel.y>ymax) else ymax
        nx = xmax-xmin if(xmax>xmin) else 1
        ny = ymax-ymin if(ymax>ymin) else 1
        mu_x  = mu_x/self.n
        mu_y  = mu_y/self.n
        mu_x2 = mu_x2/self.n
        mu_y2 = mu_y2/self.n
        varx  = mu_x2-mu_x**2
        vary  = mu_y2-mu_y**2
        se_x  = _1s12_/math.sqrt(self.n)
        se_y  = _1s12_/math.sqrt(self.n)
        return mu_x,mu_y,se_x,se_y,nx,ny
    def __str__(self):
        # for p in self.pixels: print(p)
        return f"Cluster: xy={self.x,self.y} [pixels], r={self.xmm,self.ymm,self.zmm} [mm], size={self.n}"

class MCparticle:
    def __init__(self,det,pdg,loc_start,loc_end):
        self.pdg = pdg
        self.pos1 = ROOT.Math.XYZPoint( loc_start.X()-cfg["pix_x"]*cfg["npix_x"]/2., loc_start.Y()-cfg["pix_y"]*cfg["npix_y"]/2., cfg["rdetectors"][det][2] )
        self.pos2 = ROOT.Math.XYZPoint( loc_end.X()-cfg["pix_x"]*cfg["npix_x"]/2.,   loc_end.Y()-cfg["pix_y"]*cfg["npix_y"]/2.,   cfg["rdetectors"][det][2] )
    def __str__(self):
        return f"MCparticle: pdg={self.pdg}, pos1=({self.pos1.X(),self.pos1.Y(),self.pos1.Z()}), pos2=({self.pos2.X(),self.loc_end.Y(),self.pos2.Z()})"

class FakeMCparticle:
    def __init__(self,slp,itp,vtx):
        self.slp = slp
        self.itp = itp
        self.vtx = vtx
    def __str__(self):
        return f"FakeMCparticle: slp={self.slp}, itp={self.itp}, vtx={vtx}"

class TrackSeed:
    def __init__(self,seed,tunnelid,clusters):
        self.clsids = seed
        self.tunnelid = tunnelid
        self.x  = {}
        self.y  = {}
        self.z  = {}
        self.dx = {}
        self.dy = {}
        self.xsize = {}
        self.ysize = {}
        for idet,det in enumerate(cfg["detectors"]):
            icls = seed[idet]
            self.x.update({  det:clusters[det][icls].xmm  })
            self.y.update({  det:clusters[det][icls].ymm  })
            self.z.update({  det:clusters[det][icls].zmm  })
            self.dx.update({ det:clusters[det][icls].dxmm })
            self.dy.update({ det:clusters[det][icls].dymm })
            self.xsize.update({ det:clusters[det][icls].xsizemm })
            self.ysize.update({ det:clusters[det][icls].ysizemm })
    def __str__(self):
        return f"TrackSeed: "

class Track:
    def __init__(self,trkcls,points,errors,chisq,ndof,direction,centroid,params,success):
        self.trkcls = trkcls
        self.points = points
        self.errors = errors
        self.chisq = chisq
        self.ndof = ndof
        self.chi2ndof = chisq/ndof if(ndof>0) else 99999
        self.direction = direction
        self.centroid = centroid
        self.params = params
        self.success = success
        self.theta,self.phi = self.angles(direction)
        self.maxcls = self.max_cls_size()
    def angles(self,direction):
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]
        theta = np.arctan(np.sqrt(dx*dx+dy*dy)/dz)
        phi   = np.arctan(dy/dx)
        return theta,phi
    def max_cls_size(self):
        maxcls = 0
        for det,cl in self.trkcls.items():
            if(cl.n>maxcls): maxcls = cl.n
        return maxcls
    def __str__(self):
        return f"Track: chisq={self.chisq}, ndof={self.ndof}, chi2ndof={self.chi2ndof}"

class Meta:
    def __init__(self,run,start,end,dur):
        self.run = run
        self.start = start
        self.end = end
        self.dur = dur
    def __str__(self):
        return f"Meta: "

class Event:
    def __init__(self,meta,trigger,timestamp_bgn,timestamp_end,dipole,saveprimitive=True):
        self.saveprimitive   = saveprimitive
        self.meta            = meta
        self.trigger         = trigger
        self.timestamp_bgn   = timestamp_bgn
        self.timestamp_end   = timestamp_end
        self.dipole          = dipole
        self.errors          = {}
        self.pixels          = {}
        self.npixels         = {}
        self.clusters        = {}
        self.nclusters       = {}
        self.seeds           = []
        self.tracks          = []
        self.mcparticles     = []
        self.fakemcparticles = []
        self.misalignment  = cfg["misalignment"]
    def __str__(self):
        return f"Event: meta={self.meta}"
    def set_event_errors(self,errors):
        self.errors = errors
    def set_event_pixels(self,pixels):
        for det in cfg["detectors"]: self.npixels.update({det:len(pixels[det])})
        self.pixels = pixels.copy() if(self.saveprimitive) else {}
    def set_event_clusters(self,clusters):
        for det in cfg["detectors"]: self.nclusters.update({det:len(clusters[det])})
        self.clusters = clusters if(self.saveprimitive) else {}
    def set_event_seeds(self,seeds):
        self.seeds = seeds
    def set_event_tracks(self,tracks):
        self.tracks = tracks
    def set_event_mcparticles(self,mcparticles):
        self.mcparticles = mcparticles
    def set_event_fakemcparticles(self,fakemcparticles):
        self.fakemcparticles = fakemcparticles
        
class MinimalEvent:
    def __init__(self,trigger,tracks):
        self.trigger = trigger
        self.tracks  = tracks
