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


class Hit:
    def __init__(self,det,x,y,raw,q=-1):
        self.x = x
        self.y = y
        self.q = q
        self.raw = raw
        self.xmm = self.x*cfg["pix_x"]-cfg["chipX"]/2.
        self.ymm = self.y*cfg["pix_y"]-cfg["chipY"]/2.
        self.zmm = cfg["rdetectors"][det][2]
    def __str__(self):
        return f"Pixel: x={self.x}, y={self.y}, raw={self.raw}, q={self.q}, r=({self.xmm,self.ymm,self.zmm}) [mm]"

class Cls:
    def __init__(self,pixels,det,CID):
        self.det = det
        self.CID = CID
        self.DID = cfg["detectors"].index(det)
        self.pixels = pixels
        self.n = len(pixels)
        self.x,self.y,self.dx,self.dy = self.build(pixels) 
        self.dxmm = self.dx*cfg["pix_x"]
        self.dymm = self.dy*cfg["pix_y"]
        self.xmm0 = self.x*cfg["pix_x"]-cfg["chipX"]/2. ### original x (with misalignment)
        self.ymm0 = self.y*cfg["pix_y"]-cfg["chipY"]/2. ### original y (with misalignment)
        self.xmm,self.ymm = align(det,self.xmm0,self.ymm0) ### aligned x,y
        self.zmm  = cfg["rdetectors"][det][2]
        ### add known offset in x-y if any (MC...)
        self.xmm0 += cfg["offsets_x"][det]
        self.ymm0 += cfg["offsets_y"][det]
        self.xmm  += cfg["offsets_x"][det]
        self.ymm  += cfg["offsets_y"][det]
    def build(self,pixels):
        x = 0
        y = 0
        xmin = +1e10
        xmax = -1e10
        ymin = +1e10
        ymax = -1e10
        for pixel in pixels:
            x += pixel.x
            y += pixel.y
            xmin = pixel.x if(pixel.x<=xmin) else xmin
            xmax = pixel.x if(pixel.x>=xmax) else xmax
            ymin = pixel.y if(pixel.y<=ymin) else ymin
            ymax = pixel.y if(pixel.y>=ymax) else ymax
        dx = xmax-xmin if(self.n>0 and (xmax-xmin)>0) else 1
        dy = ymax-ymin if(self.n>0 and (ymax-ymin)>0) else 1
        x = x/self.n if(self.n>0) else -99999
        y = y/self.n if(self.n>0) else -99999
        return x,y,dx,dy
    def __str__(self):
        # for p in self.pixels: print(p)
        return f"Cluster: x={self.x}, y={self.y}, r=({self.xmm,self.ymm,self.zmm}) [mm], size={self.n}"


class MCparticle:
    def __init__(self,det,pdg,loc_start,loc_end):
        self.pdg = pdg
        self.pos1 = ROOT.Math.XYZPoint( loc_start.X()-cfg["pix_x"]*cfg["npix_x"]/2., loc_start.Y()-cfg["pix_y"]*cfg["npix_y"]/2., cfg["rdetectors"][det][2] )
        self.pos2 = ROOT.Math.XYZPoint( loc_end.X()-cfg["pix_x"]*cfg["npix_x"]/2.,   loc_end.Y()-cfg["pix_y"]*cfg["npix_y"]/2.,   cfg["rdetectors"][det][2] )
    def __str__(self):
        return f"MCparticle: pdg={self.pdg}, pos1=({self.pos1.X(),self.pos1.Y(),self.pos1.Z()}), pos2=({self.pos2.X(),self.loc_end.Y(),self.pos2.Z()})"


class Track:
    def __init__(self,cls,points,errors,chisq,ndof,direction,centroid,params,success):
        self.clusters = cls
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
    def angles(self,direction):
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]
        theta = np.arctan(np.sqrt(dx*dx+dy*dy)/dz)
        phi   = np.arctan(dy/dx)
        return theta,phi
    def __str__(self):
        return f"Track: "



class Event:
    def __init__(self,pixels,clusters,tracks,mcparticles=None):
        self.pixels = pixels
        self.clusters = clusters
        self.tracks = tracks
        self.mcparticles = mcparticles
    def __str__(self):
        return f"Event: pixels={self.pixels}, clusters={self.clusters}, tracks={self.tracks}, mcparticles={self.mcparticles}"
