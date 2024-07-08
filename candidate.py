#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT
# from ROOT import *

import config
from config import *
import objects
from objects import *


def SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx=[],evtx=[]):
    isvtx = (len(vtx)>0 and len(evtx)>0)
    clusters = [ [cfg["xVtx"], cfg["yVtx"], cfg["zVtx"]] ]  if(isvtx) else []
    clerrors = [ [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] ] if(isvtx) else []
    for det in cfg["detectors"]:
        clusters.append( [clsx[det],  clsy[det],  clsz[det]] )
        clerrors.append( [clsdx[det], clsdy[det], cfg["ezCls"]] )
    points = np.array(clusters)
    errors = np.array(clerrors)
    return points,errors


def Chi2_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx=[],evtx=[]):
    isvtx = (len(vtx)>0 and len(evtx)>0)
    clusters_x = [cfg["xVtx"]]  if(isvtx) else []
    clusters_y = [cfg["yVtx"]]  if(isvtx) else []
    clusters_z = [cfg["zVtx"]]  if(isvtx) else []
    clerrors_x = [cfg["exVtx"]] if(isvtx) else []
    clerrors_y = [cfg["eyVtx"]] if(isvtx) else []
    clerrors_z = [cfg["ezVtx"]] if(isvtx) else []
    for det in cfg["detectors"]:
        clusters_x.append( clsx[det] )
        clusters_y.append( clsy[det] )
        clusters_z.append( clsz[det] )
        clerrors_x.append( clsdx[det] )
        clerrors_y.append( clsdy[det] )
        clerrors_z.append( cfg["ezCls"] )
    points = np.array([ clusters_x,clusters_y,clusters_z ])
    errors = np.array([ clerrors_x,clerrors_y,clerrors_z ])
    return points,errors