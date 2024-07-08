#!/usr/bin/python
import os
import math
import array
import numpy as np
import ROOT
# from ROOT import *

import objects
from objects import *

def get_truth_cvr(truth_tree,ientry):
    truth_tree.GetEntry(ientry)
    mcparticles = {}
    for det in cfg["detectors"]:
        mcparticles.update({det:[]})
    for i in range(truth_tree.ALPIDE_0.size()):
        det = "ALPIDE_0"
        pdg   = truth_tree.ALPIDE_0[i].getID()
        start = truth_tree.ALPIDE_0[i].getLocalStart()
        end   = truth_tree.ALPIDE_0[i].getLocalEnd()
        mcparticles[det].append( MCparticle(det,pdg,start,end) )
    for i in range(truth_tree.ALPIDE_1.size()):
        det = "ALPIDE_1"
        pdg   = truth_tree.ALPIDE_1[i].getID()
        start = truth_tree.ALPIDE_1[i].getLocalStart()
        end   = truth_tree.ALPIDE_1[i].getLocalEnd()
        mcparticles[det].append( MCparticle(det,pdg,start,end) )
    for i in range(truth_tree.ALPIDE_2.size()):
        det = "ALPIDE_2"
        pdg   = truth_tree.ALPIDE_2[i].getID()
        start = truth_tree.ALPIDE_2[i].getLocalStart()
        end   = truth_tree.ALPIDE_2[i].getLocalEnd()
        mcparticles[det].append( MCparticle(det,pdg,start,end) )
    return mcparticles


def getTruPos(det,mcparticles,pdg):
    xtru = -99999
    ytru = -99999
    ztru = -99999
    if(len(mcparticles)==0): return xtru,ytru,ztru
    for prt in mcparticles[det]:
        if(abs(prt.pdg)!=pdg): continue ### take only the target pdgId
        xtru = prt.pos1.X()
        ytru = prt.pos1.Y()
        ztru = prt.pos1.Z()
        break ### take only the first mcparticle that matches
    return xtru,ytru,ztru