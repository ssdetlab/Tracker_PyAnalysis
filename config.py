#!/usr/bin/python
import os
import math
import array
import numpy as np
import sys
import configparser

##########################################################
##########################################################
##########################################################

### should be called once from main
def init_config(fname,show):
    ConfigCls = Config(fname,show)
    # https://stackoverflow.com/questions/13034496/using-global-variables-between-files
    global cfg
    cfg = ConfigCls.map

def show_config():
    print("Configuration map:")
    for key,val in cfg.items(): print(f"{key}: {val}")
    print("")

### config file looks like that:
# [SECTION_NAME]
# key1 = value1
# key2 = value2
class Config:
    def __init__(self,fname,doprint=False):
        self.fname = fname
        self.doprint = doprint
        self.configurator = configparser.RawConfigParser()
        self.configurator.optionxform = str ### preserve case sensitivity
        self.map = {} ### the config map
        self.set(fname,doprint)
        
    def read(self,fname):
        if(self.doprint): print("Reading configuration from: ",fname)
        self.configurator.read(fname)
        
    def getF(self,section,var):
        expr = dict(self.configurator.items(section))[var]
        if(not expr.isnumeric()):
            return float(eval(expr))
        return float(dict(self.configurator.items(section))[var])
       
    def getI(self,section,var):
        expr = dict(self.configurator.items(section))[var]
        if(not expr.isnumeric()):
            return int(eval(expr))
        return int(dict(self.configurator.items(section))[var])
    
    def getB(self,section,var):
        return True if(int(dict(self.configurator.items(section))[var])==1) else False
    
    def getS(self,section,var):
        return str(dict(self.configurator.items(section))[var])

    def getArrS(self,section,var):
        s = self.getS(section,var)
        return s.split(" ")
    
    def getArrI(self,section,var):
        s = self.getS(section,var).split(",")
        i = [int(x) for x in s]
        return i
        
    def getArrF(self,section,var):
        s = self.getS(section,var).split(",")
        f = [float(x) for x in s]
        return f
    
    def getMapI2S(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sval = x[1]
            m.update({int(sval):name})
        return m
    
    def getMap2ArrF(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sarr = x[1].split(",")
            farr = [float(n) for n in sarr]
            m.update({name:farr})
        return m
        
    def getMap2MapF(self,section,var):
        s = self.getS(section,var).split(" ")
        m = {}
        for x in s:
            x = x.split(":")
            name = x[0]
            sarr = x[1].split(",")
            ff = {}
            for ss in sarr:
                v = ss.split("=")
                ff.update({v[0]:float(v[1])})
            m.update({name:ff})
        return m
    
    def add(self,name,var):
        self.map.update( {name:var} )
    
    def set(self,fname,doprint=False):
        ### read
        self.read(fname)
        ### set
        self.add("isMC", self.getB('RUN','isMC'))
        self.add("isCVMFS", self.getB('RUN','isCVMFS'))
        self.add("doVtx", self.getB('RUN','doVtx'))
        self.add("runtype", self.getS('RUN','runtype'))
        self.add("pdgIdMatch", self.getI('RUN','pdgIdMatch'))
        self.add("nmax2process", self.getI('RUN','nmax2process'))
        self.add("nmax2processMP", self.getI('RUN','nmax2processMP'))
        self.add("nCPU", self.getI('RUN','nCPU'))
        self.add("doplot", self.getB('RUN','doplot'))
        self.add("doDiagnostics", self.getB('RUN','doDiagnostics'))
        self.add("doNoiseScan", self.getB('RUN','doNoiseScan'))
        self.add("isCVRroot", self.getB('RUN','isCVRroot'))
        self.add("nprintout", self.getI('RUN','nprintout'))
        self.add("inputfile", self.getS('RUN','inputfile'))

        self.add("npix_x", self.getI('CHIP','npix_x'))
        self.add("npix_y", self.getI('CHIP','npix_y'))
        self.add("pix_x",  self.getF('CHIP','pix_x'))
        self.add("pix_y",  self.getF('CHIP','pix_y'))
        self.add("chipX",  self.map["npix_x"]*self.map["pix_x"])
        self.add("chipY",  self.map["npix_y"]*self.map["pix_y"])

        self.add("xVtx", self.getF('VTX','xVtx'))
        self.add("yVtx", self.getF('VTX','yVtx'))
        self.add("zVtx", self.getF('VTX','zVtx'))
        self.add("exVtx", self.getF('VTX','exVtx'))
        self.add("eyVtx", self.getF('VTX','exVtx'))
        self.add("ezVtx", self.getF('VTX','exVtx'))

        self.add("ezCls", self.getF('CLUSTER','ezCls'))

        self.add("lineScaleUp", self.getF('WORLD','lineScaleUp'))
        self.add("lineScaleDn", self.getF('WORLD','lineScaleDn'))

        self.add("pTrim", self.getF('NOISE','pTrim'))
        self.add("zeroSupp", self.getB('NOISE','zeroSupp'))
        self.add("nSigma", self.getF('NOISE','nSigma'))

        self.add("detectors", self.getArrS('DETECTOR','detectors'))
        self.add("plane2det", self.getMapI2S('DETECTOR','plane2det'))
        self.add("rdetectors", self.getMap2ArrF('DETECTOR','rdetectors'))
        self.add("misalignment", self.getMap2MapF('DETECTOR','misalignment'))
        self.add("maxchi2align", self.getF('DETECTOR','maxchi2align'))
        self.add("axes2align", self.getS('DETECTOR','axes2align'))
        self.add("naligniter", self.getI('DETECTOR','naligniter'))
        self.add("alignmentbins", self.getMap2MapF('DETECTOR','alignmentbins'))
        
        firstdet = self.map["detectors"][0]
        lastdet  = self.map["detectors"][-1]
        
        # detectorslist = list(self.map["detectors"])
        # rdetectorslist = detectorslist.reverse()
        # self.add("detectorslist", detectorslist)
        # self.add("rdetectorslist", detectorslist)
        
        self.add("worldmargins", self.getF('DETECTOR','worldmargins'))
        # self.add("zFirst", self.map["zVtx"]*(1-self.map["worldmargins"]))
        # self.add("zLast", self.map["rdetectors"]["ALPIDE_2"][2]*(1+self.map["worldmargins"]))
        if(self.map["doVtx"]):
            self.add("zFirst", self.map["zVtx"]*(1-self.map["worldmargins"]))
        else:
            self.add("zFirst", self.map["rdetectors"][firstdet][2]*(1-self.map["worldmargins"]))
        self.add("zLast", self.map["rdetectors"][lastdet][2]*(1+self.map["worldmargins"]))
        self.add("worldscales", self.getMap2ArrF('DETECTOR','worldscales'))
        self.add("worldcenter", self.getArrF('DETECTOR','worldcenter'))
        self.add("worldradius",  self.getF('DETECTOR','worldradius'))
        world = {}
        for axis,scales in self.map["worldscales"].items():
            bounds = -9999
            if(axis=="x"): bounds = [ -self.map["chipX"]*scales[0],+self.map["chipX"]*scales[1] ]
            if(axis=="y"): bounds = [ -self.map["chipY"]*scales[0],+self.map["chipY"]*scales[1] ]
            if(axis=="z"): bounds = [ self.map["zFirst"]*scales[0], self.map["zLast"]*scales[1] ]
            world.update( {axis:bounds} )
        self.add("world", world)
        
        offsets_x = {}
        offsets_y = {}
        for det in self.map["detectors"]:
            offsets_x.update( {det:self.map["rdetectors"][det][0]} )
            offsets_y.update( {det:self.map["rdetectors"][det][1]} )
        self.add("offsets_x", offsets_x)
        self.add("offsets_y", offsets_y)
        
        self.add("cuts", self.getArrS('CUTS','cuts'))
    
        if(doprint):
            print("Configuration map:")
            for key,val in self.map.items():
                print(f"{key}: {val}")
            print("")

    def __str__(self):
        return f"Config map: {self.map}"