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
    if(not os.path.isfile(fname)):
        print(f"Config file {fname} does not exist. Quitting.")
        quit()
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
        s = self.getS(section,var).split(" ")
        i = [int(x) for x in s]
        return i
        
    def getArrF(self,section,var):
        s = self.getS(section,var).split(" ")
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
        self.add("doVtx", self.getB('RUN','doVtx'))
        self.add("runtype", self.getS('RUN','runtype'))
        self.add("pdgIdMatch", self.getI('RUN','pdgIdMatch'))
        self.add("nmax2process", self.getI('RUN','nmax2process'))
        self.add("first2process", self.getI('RUN','first2process'))
        if(self.map["first2process"]<=0): self.map["first2process"] = 0
        self.add("nCPU", self.getI('RUN','nCPU'))
        self.add("nprintout", self.getI('RUN','nprintout'))
        self.add("skipmasking", self.getB('RUN','skipmasking'))
        self.add("inputfile", self.getS('RUN','inputfile'))
        
        self.add("runnums", self.getArrI('MULTIRUN','runnums'))

        self.add("npix_x", self.getI('CHIP','npix_x'))
        self.add("npix_y", self.getI('CHIP','npix_y'))
        self.add("pix_x",  self.getF('CHIP','pix_x'))
        self.add("pix_y",  self.getF('CHIP','pix_y'))
        self.add("chipX",  self.map["npix_x"]*self.map["pix_x"])
        self.add("chipY",  self.map["npix_y"]*self.map["pix_y"])
        
        self.add("seed_allow_neigbours", self.getB('SEED','seed_allow_neigbours'))
        self.add("seed_thetax_scale", self.getF('SEED','seed_thetax_scale'))
        self.add("seed_rhox_scale", self.getF('SEED','seed_rhox_scale'))
        self.add("seed_thetay_scale", self.getF('SEED','seed_thetay_scale'))
        self.add("seed_rhoy_scale", self.getF('SEED','seed_rhoy_scale'))
        self.add("seed_nbins_theta_020", self.getI('SEED','seed_nbins_theta_020'))
        self.add("seed_nbins_theta_200", self.getI('SEED','seed_nbins_theta_200'))
        self.add("seed_nbins_theta_inf", self.getI('SEED','seed_nbins_theta_inf'))
        self.add("seed_nbins_rho_020", self.getI('SEED','seed_nbins_rho_020'))
        self.add("seed_nbins_rho_200", self.getI('SEED','seed_nbins_rho_200'))
        self.add("seed_nbins_rho_inf", self.getI('SEED','seed_nbins_rho_inf'))
        
        self.add("lut_nbinsx_020", self.getI('LUT','lut_nbinsx_020'))
        self.add("lut_nbinsy_020", self.getI('LUT','lut_nbinsy_020'))
        self.add("lut_nbinsx_200", self.getI('LUT','lut_nbinsx_200'))
        self.add("lut_nbinsy_200", self.getI('LUT','lut_nbinsy_200'))
        self.add("lut_nbinsx_inf", self.getI('LUT','lut_nbinsx_inf'))
        self.add("lut_nbinsy_inf", self.getI('LUT','lut_nbinsy_inf'))
        self.add("lut_scaleX", self.getF('LUT','lut_scaleX'))
        self.add("lut_scaleY", self.getF('LUT','lut_scaleY'))
        self.add("lut_widthx_020", self.getF('LUT','lut_widthx_020'))
        self.add("lut_widthy_020", self.getF('LUT','lut_widthy_020'))
        self.add("lut_widthx_200", self.getF('LUT','lut_widthx_200'))
        self.add("lut_widthy_200", self.getF('LUT','lut_widthy_200'))
        self.add("lut_widthx_inf", self.getF('LUT','lut_widthx_inf'))
        self.add("lut_widthy_inf", self.getF('LUT','lut_widthy_inf'))

        self.add("xVtx", self.getF('VTX','xVtx'))
        self.add("yVtx", self.getF('VTX','yVtx'))
        self.add("zVtx", self.getF('VTX','zVtx'))
        self.add("exVtx", self.getF('VTX','exVtx'))
        self.add("eyVtx", self.getF('VTX','exVtx'))
        self.add("ezVtx", self.getF('VTX','exVtx'))

        self.add("ezCls", self.getF('CLUSTER','ezCls'))
        self.add("allow_diagonals", self.getB('CLUSTER','allow_diagonals'))

        self.add("worldbounds", self.getMap2ArrF('WORLD','worldbounds'))
        world = {}
        for axis,bound in self.map["worldbounds"].items():
            bounds = [ bound[0], bound[1] ]
            world.update( {axis:bounds} )            
        self.add("world", world)

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
        self.add("alignmentbounds", self.getMap2MapF('DETECTOR','alignmentbounds'))
        
        firstdet = self.map["detectors"][0]
        lastdet  = self.map["detectors"][-1]
        
        self.add("zWindow",       self.getF('WINDOW','zWindow'))
        self.add("xWindow",       self.getF('WINDOW','xWindow'))
        self.add("yWindowMin",    self.getF('WINDOW','yWindowMin'))
        self.add("xWindowWidth",  self.getF('WINDOW','xWindowWidth'))
        self.add("yWindowHeight", self.getF('WINDOW','yWindowHeight'))
        
        self.add("Rpipe", self.getF('BEAMPIPE','Rpipe'))
        self.add("yMidWin2PipeCenter", self.getF('BEAMPIPE','yMidWin2PipeCenter'))
        
        thetaz = self.getF('TRANSFORMATIONS','thetaz')*np.pi/180.
        self.add("thetaz", thetaz)
        self.add("xOffset", self.getF('TRANSFORMATIONS','xOffset'))
        self.add("yBoxBot2WinBot", self.getF('TRANSFORMATIONS','yBoxBot2WinBot'))
        self.add("yMidChip2BoxBot", self.getF('TRANSFORMATIONS','yMidChip2BoxBot'))
        self.add("zWin2Box", self.getF('TRANSFORMATIONS','zWin2Box'))
        self.add("zBox2chip", self.getF('TRANSFORMATIONS','zBox2chip'))
        yOffset = self.map["yWindowMin"]+self.map["yBoxBot2WinBot"]+self.map["yMidChip2BoxBot"]/2.
        zOffset = self.map["zWin2Box"]+self.map["zBox2chip"]
        self.add("yOffset", yOffset)
        self.add("zOffset", zOffset)
        
        offsets_x = {}
        offsets_y = {}
        for det in self.map["detectors"]:
            offsets_x.update( {det:self.map["rdetectors"][det][0]} )
            offsets_y.update( {det:self.map["rdetectors"][det][1]} )
        self.add("offsets_x", offsets_x)
        self.add("offsets_y", offsets_y)
        
        self.add("fast",    self.getB('FIT','fast'))
        self.add("method0", self.getS('FIT','method0'))
        self.add("method1", self.getArrS('FIT','method1'))
        
        self.add("cuts", self.getArrS('CUTS','cuts'))
        self.add("cut_chi2dof", self.getF('CUTS','cut_chi2dof'))
        self.add("cut_ROI_xmin", self.getF('CUTS','cut_ROI_xmin'))
        self.add("cut_ROI_xmax", self.getF('CUTS','cut_ROI_xmax'))
        self.add("cut_ROI_ymin", self.getF('CUTS','cut_ROI_ymin'))
        self.add("cut_ROI_ymax", self.getF('CUTS','cut_ROI_ymax'))
    
        if(doprint):
            print("Configuration map:")
            for key,val in self.map.items():
                print(f"{key}: {val}")
            print("")

    def __str__(self):
        return f"Config map: {self.map}"