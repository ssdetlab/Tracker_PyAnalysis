#!/usr/bin/python
import os
import math
import subprocess
import array
import numpy as np
import ROOT
# from ROOT import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit

import config
from config import *
import utils
from utils import *
import objects
from objects import *
import chi2_fit
from chi2_fit import *

### global parameters for pixel matrix
pix_x_nbins = cfg["npix_x"]+1
pix_x_min  = -0.5
pix_x_max  = cfg["npix_x"]+0.5
pix_y_nbins = cfg["npix_y"]+1
pix_y_min  = -0.5
pix_y_max  = cfg["npix_y"]+0.5

absRes  = 0.05
absChi2 = 20
if(cfg["runtype"]=="source"):
    absRes  *= 20
    absChi2 *= 20
nResBins = int(absRes*1000)

### book histos
def book_histos(tfo):
    histos = {}
    
    tfo.cd()
    for det in cfg["detectors"]: tfo.mkdir(det)
    tfo.cd()
    
    histos.update( { "h_events" : ROOT.TH1D("h_events",";;Events",1,0,1) } )
    histos["h_events"].GetXaxis().SetBinLabel(1,"")
    
    histos.update( { "h_cutflow"   : ROOT.TH1D("h_cutflow",";;Events",10,0,10) } )
    for b in range(1,len(cfg["cuts"])+1):
        histos["h_cutflow"].GetXaxis().SetBinLabel(b,cfg["cuts"][b-1])
    
    histos.update( { "h_3Dchi2err"      : ROOT.TH1D("h_3Dchi2err",";3D-#chi^{2} fit w/err: #chi^{2}/N_{dof};Tracks",200,0,absChi2) } )
    histos.update( { "h_3Dchi2err_full" : ROOT.TH1D("h_3Dchi2err_full",";3D-#chi^{2} fit w/err: #chi^{2}/N_{dof};Tracks",200,0,absChi2*10) } )
    histos.update( { "h_3Dchi2err_zoom" : ROOT.TH1D("h_3Dchi2err_zoom",";3D-#chi^{2} fit w/err: #chi^{2}/N_{dof};Tracks",200,0,absChi2/5.) } )
    histos.update( { "h_npix"           : ROOT.TH1D("h_npix",";N_{pixels}/detector;Events",30,0,30) } )

    histos.update( { "h_Chi2fit_res_trk2vtx_x" : ROOT.TH1D("h_Chi2fit_res_trk2vtx_x",";x_{trk}-x_{vtx} [mm];Events",nResBins,-absRes,+absRes) } )
    histos.update( { "h_Chi2fit_res_trk2vtx_y" : ROOT.TH1D("h_Chi2fit_res_trk2vtx_y",";y_{trk}-y_{vtx} [mm];Events",nResBins,-absRes,+absRes) } )
    
    histos.update( { "h_Chi2_phi"            : ROOT.TH1D("h_Chi2_phi",";Chi2 fit: #phi;Tracks",100,-np.pi,+np.pi) } )
    histos.update( { "h_Chi2_theta"          : ROOT.TH1D("h_Chi2_theta",";Chi2 fit: #theta;Tracks",100,0,np.pi) } )
    histos.update( { "h_Chi2_theta_weighted" : ROOT.TH1D("h_Chi2_theta_weighted",";Chi2 fit: #theta weighted;Tracks",100,0,np.pi) } )
    
    histos.update( { "h_3Dsphere"   : ROOT.TH3D("h_3Dsphere",  ";x [mm];y [mm];z [mm]",50,1.2*cfg["world"]["x"][0],1.2*cfg["world"]["x"][1], 50,1.2*cfg["world"]["y"][0],1.2*cfg["world"]["y"][1], 50,1.2*cfg["world"]["z"][0],1.2*cfg["world"]["z"][1]) } )
    histos.update( { "h_3Dsphere_a" : ROOT.TH3D("h_3Dsphere_a",";x [mm];y [mm];z [mm]",50,1.2*cfg["world"]["x"][0],1.2*cfg["world"]["x"][1], 50,1.2*cfg["world"]["y"][0],1.2*cfg["world"]["y"][1], 50,1.2*cfg["world"]["z"][0],1.2*cfg["world"]["z"][1]) } )
    histos.update( { "h_3Dsphere_b" : ROOT.TH3D("h_3Dsphere_b",";x [mm];y [mm];z [mm]",50,1.2*cfg["world"]["x"][0],1.2*cfg["world"]["x"][1], 50,1.2*cfg["world"]["y"][0],1.2*cfg["world"]["y"][1], 50,1.2*cfg["world"]["z"][0],1.2*cfg["world"]["z"][1]) } )
    
    histos.update( { "h_tru_3D"   : ROOT.TH3D("h_tru_3D",  ";x [mm];y [mm];z [mm]",50,1.2*cfg["world"]["x"][0],1.2*cfg["world"]["x"][1], 50,1.2*cfg["world"]["y"][0],1.2*cfg["world"]["y"][1], 50,1.2*cfg["world"]["z"][0],1.2*cfg["world"]["z"][1]) } )
    histos.update( { "h_cls_3D"   : ROOT.TH3D("h_cls_3D",  ";x [mm];y [mm];z [mm]",50,1.2*cfg["world"]["x"][0],1.2*cfg["world"]["x"][1], 50,1.2*cfg["world"]["y"][0],1.2*cfg["world"]["y"][1], 50,1.2*cfg["world"]["z"][0],1.2*cfg["world"]["z"][1]) } )
    histos.update( { "h_fit_3D"   : ROOT.TH3D("h_fit_3D",  ";x [mm];y [mm];z [mm]",50,1.2*cfg["world"]["x"][0],1.2*cfg["world"]["x"][1], 50,1.2*cfg["world"]["y"][0],1.2*cfg["world"]["y"][1], 50,1.2*cfg["world"]["z"][0],1.2*cfg["world"]["z"][1]) } )
    
    histos.update( { "h_csize_vs_trupos" : ROOT.TH2D("h_csize_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Mean cluster size",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    histos.update( { "h_ntrks_vs_trupos" : ROOT.TH2D("h_ntrks_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Mean cluster size",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
    histos.update( { "h_csize_1_vs_trupos" : ROOT.TH2D("h_csize_1_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=1",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    histos.update( { "h_ntrks_1_vs_trupos" : ROOT.TH2D("h_ntrks_1_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=1",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
    histos.update( { "h_csize_2_vs_trupos" : ROOT.TH2D("h_csize_2_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=2",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    histos.update( { "h_ntrks_2_vs_trupos" : ROOT.TH2D("h_ntrks_2_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=2",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )

    histos.update( { "h_csize_3_vs_trupos" : ROOT.TH2D("h_csize_3_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=3",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    histos.update( { "h_ntrks_3_vs_trupos" : ROOT.TH2D("h_ntrks_3_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=3",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
    histos.update( { "h_csize_4_vs_trupos" : ROOT.TH2D("h_csize_4_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    histos.update( { "h_ntrks_4_vs_trupos" : ROOT.TH2D("h_ntrks_4_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size=4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
    histos.update( { "h_csize_n_vs_trupos" : ROOT.TH2D("h_csize_n_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size>4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    histos.update( { "h_ntrks_n_vs_trupos" : ROOT.TH2D("h_ntrks_n_vs_trupos",";x_{tru} [mm]; y_{tru} [mm];Cluster size>4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )


    for det in cfg["detectors"]:
        
        tfo.cd(det)
        
        histos.update( { "h_pix_occ_1D_"+det        : ROOT.TH1D("h_pix_occ_1D_"+det,";Pixel;Hits",cfg["npix_x"]*cfg["npix_y"],1,cfg["npix_x"]*cfg["npix_y"]+1) } )
        histos.update( { "h_pix_occ_1D_masked_"+det : ROOT.TH1D("h_pix_occ_1D_masked_"+det,";Pixel;Hits",cfg["npix_x"]*cfg["npix_y"],1,cfg["npix_x"]*cfg["npix_y"]+1) } )
        histos.update( { "h_pix_occ_2D_"+det        : ROOT.TH2D("h_pix_occ_2D_"+det,";x;y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )
        histos.update( { "h_pix_occ_2D_masked_"+det : ROOT.TH2D("h_pix_occ_2D_masked_"+det,";x;y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )
        
        histos.update( { "h_cls_occ_2D_"+det        : ROOT.TH2D("h_cls_occ_2D_"+det,";x;y;Clusters",cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2.) } )
        histos.update( { "h_cls_occ_2D_masked_"+det : ROOT.TH2D("h_cls_occ_2D_masked_"+det,";x;y;Clusters",cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2.) } )
        
        histos.update( { "h_csize_vs_trupos_"+det : ROOT.TH2D("h_csize_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Mean cluster size",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        histos.update( { "h_ntrks_vs_trupos_"+det : ROOT.TH2D("h_ntrks_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Mean cluster size",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        
        histos.update( { "h_csize_1_vs_trupos_"+det : ROOT.TH2D("h_csize_1_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=1",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        histos.update( { "h_ntrks_1_vs_trupos_"+det : ROOT.TH2D("h_ntrks_1_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=1",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
        histos.update( { "h_csize_2_vs_trupos_"+det : ROOT.TH2D("h_csize_2_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=2",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        histos.update( { "h_ntrks_2_vs_trupos_"+det : ROOT.TH2D("h_ntrks_2_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=2",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )

        histos.update( { "h_csize_3_vs_trupos_"+det : ROOT.TH2D("h_csize_3_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=3",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        histos.update( { "h_ntrks_3_vs_trupos_"+det : ROOT.TH2D("h_ntrks_3_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=3",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
        histos.update( { "h_csize_4_vs_trupos_"+det : ROOT.TH2D("h_csize_4_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        histos.update( { "h_ntrks_4_vs_trupos_"+det : ROOT.TH2D("h_ntrks_4_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size=4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
    
        histos.update( { "h_csize_n_vs_trupos_"+det : ROOT.TH2D("h_csize_n_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size>4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        histos.update( { "h_ntrks_n_vs_trupos_"+det : ROOT.TH2D("h_ntrks_n_vs_trupos_"+det,";x_{tru} [mm]; y_{tru} [mm];Cluster size>4",80,0.,2.*cfg["pix_x"], 80,0.,2.*cfg["pix_y"]) } )
        
        histos.update( { "h_tru_occ_2D_"+det : ROOT.TH2D("h_tru_occ_2D_"+det,";Fit x; Fit y;Tracks",200,-cfg["chipX"]/2.+cfg["offsets_x"][det],+cfg["chipX"]/2.+cfg["offsets_x"][det], 100,-cfg["chipY"]/2.+cfg["offsets_y"][det],+cfg["chipY"]/2.+cfg["offsets_y"][det]) } )

        histos.update( { "h_fit_occ_2D_"+det : ROOT.TH2D("h_fit_occ_2D_"+det,";Fit x; Fit y;Tracks",200,-cfg["chipX"]/2.+cfg["offsets_x"][det],+cfg["chipX"]/2.+cfg["offsets_x"][det], 100,-cfg["chipY"]/2.+cfg["offsets_y"][det],+cfg["chipY"]/2.+cfg["offsets_y"][det]) } )
        
        histos.update( { "h_big_cls_2D_"+det : ROOT.TH2D("h_big_cls_2D_"+det,";Fit x; Fit y",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )

        histos.update( { "h_Chi2fit_res_trk2cls_x_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_x_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins,-absRes,+absRes) } )
        histos.update( { "h_Chi2fit_res_trk2cls_y_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_y_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins,-absRes,+absRes) } )
    
        histos.update( { "h_Chi2fit_res_trk2tru_x_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_x_"+det,";"+det+" x_{trk}-x_{tru} [mm];Events",nResBins,-absRes,+absRes) } )
        histos.update( { "h_Chi2fit_res_trk2tru_y_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_y_"+det,";"+det+" y_{trk}-y_{tru} [mm];Events",nResBins,-absRes,+absRes) } )    
        
        histos.update( { "h_ncls_"+det          : ROOT.TH1D("h_ncls_"+det,";Number of clusters;Events",10,0,10) } )
        histos.update( { "h_cls_size_"+det      : ROOT.TH1D("h_cls_size_"+det,";Cluster size;Events",10,0.5,10.5) } )
        histos.update( { "h_cls_size_ncol_"+det : ROOT.TH1D("h_cls_size_ncol_"+det,";Cluster size in x;Events",10,0.5,10.5) } )
        histos.update( { "h_cls_size_nrow_"+det : ROOT.TH1D("h_cls_size_nrow_"+det,";Cluster size in y;Events",10,0.5,10.5) } )
                
        histos.update( { "h_ncls_masked_"+det          : ROOT.TH1D("h_ncls_masked_"+det,";Number of clusters;Events",10,0.5,10.5) } )
        histos.update( { "h_cls_size_masked_"+det      : ROOT.TH1D("h_cls_size_masked_"+det,";Cluster size;Events",10,0.5,10.5) } )
        histos.update( { "h_cls_size_ncol_masked_"+det : ROOT.TH1D("h_cls_size_ncol_masked_"+det,";Cluster size in x;Events",10,0.5,10.5) } )
        histos.update( { "h_cls_size_nrow_masked_"+det : ROOT.TH1D("h_cls_size_nrow_masked_"+det,";Cluster size in y;Events",10,0.5,10.5) } )
            
    for hname,hist in histos.items():
        hist.SetLineColor(ROOT.kBlack)
        hist.Sumw2()
    
    return histos


def book_alignment_histos(tfo):
    histos = {}
    tfo.cd()
    histos.update( {"hChi2dof":ROOT.TH1D("hChi2dof",";Original #chi^{2}/N_{DoF};Tracks",100,0,10)} )
    histos.update( {"hSVDchi2dof":ROOT.TH1D("hSVDchi2dof",";SVD #chi^{2}/N_{DoF};Tracks",100,0,10)} )
    histos.update( {"hTransform":ROOT.TH3D("hTransform",";x [mm];y [mm];#theta",int(cfg["alignmentbins"]["dx"]["bins"]),cfg["alignmentbins"]["dx"]["min"],cfg["alignmentbins"]["dx"]["max"],
                                                                           int(cfg["alignmentbins"]["dy"]["bins"]),cfg["alignmentbins"]["dy"]["min"],cfg["alignmentbins"]["dy"]["max"],
                                                                           int(cfg["alignmentbins"]["theta"]["bins"]),cfg["alignmentbins"]["theta"]["min"],cfg["alignmentbins"]["theta"]["max"])})
    return histos


def GetPixMatrix():
    h2D = {}
    for det in cfg["detectors"]:
        h2D.update( { det: ROOT.TH2D("h_pix_matrix_"+det,";x;y;A.U.",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max)  } )
        h2D[det].SetDirectory(0)
    return h2D


def fillPixOcc(det,pixels,masked,histos):
    for pix in pixels:
        i = histos["h_pix_occ_2D_"+det].FindBin(pix.x,pix.y)
        histos["h_pix_occ_1D_"+det].AddBinContent(i,1)
        histos["h_pix_occ_2D_"+det].Fill(pix.x,pix.y)
        if(i not in masked):
            histos["h_pix_occ_1D_masked_"+det].AddBinContent(i,1)
            histos["h_pix_occ_2D_masked_"+det].Fill(pix.x,pix.y)


def fillClsHists(det,clusters,masked,histos):
    histos["h_ncls_"+det].Fill(len(clusters))
    for c in clusters:
        noisy = False
        for pix in c.pixels:
            i = histos["h_pix_occ_2D_"+det].FindBin(pix.x,pix.y)
            if(i in masked):
                noisy = True
                break
        ### not masked
        histos["h_cls_size_"+det].Fill(len(c.pixels))
        histos["h_cls_size_ncol_"+det].Fill(c.dx)
        histos["h_cls_size_nrow_"+det].Fill(c.dy)
        histos["h_cls_occ_2D_"+det].Fill(c.xmm,c.ymm)
        if(not noisy):
            histos["h_cls_size_masked_"+det].Fill(len(c.pixels))
            histos["h_cls_size_ncol_masked_"+det].Fill(c.dx)
            histos["h_cls_size_nrow_masked_"+det].Fill(c.dy)
            histos["h_cls_occ_2D_masked_"+det].Fill(c.xmm,c.ymm)


def fillFitOcc(params,hname2,hname3,histos):
    for det in cfg["detectors"]:
        x,y,z = line(cfg["rdetectors"][det][2],params)
        histos[hname2+"_"+det].Fill(x,y)
        histos[hname3].Fill(x,y,z)


def fill_trk2cls_residuals(points,direction,centroid,hname,histos):
    for det in cfg["detectors"]:
        dx,dy = res_track2cluster(det,points,direction,centroid)
        histos[hname+"_x_"+det].Fill(dx)
        histos[hname+"_y_"+det].Fill(dy)


def fill_trk2vtx_residuals(vtx,direction,centroid,hname,histos):
    dxv,dyv = res_track2vertex(vtx,direction,centroid)
    histos[hname+"_x"].Fill(dxv)
    histos[hname+"_y"].Fill(dyv)


def fill_trk2tru_residuals(mcparticles,pdgIdMatch,points,direction,centroid,hname,histos):
    for det in cfg["detectors"]:
        dx,dy = res_track2truth(det,mcparticles,pdgIdMatch,points,direction,centroid)
        # print(dy,offsets_y[det])
        histos[hname+"_x_"+det].Fill(dx)
        histos[hname+"_y_"+det].Fill(dy)
