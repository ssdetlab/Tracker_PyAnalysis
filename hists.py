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
import errors
from errors import *
import counters
from counters import *

def GetLogBinning(nbins,xmin,xmax):
    logmin  = math.log10(xmin)
    logmax  = math.log10(xmax)
    logbinwidth = (logmax-logmin)/nbins
    # Bin edges
    xbins = [xmin,] #the lowest edge first
    for i in range(1,nbins+1):
        xbins.append( ROOT.TMath.Power( 10,(logmin + i*logbinwidth) ) )
    arrxbins = array.array("d", xbins)
    return arrxbins

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
nResBins = int(absRes*600)

### scaled chip size due to misalignments
chipXmin = -( cfg["chipX"]*(1.+cfg["lut_scaleX"]) )/2.
chipXmax = +( cfg["chipX"]*(1.+cfg["lut_scaleX"]) )/2.
chipYmin = -( cfg["chipY"]*(1.+cfg["lut_scaleY"]) )/2.
chipYmax = +( cfg["chipY"]*(1.+cfg["lut_scaleY"]) )/2.
nXchip = 600
nYchip = 300

trkarr = GetLogBinning(50,0.5,300)
ntrkarr = len(trkarr)-1

### book histos
def book_histos(tfo):
    histos = {}
    
    tfo.cd()
    for det in cfg["detectors"]: tfo.mkdir(det)
    tfo.cd()
    
    histos.update( { "h_events" : ROOT.TH1D("h_events",";;Events",1,0,1) } )
    histos["h_events"].GetXaxis().SetBinLabel(1,"")
    
    histos.update( { "h_errors" : ROOT.TH1D("h_errors",";;Events",len(ERRORS),0,len(ERRORS)) } )
    for b in range(1,len(ERRORS)+1):
        histos["h_errors"].GetXaxis().SetBinLabel(b,ERRORS[b-1])
        
    histos.update( { "h_counters" : ROOT.TH1D("h_counters",";;Frequency",len(COUNTERS),0,len(COUNTERS)) } )
    for b in range(1,len(COUNTERS)+1):
        histos["h_counters"].GetXaxis().SetBinLabel(b,COUNTERS[b-1])
    
    histos.update( { "h_cutflow"   : ROOT.TH1D("h_cutflow",";;Events",10,0,10) } )
    for b in range(1,len(cfg["cuts"])+1):
        cutname = cfg["cuts"][b-1]
        if(cutname=="#chi^{2}/N_{DoF}#leqX"): cutname = cutname.replace("X",str(cfg["cut_chi2dof"]))
        histos["h_cutflow"].GetXaxis().SetBinLabel(b,cutname)
    
    histos.update( { "h_3Dchi2err"      : ROOT.TH1D("h_3Dchi2err",";#chi^{2}/N_{dof};Tracks",200,0,absChi2) } )
    histos.update( { "h_3Dchi2err_all"  : ROOT.TH1D("h_3Dchi2err_all",";#chi^{2}/N_{dof};Tracks",500,0,absChi2*200) } )
    histos.update( { "h_3Dchi2err_full" : ROOT.TH1D("h_3Dchi2err_full",";#chi^{2}/N_{dof};Tracks",200,0,absChi2*10) } )
    histos.update( { "h_3Dchi2err_zoom" : ROOT.TH1D("h_3Dchi2err_zoom",";#chi^{2}/N_{dof};Tracks",200,0,absChi2/5.) } )
    histos.update( { "h_3Dchi2err_0to1" : ROOT.TH1D("h_3Dchi2err_0to1",";#chi^{2}/N_{dof};Tracks",200,0,1.) } )
    histos.update( { "h_npix"           : ROOT.TH1D("h_npix",";N_{pixels}/detector;Events",30,0,30) } )

    histos.update( { "h_Chi2fit_res_trk2vtx_x" : ROOT.TH1D("h_Chi2fit_res_trk2vtx_x",";x_{trk}-x_{vtx} [mm];Events",nResBins,-absRes,+absRes) } )
    histos.update( { "h_Chi2fit_res_trk2vtx_y" : ROOT.TH1D("h_Chi2fit_res_trk2vtx_y",";y_{trk}-y_{vtx} [mm];Events",nResBins,-absRes,+absRes) } )
    
    histos.update( { "h_nSeeds"               : ROOT.TH1D("h_nSeeds",";N_{seeds}/Event;Events",250,0,250) } )
    histos.update( { "h_nSeeds_log"           : ROOT.TH1D("h_nSeeds_log",";N_{seeds}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nSeeds_full"          : ROOT.TH1D("h_nSeeds_full",";N_{seeds}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nSeeds_mid"           : ROOT.TH1D("h_nSeeds_mid",";N_{seeds}/Event;Events",100,0,100) } )
    histos.update( { "h_nSeeds_zoom"          : ROOT.TH1D("h_nSeeds_zoom",";N_{seeds}/Event;Events",40,0,40) } )
    histos.update( { "h_nTracks"              : ROOT.TH1D("h_nTracks",";N_{tracks}/Event;Events",250,0,250) } )
    histos.update( { "h_nTracks_log"          : ROOT.TH1D("h_nTracks_log",";N_{tracks}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTracks_full"         : ROOT.TH1D("h_nTracks_full",";N_{tracks}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTracks_mid"          : ROOT.TH1D("h_nTracks_mid",";N_{tracks}/Event;Events",100,0,100) } )
    histos.update( { "h_nTracks_zoom"         : ROOT.TH1D("h_nTracks_zoom",";N_{tracks}/Event;Events",40,0,40) } )
    histos.update( { "h_nTracks_success"      : ROOT.TH1D("h_nTracks_success",";N_{tracks}/Event;Events",250,0,250) } )
    histos.update( { "h_nTracks_success_log"  : ROOT.TH1D("h_nTracks_success_log",";N_{tracks}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTracks_success_full" : ROOT.TH1D("h_nTracks_success_full",";N_{tracks}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTracks_success_mid"  : ROOT.TH1D("h_nTracks_success_mid",";N_{tracks}/Event;Events",100,0,100) } )
    histos.update( { "h_nTracks_success_zoom" : ROOT.TH1D("h_nTracks_success_zoom",";N_{tracks}/Event;Events",40,0,40) } )
    histos.update( { "h_nTracks_goodchi2"     : ROOT.TH1D("h_nTracks_goodchi2",";N_{tracks}/Event;Events",250,0,250) } )
    histos.update( { "h_nTracks_goodchi2_log" : ROOT.TH1D("h_nTracks_goodchi2_log",";N_{tracks}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTracks_goodchi2_full": ROOT.TH1D("h_nTracks_goodchi2_full",";N_{tracks}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTracks_goodchi2_mid" : ROOT.TH1D("h_nTracks_goodchi2_mid",";N_{tracks}/Event;Events",100,0,100) } )
    histos.update( { "h_nTracks_goodchi2_zoom": ROOT.TH1D("h_nTracks_goodchi2_zoom",";N_{tracks}/Event;Events",40,0,40) } )
    histos.update( { "h_nTracks_selected"     : ROOT.TH1D("h_nTracks_selected",";N_{tracks}/Event;Events",250,0,250) } )
    histos.update( { "h_nTracks_selected_log" : ROOT.TH1D("h_nTracks_selected_log",";N_{tracks}/Event;Events",ntrkarr,trkarr) } )
    histos.update( { "h_nTracks_selected_full": ROOT.TH1D("h_nTracks_selected_full",";N_{tracks}/Event;Events",2000,0,20000) } )
    histos.update( { "h_nTracks_selected_mid" : ROOT.TH1D("h_nTracks_selected_mid",";N_{tracks}/Event;Events",100,0,100) } )
    histos.update( { "h_nTracks_selected_zoom": ROOT.TH1D("h_nTracks_selected_zoom",";N_{tracks}/Event;Events",40,0,40) } )

    histos.update( { "h_Chi2_phi"             : ROOT.TH1D("h_Chi2_phi",";Chi2 fit: #phi;Tracks",100,-np.pi,+np.pi) } )
    histos.update( { "h_Chi2_theta"           : ROOT.TH1D("h_Chi2_theta",";Chi2 fit: #theta;Tracks",100,0,np.pi) } )
    histos.update( { "h_Chi2_theta_weighted"  : ROOT.TH1D("h_Chi2_theta_weighted",";Chi2 fit: #theta weighted;Tracks",100,0,np.pi) } )
    
    histos.update( { "h_tru_3D"   : ROOT.TH3D("h_tru_3D", ";x [mm] w/alignment;y [mm] w/alignment;z [mm]",100,chipXmin,chipXmax, 50,chipYmin,chipYmax, 50,-5,95) } )
    histos.update( { "h_cls_3D"   : ROOT.TH3D("h_cls_3D", ";x [mm] w/alignment;y [mm] w/alignment;z [mm]",100,chipXmin,chipXmax, 50,chipYmin,chipYmax, 50,-5,95) } )
    histos.update( { "h_trk_3D"   : ROOT.TH3D("h_trk_3D", ";x [mm] w/alignment;y [mm] w/alignment;z [mm]",100,chipXmin,chipXmax, 50,chipYmin,chipYmax, 50,-5,95) } )
    
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

        histos.update( { "h_errors_"+det : ROOT.TH1D("h_errors_"+det,";;Events",len(ERRORS),0,len(ERRORS)) } )
        for b in range(1,len(ERRORS)+1):
            histos["h_errors_"+det].GetXaxis().SetBinLabel(b,ERRORS[b-1])
        
        histos.update( { "h_pix_occ_1D_"+det        : ROOT.TH1D("h_pix_occ_1D_"+det,";Pixel;Hits",cfg["npix_x"]*cfg["npix_y"],1,cfg["npix_x"]*cfg["npix_y"]+1) } )
        histos.update( { "h_pix_occ_1D_masked_"+det : ROOT.TH1D("h_pix_occ_1D_masked_"+det,";Pixel;Hits",cfg["npix_x"]*cfg["npix_y"],1,cfg["npix_x"]*cfg["npix_y"]+1) } )
        histos.update( { "h_pix_occ_2D_"+det        : ROOT.TH2D("h_pix_occ_2D_"+det,";x;y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )
        histos.update( { "h_pix_occ_2D_masked_"+det : ROOT.TH2D("h_pix_occ_2D_masked_"+det,";x;y;Hits",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )
        
        histos.update( { "h_cls_occ_2D_"+det        : ROOT.TH2D("h_cls_occ_2D_"+det,";x [mm];y [mm];Clusters",cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2.) } )
        histos.update( { "h_cls_occ_2D_masked_"+det : ROOT.TH2D("h_cls_occ_2D_masked_"+det,";x [mm];y [mm];Clusters",cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2.) } )

        # histos.update( { "h_trk_occ_2D_"+det        : ROOT.TH2D("h_trk_occ_2D_"+det,";x [mm];y [mm];Tracks",       nXchip,chipXmin,chipXmax, nXchip,chipYmin,chipYmax) } )
        # histos.update( { "h_trk_occ_2D_masked_"+det : ROOT.TH2D("h_trk_occ_2D_masked_"+det,";x [mm];y [mm];Tracks",nXchip,chipXmin,chipXmax, nXchip,chipYmin,chipYmax) } )
        
        histos.update( { "h_trk_occ_2D_"+det : ROOT.TH2D("h_trk_occ_2D_"+det,";x [mm] w/alignment;y [mm] w/alignment;Tracks",nXchip,chipXmin,chipXmax, nXchip,chipYmin,chipYmax) } )

        histos.update( { "h_tru_occ_2D_"+det : ROOT.TH2D("h_tru_occ_2D_"+det,";x [mm];y [mm];Tracks",nXchip,chipXmin,chipXmax, nXchip,chipYmin,chipYmax) } )
        
        histos.update( { "h_ncls_"+det          : ROOT.TH1D("h_ncls_"+det,";Number of clusters;Events",1,0,1) } )
        histos.update( { "h_ncls_masked_"+det   : ROOT.TH1D("h_ncls_masked_"+det,";Number of clusters;Events",1,0,1) } )

        histos.update( { "h_cls_size_"+det        : ROOT.TH1D("h_cls_size_"+det,";Cluster size;Events",100,0.5,100.5) } )
        histos.update( { "h_cls_size_masked_"+det : ROOT.TH1D("h_cls_size_masked_"+det,";Cluster size;Events",100,0.5,100.5) } )
        
        histos.update( { "h_cls_size_zoom_"+det        : ROOT.TH1D("h_cls_size_zoom_"+det,";Cluster size;Events",11,0.5,11.5) } )
        histos.update( { "h_cls_size_zoom_masked_"+det : ROOT.TH1D("h_cls_size_zoom_masked_"+det,";Cluster size;Events",11,0.5,11.5) } )

        histos.update( { "h_cls_size_ncol_"+det :        ROOT.TH1D("h_cls_size_ncol_"+det,";Cluster size in x;Events",50,0.5,50.5) } )
        histos.update( { "h_cls_size_ncol_masked_"+det : ROOT.TH1D("h_cls_size_ncol_masked_"+det,";Cluster size in x;Events",50,0.5,50.5) } )
        histos.update( { "h_cls_size_nrow_"+det :        ROOT.TH1D("h_cls_size_nrow_"+det,";Cluster size in y;Events",50,0.5,50.5) } )                
        histos.update( { "h_cls_size_nrow_masked_"+det : ROOT.TH1D("h_cls_size_nrow_masked_"+det,";Cluster size in y;Events",50,0.5,50.5) } )
        
        histos.update( { "h_csize_vs_x_"+det        : ROOT.TH2D("h_csize_vs_x_"+det,";x;Cluster size;Clusters",cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., 100,0.5,100.5) } )
        histos.update( { "h_csize_vs_x_masked_"+det : ROOT.TH2D("h_csize_vs_x_masked_"+det,";x;Cluster size;Clusters",cfg["npix_x"]+1,-cfg["chipX"]/2.,+cfg["chipX"]/2., 100,0.5,100.5) } )
        histos.update( { "h_csize_vs_y_"+det        : ROOT.TH2D("h_csize_vs_y_"+det,";y;Cluster size;Clusters",cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2., 100,0.5,100.5) } )
        histos.update( { "h_csize_vs_y_masked_"+det : ROOT.TH2D("h_csize_vs_y_masked_"+det,";y;Cluster size;Clusters",cfg["npix_y"]+1,-cfg["chipY"]/2.,+cfg["chipY"]/2., 100,0.5,100.5) } )
        
        histos.update( { "h_response_x_"+det : ROOT.TH1D("h_response_x_"+det,";#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",30,-5,+5) } )
        histos.update( { "h_response_y_"+det : ROOT.TH1D("h_response_y_"+det,";#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",30,-5,+5) } )
        histos.update( { "h_response_x_vs_csize_"+det : ROOT.TH2D("h_response_x_vs_csize_"+det,";Cluster size in x;#frac{x_{trk}-x_{cls}}{#sigma(x_{cls})};Tracks",10,1,11, 30,-5,+5) } )
        histos.update( { "h_response_y_vs_csize_"+det : ROOT.TH2D("h_response_y_vs_csize_"+det,";Cluster size in y;#frac{y_{trk}-y_{cls}}{#sigma(y_{cls})};Tracks",10,1,11, 30,-5,+5) } )
        
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
                
        histos.update( { "h_big_cls_2D_"+det : ROOT.TH2D("h_big_cls_2D_"+det,";Fit x; Fit y",pix_x_nbins,pix_x_min,pix_x_max, pix_y_nbins,pix_y_min,pix_y_max) } )

        histos.update( { "h_Chi2fit_res_trk2cls_x_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_x_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins,-absRes,+absRes) } )
        histos.update( { "h_Chi2fit_res_trk2cls_y_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_y_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins,-absRes,+absRes) } )

        histos.update( { "h_Chi2fit_res_trk2cls_pass_x_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_pass_x_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins,-absRes,+absRes) } )
        histos.update( { "h_Chi2fit_res_trk2cls_pass_y_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_pass_y_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins,-absRes,+absRes) } )

        histos.update( { "h_Chi2fit_res_trk2cls_x_mid_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_x_mid_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins*2,-absRes*5,+absRes*5) } )
        histos.update( { "h_Chi2fit_res_trk2cls_y_mid_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_y_mid_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins*2,-absRes*5,+absRes*5) } )

        histos.update( { "h_Chi2fit_res_trk2cls_pass_x_mid_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_pass_x_mid_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins*2,-absRes*5,+absRes*5) } )
        histos.update( { "h_Chi2fit_res_trk2cls_pass_y_mid_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_pass_y_mid_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins*2,-absRes*5,+absRes*5) } )
        
        histos.update( { "h_Chi2fit_res_trk2cls_x_full_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_x_full_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins*2,-absRes*50,+absRes*50) } )
        histos.update( { "h_Chi2fit_res_trk2cls_y_full_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_y_full_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins*2,-absRes*50,+absRes*50) } )

        histos.update( { "h_Chi2fit_res_trk2cls_pass_x_full_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_pass_x_full_"+det,";"+det+" x_{trk}-x_{cls} [mm];Events",nResBins*2,-absRes*50,+absRes*50) } )
        histos.update( { "h_Chi2fit_res_trk2cls_pass_y_full_"+det : ROOT.TH1D("h_Chi2fit_res_trk2cls_pass_y_full_"+det,";"+det+" y_{trk}-y_{cls} [mm];Events",nResBins*2,-absRes*50,+absRes*50) } )
    
        histos.update( { "h_Chi2fit_res_trk2tru_x_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_x_"+det,";"+det+" x_{trk}-x_{tru} [mm];Events",nResBins,-absRes,+absRes) } )
        histos.update( { "h_Chi2fit_res_trk2tru_y_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_y_"+det,";"+det+" y_{trk}-y_{tru} [mm];Events",nResBins,-absRes,+absRes) } )    
        
        histos.update( { "h_Chi2fit_res_trk2tru_x_mid_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_x_mid_"+det,";"+det+" x_{trk}-x_{tru} [mm];Events",nResBins*2,-absRes*5,+absRes*5) } )
        histos.update( { "h_Chi2fit_res_trk2tru_y_mid_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_y_mid_"+det,";"+det+" y_{trk}-y_{tru} [mm];Events",nResBins*2,-absRes*5,+absRes*5) } )
        
        histos.update( { "h_Chi2fit_res_trk2tru_x_full_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_x_full_"+det,";"+det+" x_{trk}-x_{tru} [mm];Events",nResBins*2,-absRes*50,+absRes*50) } )
        histos.update( { "h_Chi2fit_res_trk2tru_y_full_"+det : ROOT.TH1D("h_Chi2fit_res_trk2tru_y_full_"+det,";"+det+" y_{trk}-y_{tru} [mm];Events",nResBins*2,-absRes*50,+absRes*50) } )    
            
    for hname,hist in histos.items():
        hist.SetLineColor(ROOT.kBlack)
        hist.Sumw2()
    
    return histos


# def book_alignment_histos(tfo):
#     histos = {}
#     tfo.cd()
#     histos.update( {"hChi2dof":ROOT.TH1D("hChi2dof",";Original #chi^{2}/N_{DoF};Tracks",100,0,10)} )
#     histos.update( {"hSVDchi2dof":ROOT.TH1D("hSVDchi2dof",";SVD #chi^{2}/N_{DoF};Tracks",100,0,10)} )
#     histos.update( {"hTransform":ROOT.TH3D("hTransform",";x [mm];y [mm];#theta",int(cfg["alignmentbins"]["dx"]["bins"]),cfg["alignmentbins"]["dx"]["min"],cfg["alignmentbins"]["dx"]["max"],
#                                                                            int(cfg["alignmentbins"]["dy"]["bins"]),cfg["alignmentbins"]["dy"]["min"],cfg["alignmentbins"]["dy"]["max"],
#                                                                            int(cfg["alignmentbins"]["theta"]["bins"]),cfg["alignmentbins"]["theta"]["min"],cfg["alignmentbins"]["theta"]["max"])})
#     return histos


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
    for c in clusters:
        noisy = False
        if(not cfg["skipmasking"]):
            for pix in c.pixels:
                i = histos["h_pix_occ_2D_"+det].FindBin(pix.x,pix.y)
                if(i in masked):
                    noisy = True
                    break
        histos["h_cls_3D"].Fill( c.xmm,c.ymm,c.zmm )
        ### not masked
        histos["h_ncls_"+det].Fill(0.5,len(clusters))
        # histos["h_cls_occ_2D_"+det].Fill(c.xmm,c.ymm)
        histos["h_cls_occ_2D_"+det].Fill(c.xmm0,c.ymm0)
        histos["h_cls_size_"+det].Fill(c.n)
        histos["h_cls_size_zoom_"+det].Fill(c.n)
        histos["h_cls_size_ncol_"+det].Fill(c.nx)
        histos["h_cls_size_nrow_"+det].Fill(c.ny)
        histos["h_csize_vs_x_"+det].Fill(c.xmm0,c.n)
        histos["h_csize_vs_y_"+det].Fill(c.ymm0,c.n)
        if(not noisy):
            histos["h_ncls_masked_"+det].Fill(0.5,len(clusters))
            # histos["h_cls_occ_2D_masked_"+det].Fill(c.xmm,c.ymm)
            histos["h_cls_occ_2D_masked_"+det].Fill(c.xmm0,c.ymm0)
            histos["h_cls_size_masked_"+det].Fill(c.n)
            histos["h_cls_size_zoom_masked_"+det].Fill(c.n)
            histos["h_cls_size_ncol_masked_"+det].Fill(c.nx)
            histos["h_cls_size_nrow_masked_"+det].Fill(c.ny)
            histos["h_csize_vs_x_masked_"+det].Fill(c.xmm0,c.n)
            histos["h_csize_vs_y_masked_"+det].Fill(c.ymm0,c.n)


def fillFitOcc(params,hname2,hname3,histos):
    for det in cfg["detectors"]:
        x,y,z = line(cfg["rdetectors"][det][2],params)
        histos[hname2+"_"+det].Fill(x,y)
        histos[hname3].Fill(x,y,z)


def fill_trk2cls_response(points,errors,direction,centroid,nxs,nys,trkchi2dof,hname,histos,chi2threshold=100.):
    for idet,det in enumerate(cfg["detectors"]):
        if(trkchi2dof>chi2threshold): continue
        resx,resy = res_track2clusterErr(det,points,errors,direction,centroid)
        histos[hname+"_x_"+det].Fill(resx)
        histos[hname+"_y_"+det].Fill(resy)
        xsize = nxs[idet]
        ysize = nys[idet]
        histos[hname+"_x_vs_csize_"+det].Fill(xsize,resx)
        histos[hname+"_y_vs_csize_"+det].Fill(ysize,resy)


def fill_trk2cls_residuals(points,direction,centroid,trkchi2dof,hname,histos,chi2threshold=1.):
    for det in cfg["detectors"]:
        if(trkchi2dof>chi2threshold): continue
        dx,dy = res_track2cluster(det,points,direction,centroid)
        histos[hname+"_x_"+det].Fill(dx)
        histos[hname+"_y_"+det].Fill(dy)
        histos[hname+"_x_mid_"+det].Fill(dx)
        histos[hname+"_y_mid_"+det].Fill(dy)
        histos[hname+"_x_full_"+det].Fill(dx)
        histos[hname+"_y_full_"+det].Fill(dy)


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
        histos[hname+"_y_"+det].Fill(dy)
        histos[hname+"_x_mid_"+det].Fill(dx)
        histos[hname+"_y_mid_"+det].Fill(dy)
        histos[hname+"_x_full_"+det].Fill(dx)
        histos[hname+"_y_full_"+det].Fill(dy)
