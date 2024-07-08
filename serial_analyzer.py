#!/usr/bin/python
import os
import os.path
import math
import time
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
# from skspatial.objects import Line, Sphere
# from skspatial.plotting import plot_3d

import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
argus = parser.parse_args()
configfile = argus.conf

import config
from config import *
### must be called here (first) and only once!
init_config(configfile,True)

import utils
from utils import *
import svd_fit
from svd_fit import *
import chi2_fit
from chi2_fit import *
import hists
from hists import *

import objects
from objects import *
import pixels
from pixels import *
import clusters
from clusters import *
import truth
from truth import *
import noise
from noise import *
import candidate
from candidate import *



ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)

print("-----------------------------------------------------------------------------------")
print("Must first add DetectorEvent lib:")
print("export LD_LIBRARY_PATH=$HOME/tracker_event:$LD_LIBRARY_PATH")
print("-----------------------------------------------------------------------------------")

print("---- start loading libs")
### see https://root.cern/manual/python/
ROOT.gInterpreter.AddIncludePath('~/tracker_event/')
ROOT.gSystem.Load('libtrk_event_dict.dylib')
print("---- finish loading libs")

    
#####################################################################################
#####################################################################################
#####################################################################################


def GetTree(tfilename):
    tfile = ROOT.TFile(tfilename,"READ")
    ttree = None
    if(not cfg["isMC"]): ttree = tfile.Get("MyTree")
    else:
        if(cfg["isCVRroot"]): ttree = tfile.Get("Pixel")
        else:                 ttree = tfile.Get("tt")
    print("Events in tree:",ttree.GetEntries())
    if(cfg["nmax2process"]>0): print("Will process only",cfg["nmax2process"],"events")
    return tfile,ttree


def RunNoiseScan(tfilename,tfnoisename):
    tfilenoise = ROOT.TFile(tfnoisename,"RECREATE")
    tfilenoise.cd()
    h1D_noise       = {}
    h2D_noise       = {}
    for det in cfg["detectors"]:
        h1D_noise.update( { det:ROOT.TH1D("h_noisescan_pix_occ_1D_"+det,";Pixel;Hits",cfg["npix_x"]*cfg["npix_y"],1,cfg["npix_x"]*cfg["npix_y"]+1) } )
        h2D_noise.update( { det:ROOT.TH2D("h_noisescan_pix_occ_2D_"+det,";Pixel;Hits",cfg["npix_x"]+1,-0.5,cfg["npix_x"]+0.5, cfg["npix_y"]+1,-0.5,cfg["npix_y"]+0.5) } )

    ### get the tree
    tfile,ttree = GetTree(tfilename)
    
    nprocevents = 0
    for evt in ttree:
        if(cfg["nmax2process"]>0 and nprocevents>cfg["nmax2process"]): break
        ### get the pixels
        n_active_planes,pixels = get_all_pixles(evt,h2D_noise,cfg["isCVRroot"])
        for det in cfg["detectors"]:
            for pix in pixels[det]:
                i = h2D_noise[det].FindBin(pix.x,pix.y)
                h1D_noise[det].AddBinContent(i,1)
                h2D_noise[det].Fill(pix.x,pix.y)
        if(nprocevents%1000==0 and nprocevents>0): print("event:",nprocevents)
        nprocevents += 1
    ### finish
    tfilenoise.Write()
    tfilenoise.Close()
    print("Noise scan histos saved in:",tfnoisename)



#####################################################################################
#####################################################################################
#####################################################################################


def Run(tfilename,tfnoisename,tfo,histos):
    ### get the tree
    tfile,ttree = GetTree(tfilename)
    
    truth_tree = None
    if(cfg["isCVRroot"]):
        truth_tree = tfile.Get("MCParticle")
    
    masked = GetNoiseMask(tfnoisename)
    if(cfg["isMC"]):
        for det in cfg["detectors"]:
            masked.update( {det:{}} )
    
    hPixMatix = GetPixMatrix()
    
    largest_clster = {}
    for det in cfg["detectors"]:
        largest_clster.update({det:Cls([],det)})
    
    nprocevents = 0
    norigevents = -1
    ientry      = 0 ### impoortant!!
    for evt in ttree:
        ### before anything else
        if(cfg["nmax2process"]>0 and nprocevents>cfg["nmax2process"]): break
        histos["h_events"].Fill(0.5)
        histos["h_cutflow"].Fill( cfg["cuts"].index("All") )
        norigevents += 1
        
        ### truth particles
        mcparticles = {}
        if(cfg["isCVRroot"] and truth_tree is not None):
            mcparticles = get_truth_cvr(truth_tree,ientry)
            for det in cfg["detectors"]:
                xtru,ytru,ztru = getTruPos(det,mcparticles,cfg["pdgIdMatch"])
                histos["h_tru_3D"].Fill( xtru,ytru,ztru )
                histos["h_tru_occ_2D_"+det].Fill( xtru,ytru )
        ientry += 1 ### important!
        
        ### get the pixels
        n_active_planes, pixels = get_all_pixles(evt,hPixMatix,cfg["isCVRroot"])
        for det in cfg["detectors"]:
            fillPixOcc(det,pixels[det],masked[det],histos) ### fill pixel occupancy
        # if(n_active_planes!=len(cfg["detectors"])): continue ### CUT!!! ### TODO uncomment!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/det}>0") )
        
        ### check if there's no noise
        pixels_save = {}  ### to hold a copy of all pixels
        for det in cfg["detectors"]:
            goodpixels = getGoodPixels(det,pixels[det],masked[det],hPixMatix[det])  ### TODO uncomment and comment-out next line!!!
            pixels[det] = goodpixels
            pixels_save.update({det:goodpixels.copy()})

        ### run clustering
        clusters = {}
        nclusters = 0
        for det in cfg["detectors"]:
            det_clusters = GetAllClusters(pixels[det],det)
            clusters.update( {det:det_clusters} )
            # if(len(det_clusters)==1): nclusters += 1 ### TODO uncomment and comment-out next line!!!
            if(len(det_clusters)>0): nclusters += 1 ### TODO comment-out this line and uncomment the one above!!! 
        
        ### find the largest cluster
        for det in cfg["detectors"]:
            for c in clusters[det]:
                if(len(c.pixels)>len(largest_clster[det].pixels)): largest_clster[det] = c
        
        ### exactly one cluster per layer
        # if(nclusters!=len(cfg["detectors"])): continue ### CUT!!! #TODO uncomment this line!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{cls/det}==1") )
        for det in cfg["detectors"]:
            fillClsHists(det,clusters[det],masked[det],histos)
            # histos["h_cls_3D"].Fill( clusters[det][0].xmm,clusters[det][0].ymm,clusters[det][0].zmm ) #TODO uncvomment

        #######################################
        continue #TODO remove this line, I just have it for stopping after the clustering step
        #######################################

        ### diagnostics, also with truth
        if(len(mcparticles)>0 and cfg["doDiagnostics"]):
            for det in cfg["detectors"]:
                print("-------"+det+":")
                for pr in mcparticles[det]:
                    print("["+str(mcparticles[det].index(pr))+"]:",pr)
                for px in pixels_save[det]:
                    print(px)
                for cl in clusters[det]:
                    print(cl)


        # ### TODO: trying to see what is the characteristics of events with 3 single-pixel clusters alone
        # singlepixel = True
        # for det in detectors:
        #     if(len(clusters[det][0].pixels)>1):
        #         singlepixel = False
        #         break
        # if(not singlepixel): continue
        

        ### run tracking
        vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
        evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
        best_Chi2 = {}
        best_value_Chi2 = +1e10
        ### loop on all cluster combinations
        
        # for i0 in range(len(clusters["ALPIDE_0"])):
        #     for i1 in range(len(clusters["ALPIDE_1"])):
        #         for i2 in range(len(clusters["ALPIDE_2"])):
        #             for i3 in range(len(clusters["ALPIDE_3"])):
        #
        #                 clsx  = {"ALPIDE_3":clusters["ALPIDE_3"][i3].xmm,  "ALPIDE_2":clusters["ALPIDE_2"][i2].xmm,  "ALPIDE_1":clusters["ALPIDE_1"][i1].xmm,  "ALPIDE_0":clusters["ALPIDE_0"][i0].xmm}
        #                 clsy  = {"ALPIDE_3":clusters["ALPIDE_3"][i3].ymm,  "ALPIDE_2":clusters["ALPIDE_2"][i2].ymm,  "ALPIDE_1":clusters["ALPIDE_1"][i1].ymm,  "ALPIDE_0":clusters["ALPIDE_0"][i0].ymm}
        #                 clsz  = {"ALPIDE_3":clusters["ALPIDE_3"][i3].zmm,  "ALPIDE_2":clusters["ALPIDE_2"][i2].zmm,  "ALPIDE_1":clusters["ALPIDE_1"][i1].zmm,  "ALPIDE_0":clusters["ALPIDE_0"][i0].zmm}
        #                 clsdx = {"ALPIDE_3":clusters["ALPIDE_3"][i3].dxmm, "ALPIDE_2":clusters["ALPIDE_2"][i2].dxmm, "ALPIDE_1":clusters["ALPIDE_1"][i1].dxmm, "ALPIDE_0":clusters["ALPIDE_0"][i0].dxmm}
        #                 clsdy = {"ALPIDE_3":clusters["ALPIDE_3"][i3].dymm, "ALPIDE_2":clusters["ALPIDE_2"][i2].dymm, "ALPIDE_1":clusters["ALPIDE_1"][i1].dymm, "ALPIDE_0":clusters["ALPIDE_0"][i0].dymm}
        #
        #                 #############################
        #                 ### to check timing #TODO ###
        #                 #############################
        #
        #                 points_SVD,errors_SVD = SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
        #                 points_Chi2,errors_Chi2 = Chi2_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
        #                 chisq,ndof,direction_Chi2,centroid_Chi2,params_Chi2,success_Chi2 = fit_3d_chi2err(points_Chi2,errors_Chi2)
        #
        #                 # chisq,ndof,direction_Chi2,centroid_Chi2 = fit_3d_SVD(points_SVD,errors_SVD)
        #                 # success_Chi2 = True
        #                 # params_Chi2 = [1,0,0,0]
        #
        #                 chi2ndof_Chi2 = chisq/ndof if(ndof>0) else 99999
        #                 if(success_Chi2 and chi2ndof_Chi2<best_value_Chi2): ### happens only when success_Chi2==True
        #                     best_value_Chi2 = chi2ndof_Chi2
        #                     best_Chi2.update( {"svd_points":points_SVD} )
        #                     best_Chi2.update( {"points":points_Chi2} )
        #                     best_Chi2.update( {"errors":errors_Chi2} )
        #                     best_Chi2.update( {"direction":direction_Chi2} )
        #                     best_Chi2.update( {"centroid":centroid_Chi2} )
        #                     best_Chi2.update( {"chi2ndof":chi2ndof_Chi2} )
        #                     best_Chi2.update( {"params":params_Chi2} )
        

        clsx   = {}
        clsy   = {}
        clsz   = {}
        clsdx  = {}
        clsdy  = {}
        for det in cfg["detectors"]:
            clsx.update( {det:clusters[det][0].xmm} )
            clsy.update( {det:clusters[det][0].ymm} )
            clsz.update( {det:clusters[det][0].zmm} )
            clsdx.update( {det:clusters[det][0].dxmm} )
            clsdy.update( {det:clusters[det][0].dymm} )

        #############################
        ### to check timing #TODO ###
        #############################
        
        points_SVD,errors_SVD = SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
        points_Chi2,errors_Chi2 = Chi2_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
        chisq,ndof,direction_Chi2,centroid_Chi2,params_Chi2,success_Chi2 = fit_3d_chi2err(points_Chi2,errors_Chi2)
        
        # chisq,ndof,direction_Chi2,centroid_Chi2 = fit_3d_SVD(points_SVD,errors_SVD)
        # success_Chi2 = True
        # params_Chi2 = [1,0,0,0]
        
        chi2ndof_Chi2 = chisq/ndof if(ndof>0) else 99999
        if(success_Chi2 and chi2ndof_Chi2<best_value_Chi2): ### happens only when success_Chi2==True
            best_value_Chi2 = chi2ndof_Chi2
            best_Chi2.update( {"svd_points":points_SVD} )
            best_Chi2.update( {"points":points_Chi2} )
            best_Chi2.update( {"errors":errors_Chi2} )
            best_Chi2.update( {"direction":direction_Chi2} )
            best_Chi2.update( {"centroid":centroid_Chi2} )
            best_Chi2.update( {"chi2ndof":chi2ndof_Chi2} )
            best_Chi2.update( {"params":params_Chi2} )
            

        
        ### fit successful
        passFit = (len(best_Chi2)>0)
        if(passFit):
            ### get the best Chi2 fit
            points_SVD     = best_Chi2["svd_points"]
            points_Chi2    = best_Chi2["points"]
            errors_Chi2    = best_Chi2["errors"]
            direction_Chi2 = best_Chi2["direction"]
            centroid_Chi2  = best_Chi2["centroid"]
            chi2ndof_Chi2  = best_Chi2["chi2ndof"]
            params_Chi2    = best_Chi2["params"]
            if(cfg["doplot"]):
                for det in cfg["detectors"]:
                    dx,dy = res_track2cluster(det,points_SVD,direction_Chi2,centroid_Chi2)
                    print(det,"-->",dx,dy)
                plot_3d_chi2err(norigevents,points_Chi2,params_Chi2,cfg["doplot"])
                print("")

            ### fill some histos
            histos["h_3Dchi2err"].Fill(chi2ndof_Chi2)
            histos["h_3Dchi2err_full"].Fill(chi2ndof_Chi2)
            histos["h_3Dchi2err_zoom"].Fill(chi2ndof_Chi2)
            histos["h_cutflow"].Fill( cfg["cuts"].index("Fitted") )
            
            dx = direction_Chi2[0]
            dy = direction_Chi2[1]
            dz = direction_Chi2[2]
            theta = np.arctan(np.sqrt(dx*dx+dy*dy)/dz)
            phi   = np.arctan(dy/dx)
            histos["h_Chi2_phi"].Fill(phi)
            histos["h_Chi2_theta"].Fill(theta)
            if(abs(np.sin(theta))>1e-10): histos["h_Chi2_theta_weighted"].Fill( theta,abs(1/(2*np.pi*np.sin(theta))) )
            if(chi2ndof_Chi2<=10): histos["h_cutflow"].Fill( cfg["cuts"].index("#chi^{2}/N_{DoF}#leq10") )
            ### Chi2 track to cluster residuals
            fill_trk2cls_residuals(points_SVD,direction_Chi2,centroid_Chi2,"h_Chi2fit_res_trk2cls",histos)
            ### Chi2 track to truth residuals
            if(cfg["isMC"]): fill_trk2tru_residuals(mcparticles,cfg["pdgIdMatch"],points_SVD,direction_Chi2,centroid_Chi2,"h_Chi2fit_res_trk2tru",histos)
            ### Chi2 fit points on laters
            fillFitOcc(params_Chi2,"h_fit_occ_2D", "h_fit_3D",histos)
            ### Chi2 track to vertex residuals
            if(cfg["doVtx"]): fill_trk2vtx_residuals(vtx,direction_Chi2,centroid_Chi2,"h_Chi2fit_res_trk2vtx",histos)

            ### fill cluster size vs true position
            if(cfg["isCVRroot"] and truth_tree is not None):
                for det in cfg["detectors"]:
                    xtru,ytru,ztru = getTruPos(det,mcparticles,cfg["pdgIdMatch"])
                    wgt = clusters[det][0].n
                    posx = ((xtru-cfg["pix_x"]/2.)%(2*cfg["pix_x"]))
                    posy = ((ytru-cfg["pix_y"]/2.)%(2*cfg["pix_y"]))
                    histos["h_csize_vs_trupos"].Fill(posx,posy,wgt)
                    histos["h_ntrks_vs_trupos"].Fill(posx,posy)
                    histos["h_csize_vs_trupos_"+det].Fill(posx,posy,wgt)
                    histos["h_ntrks_vs_trupos_"+det].Fill(posx,posy)
                    ### divide into smaller sizes
                    strcsize = str(wgt) if(wgt<5) else "n"
                    histos["h_csize_"+strcsize+"_vs_trupos"].Fill(posx,posy,wgt)
                    histos["h_ntrks_"+strcsize+"_vs_trupos"].Fill(posx,posy)
                    histos["h_csize_"+strcsize+"_vs_trupos_"+det].Fill(posx,posy,wgt)
                    histos["h_ntrks_"+strcsize+"_vs_trupos_"+det].Fill(posx,posy)
                
                    # if(det=="ALPIDE_0"): print("Size:",wgt,"Tru:",xtru,ytru,"Residuals:",(xtru%pix_x),(ytru%pix_y))
        
        ### event counter
        if(nprocevents%10==0 and nprocevents>0): print("processed event:",nprocevents,"out of",norigevents,"events read")
        nprocevents += 1


    #######################
    ### post processing ###
    #######################
    
    
    ### cluster mean size vs position
    tfo.cd()
    hname = "h_csize_vs_trupos"
    hnewname = hname.replace("csize","mean")
    hdenname = hname.replace("csize","ntrks")
    histos.update( {hnewname:histos[hname].Clone(hnewname)} )
    histos[hnewname].Divide(histos[hdenname])
    for det in cfg["detectors"]:
        tfo.cd(det)
        hname = "h_csize_vs_trupos_"+det
        hnewname = hname.replace("csize","mean")
        hdenname = hname.replace("csize","ntrks")
        histos.update( {hnewname:histos[hname].Clone(hnewname)} )
        histos[hnewname].Divide(histos[hdenname])
    for j in range(1,6):
        tfo.cd()
        strcsize = str(j) if(j<5) else "n"
        hname = "h_csize_"+strcsize+"_vs_trupos"
        hnewname = hname.replace("csize","mean")
        hdenname = hname.replace("csize","ntrks")
        histos.update( {hnewname:histos[hname].Clone(hnewname)} )
        histos[hnewname].Divide(histos[hdenname])
        for det in cfg["detectors"]:
            tfo.cd(det)
            hname = "h_csize_"+strcsize+"_vs_trupos_"+det
            hnewname = hname.replace("csize","mean")
            hdenname = hname.replace("csize","ntrks")
            histos.update( {hnewname:histos[hname].Clone(hnewname)} )
            histos[hnewname].Divide(histos[hdenname])
    
    ### largest clusters
    for det in cfg["detectors"]:
        for pix in largest_clster[det].pixels:
            histos["h_big_cls_2D_"+det].Fill(pix.x,pix.y)
        

#############################################################################
#############################################################################
#############################################################################

# get the start time
st = time.time()

tfilenamein = cfg["inputfile"]
tfnoisename = tfilenamein.replace(".root","_noise.root")
isnoisefile = os.path.isfile(os.path.expanduser(tfnoisename))
print("Running on:",tfilenamein)
if(cfg["doNoiseScan"]):
    print("Noise run file exists?:",isnoisefile)
    if(isnoisefile):
        redonoise = input("Noise file exists - do you want to rederive it?[y/n]:")
        if(redonoise=="y" or redonoise=="Y"):
            RunNoiseScan(tfilenamein,tfnoisename)
            # print("before GetNoiseMask 1")
            masked = GetNoiseMask(tfnoisename)
            # print("after GetNoiseMask 1")
        else:
            print("Option not understood - please try again.")
    else:
        RunNoiseScan(tfilenamein,tfnoisename)
        # print("before GetNoiseMask 2")
        masked = GetNoiseMask(tfnoisename)
        # print("after GetNoiseMask 2")
    # print("before quit")
    quit()
else:
    if(not isnoisefile):
        print("Noise file",tfnoisename,"not found")
        print("Generate first by setting doNoiseScan=True")
        quit()

# print("before output")

tfilenameout = tfilenamein.replace(".root","_histograms.root")
tfo = ROOT.TFile(tfilenameout,"RECREATE")
tfo.cd()
# book_histos(tfo,absRes,absChi2)
histos = book_histos(tfo)
Run(tfilenamein,tfnoisename,tfo,histos)
tfo.cd()
tfo.Write()
tfo.Close()

# get the end time and the execution time
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')



