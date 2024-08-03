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
import hough_seeder
from hough_seeder import *



ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)

print("-----------------------------------------------------------------------------------")
print("Must first add DetectorEvent lib:")
print("export LD_LIBRARY_PATH=$PWD/DetectorEvent/20240705:$LD_LIBRARY_PATH")
print("-----------------------------------------------------------------------------------")

print("---- start loading libs")
### see https://root.cern/manual/python/
ROOT.gInterpreter.AddIncludePath('DetectorEvent/20240705/')
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
        n_active_staves, n_active_chips, pixels = get_all_pixles(evt,h2D_noise,cfg["isCVRroot"])
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
    
    nprocevents = 0
    norigevents = -1
    for ientry,evt in enumerate(ttree):
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
        
        ### get the pixels
        n_active_staves, n_active_chips, pixels = get_all_pixles(evt,hPixMatix,cfg["isCVRroot"])
        for det in cfg["detectors"]:
            fillPixOcc(det,pixels[det],masked[det],histos) ### fill pixel occupancy
        if(n_active_chips!=len(cfg["detectors"])): continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/det}>0") )
        
        ### check if there's no noise
        pixels_save = {}  ### to hold a copy of all pixels
        for det in cfg["detectors"]:
            goodpixels = getGoodPixels(det,pixels[det],masked[det],hPixMatix[det])
            pixels[det] = goodpixels
            pixels_save.update({det:goodpixels.copy()})

        ### run clustering
        clusters = {}
        nclusters = 0
        for det in cfg["detectors"]:
            det_clusters = GetAllClusters(pixels[det],det)
            clusters.update( {det:det_clusters} )
            fillClsHists(det,clusters[det],masked[det],histos)
            if(len(det_clusters)>0): nclusters += 1
        ### at least one cluster per layer
        if(nclusters<len(cfg["detectors"])): continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{cls/det}>0") )
        
        # interesting events:
        # 12196 --> 900 --> 0 seeds, OK
        # 12209 --> 96 --> 0 seeds, OK
        # 12243 --> 16 --> 16 seeds, OK?
        # 34581 --> 16 --> 16 seeds, OK? --> nice 2-muon event!!
        # 34599 --> 15 --> 0/1/2 seeds --> very depending on the binning of the 2D histo!!
        # 12717 --> 32 --> 16 seeds --> very depending on the binning of the 2D histo!!
        # 23093 --> 160 --> 1/16 seed --> nice 2-muon event, very depending on the binning of the 2D histo!!
        # 33427 --> 180 --> 0/1 seeds OK
        # 10923 --> 360 --> 2 seeds, OK
        # 24 --> 1200 --> 24 seeds, OK
        
        ### run the seeding
        # if(ientry!=33427): continue
        # ncomb = 1
        # for det in cfg["detectors"]: ncomb *= len(clusters[det])
        seeder = HoughSeeder(clusters,ientry)
        # seeder.plot(str(ientry))
        # print(f"ientry:{ientry} --> ncomb={ncomb} --> nseeds={seeder.nseeds}, nplanes={seeder.nplanes}, seed_clusters_per_detector={seeder.seed_clusters_per_detector}, found_intrscts_zx_and_zy={seeder.found_intrscts_zx_and_zy}")
        seed_cuslters = seeder.seed_clusters
        seeder.clear_h2Freq() ### TODO: very important!
        if(seeder.summary["nplanes"]<len(cfg["detectors"]) or seeder.summary["nseeds"]<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{seeds}>0") )
        histos["h_nSeeds"].Fill(seeder.summary["nseeds"])
        # quit()
        
        ### run tracking
        vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
        evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
        best_Chi2 = {}
        best_value_Chi2 = +1e10
        tracks = []
        ### loop on all cluster combinations
        for i0 in range(len(seed_cuslters["ALPIDE_0"])):
            for i1 in range(len(seed_cuslters["ALPIDE_1"])):
                for i2 in range(len(seed_cuslters["ALPIDE_2"])):
                    for i3 in range(len(seed_cuslters["ALPIDE_3"])):

                        clsx  = {"ALPIDE_3":seed_cuslters["ALPIDE_3"][i3].xmm,  "ALPIDE_2":seed_cuslters["ALPIDE_2"][i2].xmm,  "ALPIDE_1":seed_cuslters["ALPIDE_1"][i1].xmm,  "ALPIDE_0":seed_cuslters["ALPIDE_0"][i0].xmm}
                        clsy  = {"ALPIDE_3":seed_cuslters["ALPIDE_3"][i3].ymm,  "ALPIDE_2":seed_cuslters["ALPIDE_2"][i2].ymm,  "ALPIDE_1":seed_cuslters["ALPIDE_1"][i1].ymm,  "ALPIDE_0":seed_cuslters["ALPIDE_0"][i0].ymm}
                        clsz  = {"ALPIDE_3":seed_cuslters["ALPIDE_3"][i3].zmm,  "ALPIDE_2":seed_cuslters["ALPIDE_2"][i2].zmm,  "ALPIDE_1":seed_cuslters["ALPIDE_1"][i1].zmm,  "ALPIDE_0":seed_cuslters["ALPIDE_0"][i0].zmm}
                        clsdx = {"ALPIDE_3":seed_cuslters["ALPIDE_3"][i3].dxmm, "ALPIDE_2":seed_cuslters["ALPIDE_2"][i2].dxmm, "ALPIDE_1":seed_cuslters["ALPIDE_1"][i1].dxmm, "ALPIDE_0":seed_cuslters["ALPIDE_0"][i0].dxmm}
                        clsdy = {"ALPIDE_3":seed_cuslters["ALPIDE_3"][i3].dymm, "ALPIDE_2":seed_cuslters["ALPIDE_2"][i2].dymm, "ALPIDE_1":seed_cuslters["ALPIDE_1"][i1].dymm, "ALPIDE_0":seed_cuslters["ALPIDE_0"][i0].dymm}

                        #############################
                        ### to check timing #TODO ###
                        #############################

                        points_SVD,errors_SVD = SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
                        points_Chi2,errors_Chi2 = Chi2_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
                        chisq,ndof,direction_Chi2,centroid_Chi2,params_Chi2,success_Chi2 = fit_3d_chi2err(points_Chi2,errors_Chi2)
                        
                        track = Track(clusters,points_Chi2,errors_Chi2,chisq,ndof,direction_Chi2,centroid_Chi2,params_Chi2,success_Chi2)
                        tracks.append(track)

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

        ### plot
        if(ientry==12196 or ientry==12209 or ientry==12243 or ientry==34581 or ientry==34599 or ientry==12717 or ientry==23093 or ientry==33427 or ientry==10923 or ientry==24):
            fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_{ientry}.pdf")
            plot_event(ientry,fevtdisplayname,clusters,tracks,chi2threshold=1.)
        if(ientry==24): quit()

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
            if(chi2ndof_Chi2<=5): histos["h_cutflow"].Fill( cfg["cuts"].index("#chi^{2}/N_{DoF}#leq5") )
            ### Chi2 track to cluster residuals
            fill_trk2cls_residuals(points_SVD,direction_Chi2,centroid_Chi2,"h_Chi2fit_res_trk2cls",histos)
            ### Chi2 track to truth residuals
            if(cfg["isMC"]): fill_trk2tru_residuals(mcparticles,cfg["pdgIdMatch"],points_SVD,direction_Chi2,centroid_Chi2,"h_Chi2fit_res_trk2tru",histos)
            ### Chi2 fit points on laters
            fillFitOcc(params_Chi2,"h_fit_occ_2D", "h_fit_3D",histos)
            ### Chi2 track to vertex residuals
            if(cfg["doVtx"]): fill_trk2vtx_residuals(vtx,direction_Chi2,centroid_Chi2,"h_Chi2fit_res_trk2vtx",histos)

            # ### fill cluster size vs true position
            # if(cfg["isCVRroot"] and truth_tree is not None):
            #     for det in cfg["detectors"]:
            #         xtru,ytru,ztru = getTruPos(det,mcparticles,cfg["pdgIdMatch"])
            #         wgt = clusters[det][0].n
            #         posx = ((xtru-cfg["pix_x"]/2.)%(2*cfg["pix_x"]))
            #         posy = ((ytru-cfg["pix_y"]/2.)%(2*cfg["pix_y"]))
            #         histos["h_csize_vs_trupos"].Fill(posx,posy,wgt)
            #         histos["h_ntrks_vs_trupos"].Fill(posx,posy)
            #         histos["h_csize_vs_trupos_"+det].Fill(posx,posy,wgt)
            #         histos["h_ntrks_vs_trupos_"+det].Fill(posx,posy)
            #         ### divide into smaller sizes
            #         strcsize = str(wgt) if(wgt<5) else "n"
            #         histos["h_csize_"+strcsize+"_vs_trupos"].Fill(posx,posy,wgt)
            #         histos["h_ntrks_"+strcsize+"_vs_trupos"].Fill(posx,posy)
            #         histos["h_csize_"+strcsize+"_vs_trupos_"+det].Fill(posx,posy,wgt)
            #         histos["h_ntrks_"+strcsize+"_vs_trupos_"+det].Fill(posx,posy)
                
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
            masked = GetNoiseMask(tfnoisename)
        else:
            print("Option not understood - please try again.")
    else:
        RunNoiseScan(tfilenamein,tfnoisename)
        masked = GetNoiseMask(tfnoisename)
    print("###################################")
    print("### FINISHED RUNNING NOISE SCAN ###")
    print("### CHANGE doNoiseScan TO 0 IN ####")
    print("### THE CONFIG FILE AND RERUN #####")
    print("### THE WITH THE SAME COMMAND #####")
    print("###################################")
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



