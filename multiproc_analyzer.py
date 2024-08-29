#!/usr/bin/python
import multiprocessing as mp
# from multiprocessing.pool import ThreadPool
import time
import os
import os.path
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
# from skspatial.objects import Line, Sphere
# from skspatial.plotting import plot_3d
import pickle

import argparse
parser = argparse.ArgumentParser(description='serial_analyzer.py...')
parser.add_argument('-conf', metavar='config file', required=True,  help='full path to config file')
parser.add_argument('-dbg',  metavar='debug with single proc?', required=False,  help='debug with single proc?[0/1]')
argus = parser.parse_args()
configfile = argus.conf
debug = True if(argus.dbg is not None and argus.dbg=="1") else False

import config
from config import *
### must be called here (first) and only once!
init_config(configfile,False)

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
import errors
from errors import *


ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)


### see https://root.cern/manual/python
print("---- start loading libs")
if(os.uname()[1]=="wisett"):
    print("On DAQ PC (linux): must first add DetectorEvent lib:")
    print("export LD_LIBRARY_PATH=$HOME/work/eudaq/lib:$LD_LIBRARY_PATH")
    ROOT.gInterpreter.AddIncludePath('../eudaq/user/stave/module/inc/')
    ROOT.gInterpreter.AddIncludePath('../eudaq/user/stave/hardware/inc/')
    ROOT.gSystem.Load('libeudaq_det_event_dict.so')
else:
    print("On mac: must first add DetectorEvent lib:")
    print("export LD_LIBRARY_PATH=$PWD/DetectorEvent/20240705:$LD_LIBRARY_PATH")
    ROOT.gInterpreter.AddIncludePath('DetectorEvent/20240705/')
    ROOT.gSystem.Load('libtrk_event_dict.dylib')
print("---- finish loading libs")

###############################################################
###############################################################
###############################################################

### defined below as global
allhistos = {}


def GetTree(tfilename):
    tfile = ROOT.TFile(tfilename,"READ")
    ttree = None
    if(not cfg["isMC"]): ttree = tfile.Get("MyTree")
    else:
        if(cfg["isCVRroot"]): ttree = tfile.Get("Pixel")
        else:          ttree = tfile.Get("tt")
    return tfile,ttree


def analyze(tfilenamein,irange,evt_range,masked):
    lock = mp.Lock()
    lock.acquire()
    
    ### important
    sufx = "_"+str(irange)
    
    ### open the pickle:
    picklename = tfilenamein.replace(".root","_"+str(irange)+".pkl")
    fpickle = open(os.path.expanduser(picklename),"wb")
    
    ### histos
    tfoname = tfilenamein.replace(".root","_multiprocess_histograms"+sufx+".root")
    tfo = ROOT.TFile(tfoname,"RECREATE")
    tfo.cd()
    histos = book_histos(tfo)
    for name,hist in histos.items():
        hist.SetName(name+sufx)
        hist.SetDirectory(0)
    
    ### get the tree
    tfile,ttree = GetTree(tfilenamein)
    truth_tree = tfile.Get("MCParticle") if(cfg["isCVRroot"]) else None
    
    ### needed below
    hPixMatix = GetPixMatrix()
    
    ### start the event loop
    ievt_start = evt_range[0]
    ievt_end   = evt_range[-1]
    eventslist = []
    for ievt in range(ievt_start,ievt_end+1):
        ### get the event
        ttree.GetEntry(ievt)
        
        ### get the trigger number
        trigger_number = ttree.event.trg_n

        ### all events...
        histos["h_events"].Fill(0.5)
        histos["h_cutflow"].Fill( cfg["cuts"].index("All") )
        
        ### check event errors
        nerrors,errors = check_errors(ttree)
        if(nerrors>0):
            wgt = 1./float(len(cfg["detectors"]))
            for det in cfg["detectors"]:
                for err in errors[det]:
                    b = ERRORS.index(err)+1
                    histos["h_errors"].AddBinContent(b,wgt)
                    histos["h_errors_"+det].AddBinContent(b)
            continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("0Err") )
        
        ### truth particles
        mcparticles = get_truth_cvr(truth_tree,ievt) if(cfg["isCVRroot"] and truth_tree is not None) else {}
        if(cfg["isCVRroot"] and truth_tree is not None):
            for det in cfg["detectors"]:
                xtru,ytru,ztru = getTruPos(det,mcparticles,cfg["pdgIdMatch"])
                histos["h_tru_3D"].Fill( xtru,ytru,ztru )
                histos["h_tru_occ_2D_"+det].Fill( xtru,ytru )

        ### get the pixels
        n_active_staves, n_active_chips, pixels = get_all_pixles(ttree,hPixMatix,cfg["isCVRroot"])
        for det in cfg["detectors"]:
            fillPixOcc(det,pixels[det],masked[det],histos) ### fill pixel occupancy
        if(n_active_chips!=len(cfg["detectors"])): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/det}>0") )
        
        ### get the non-noisy pixels but this will get emptied during clustering so also keep a duplicate
        pixels_save = {}
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
        if(nclusters<len(cfg["detectors"])): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{cls/det}>0") )
        
        ### run the seeding
        seeder = HoughSeeder(clusters,ievt)
        seed_cuslters = seeder.seed_clusters
        histos["h_nSeeds"].Fill(seeder.summary["nseeds"])
        if(seeder.summary["nplanes"]<len(cfg["detectors"]) or seeder.summary["nseeds"]<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{seeds}>0") )
        
        ### prepare the clusters for the fit
        det0 = cfg["detectors"][0]
        det1 = cfg["detectors"][1]
        det2 = cfg["detectors"][2]
        det3 = cfg["detectors"][3]
        seeds = []
        for c0 in seed_cuslters[det0]:
            for c1 in seed_cuslters[det1]:
                for c2 in seed_cuslters[det2]:
                    for c3 in seed_cuslters[det3]:
                        ###
                        seed = { "x":{}, "y":{}, "z":{}, "dx":{}, "dy":{} }
                        ###
                        seed["x"].update({det0:c0.xmm})
                        seed["y"].update({det0:c0.ymm})
                        seed["z"].update({det0:c0.zmm})
                        seed["dx"].update({det0:c0.dxmm})
                        seed["dy"].update({det0:c0.dymm})
                        ###
                        seed["x"].update({det1:c1.xmm})
                        seed["y"].update({det1:c1.ymm})
                        seed["z"].update({det1:c1.zmm})
                        seed["dx"].update({det1:c1.dxmm})
                        seed["dy"].update({det1:c1.dymm})
                        ###
                        seed["x"].update({det2:c2.xmm})
                        seed["y"].update({det2:c2.ymm})
                        seed["z"].update({det2:c2.zmm})
                        seed["dx"].update({det2:c2.dxmm})
                        seed["dy"].update({det2:c2.dymm})
                        ###
                        seed["x"].update({det3:c3.xmm})
                        seed["y"].update({det3:c3.ymm})
                        seed["z"].update({det3:c3.zmm})
                        seed["dx"].update({det3:c3.dxmm})
                        seed["dy"].update({det3:c3.dymm})
                        ###
                        seeds.append(seed)
        histos["h_nSeeds"].Fill(len(seeds))
        # print(f"Event #{ievt} with {len(seeds)} seeds for {len(clusters[det0])} in {det0}, {len(clusters[det1])} in {det1}, {len(clusters[det2])} in {det2}, {len(clusters[det3])} in {det3}")
        # if(len(seeds)>100):
        #     print(f"{det0}:")
        #     for i,c in enumerate(clusters[det0]): print(f"  [{i}]: {c}")
        #     print(f"{det1}:")
        #     for i,c in enumerate(clusters[det1]): print(f"  [{i}]: {c}")
        #     print(f"{det2}:")
        #     for i,c in enumerate(clusters[det2]): print(f"  [{i}]: {c}")
        #     print(f"{det3}:")
        #     for i,c in enumerate(clusters[det3]): print(f"  [{i}]: {c}")
        #     print("")

        ### get the event tracks
        vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
        evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
        
        ### loop over all seeds:
        tracks = []
        n_successful_tracks = 0
        n_goodchi2_tracks   = 0
        for seed in seeds:
            clsx = seed["x"]
            clsy = seed["y"]
            clsz = seed["z"]
            clsdx = seed["dx"]
            clsdy = seed["dy"]
            
            points_SVD,errors_SVD = SVD_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
            points_Chi2,errors_Chi2 = Chi2_candidate(clsx,clsy,clsz,clsdx,clsdy,vtx,evtx)
            chisq,ndof,direction,centroid,params,success = fit_3d_chi2err(points_Chi2,errors_Chi2)
            chisq_SVD,ndof_SVD,direction_SVD,centroid_SVD = fit_3d_SVD(points_SVD,errors_SVD)
            chi2ndof = chisq/ndof if(ndof>0) else 99999
            track = Track(clusters,points_Chi2,errors_Chi2,chisq,ndof,direction,centroid,params,success)
            tracks.append(track)
            if(success):                      n_successful_tracks += 1
            if(chi2ndof<=cfg["cut_chi2dof"]): n_goodchi2_tracks += 1
            
            # if(n_active_chips==4): print(f"n_goodchi2_tracks={n_goodchi2_tracks}, chi2ndof={chi2ndof}")

            histos["h_3Dchi2err"].Fill(chi2ndof)
            histos["h_3Dchi2err_full"].Fill(chi2ndof)
            histos["h_3Dchi2err_zoom"].Fill(chi2ndof)
            histos["h_Chi2_phi"].Fill(track.phi)
            histos["h_Chi2_theta"].Fill(track.theta)
            if(abs(np.sin(track.theta))>1e-10): histos["h_Chi2_theta_weighted"].Fill( track.theta,abs(1/(2*np.pi*np.sin(track.theta))) )
            
            ### Chi2 track to cluster residuals
            fill_trk2cls_residuals(points_SVD,direction,centroid,"h_Chi2fit_res_trk2cls",histos)
            # fill_trk2cls_residuals(points_SVD,direction_SVD,centroid_SVD,"h_Chi2fit_res_trk2cls",histos)
            ### Chi2 track to truth residuals
            if(cfg["isMC"]): fill_trk2tru_residuals(mcparticles,cfg["pdgIdMatch"],points_SVD,direction,centroid,"h_Chi2fit_res_trk2tru",histos)
            ### Chi2 fit points on laters
            fillFitOcc(params,"h_fit_occ_2D", "h_fit_3D",histos)
            ### Chi2 track to vertex residuals
            if(cfg["doVtx"]): fill_trk2vtx_residuals(vtx,direction,centroid,"h_Chi2fit_res_trk2vtx",histos)
        histos["h_nTracks"].Fill(len(tracks))
        histos["h_nTracks_success"].Fill( n_successful_tracks )
        histos["h_nTracks_goodchi2"].Fill( n_goodchi2_tracks )
        
        if(n_successful_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("Fitted") )
        
        if(n_goodchi2_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("#chi^{2}/N_{DoF}#leqX") )
        
        ### plot
        #if(ievt==12196 or ievt==12209 or ievt==12243 or ievt==34581 or ievt==34599 or ievt==12717 or ievt==23093 or ievt==33427 or ievt==10923 or ievt==24):
        fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_{ievt}.pdf")
        seeder.plot_seeder(fevtdisplayname)
        plot_event(ievt,fevtdisplayname,clusters,tracks,chi2threshold=cfg["cut_chi2dof"])
        
        ### fill the event data and add to events
        eventslist.append( Event(pixels_save,clusters,tracks,mcparticles) )
        
    ### end
    pickle.dump(eventslist, fpickle, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpickle.close()
    print("Worker of",irange,"is done!")
    lock.release()
    return histos



def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)

def collect_histos(histos):
    ### https://www.machinelearningplus.com/python/parallel-processing-python/
    global allhistos ### defined above!!!
    for name,hist in allhistos.items():
        hist.Add(histos[name])


if __name__ == "__main__":
    # get the start time
    st = time.time()
    
    # print config once
    show_config()
    
    ### architecture depndent
    nCPUs = mp.cpu_count()
    print("nCPUs available:",nCPUs)
    print("nCPUs configured:",cfg["nCPU"])
    if(cfg["nCPU"]<1):
        print("nCPU config cannot be <1, quitting")
        quit()
    elif(cfg["nCPU"]>=1 and cfg["nCPU"]<=nCPUs):
        nCPUs = cfg["nCPU"]
    else:
        print("nCPU config cannot be greater than",nCPUs,", quitting")
        quit()

    ### Create a pool of workers
    pool = mp.Pool(nCPUs)
    
    ### make event display dir
    paths = cfg["inputfile"].split("/")
    evtdspdir = ""
    for i in range(len(paths)-1): evtdspdir += paths[i]+"/"
    evtdspdir += "event_displays"
    ROOT.gSystem.Exec(f"/bin/mkdir -p {evtdspdir}")
    
    # Parallelize the analysis
    tfilenamein = cfg["inputfile"]
    tfnoisename = tfilenamein.replace(".root","_noise.root")
    masked = GetNoiseMask(tfnoisename)
    
    ### the output histos
    tfilenameout = tfilenamein.replace(".root","_multiprocess_histograms.root")
    tfo = ROOT.TFile(tfilenameout,"RECREATE")
    tfo.cd()
    allhistos = book_histos(tfo)
    
    ### start the loop
    print("\nStarting the loop:")
    tfile0,ttree0 = GetTree(tfilenamein)
    neventsintree = ttree0.GetEntries()
    # nevents = cfg["nmax2processMP"] if(cfg["nmax2processMP"]>0 and cfg["nmax2processMP"]<=neventsintree) else neventsintree
    nevents = neventsintree
    if(cfg["nmax2processMP"]>0 and cfg["nmax2processMP"]<=neventsintree):
        nevents = cfg["nmax2processMP"]
        print("Going to analyze only",nevents,"events out of the",neventsintree,"available in the tree")
    else:
        print("config nmax2processMP =",cfg["nmax2processMP"],"--> will analyze all events in the tree:",neventsintree)
    bundle = nCPUs
    fullrange = range(nevents)
    ranges = np.array_split(fullrange,bundle)
    for irng,rng in enumerate(ranges):
        print("Submitting range["+str(irng)+"]:",rng[0],"...",rng[-1])
        if(debug):
            histos = analyze(tfilenamein,irng,rng,masked)
        else:
            pool.apply_async(analyze, args=(tfilenamein,irng,rng,masked), callback=collect_histos, error_callback=collect_errors)
    
    ### Wait for all the workers to finish
    pool.close()
    pool.join()
    
    
    ### remove worker root files (they are anyhow empty out of the worker scope)
    for irng,rng in enumerate(ranges):
        sufx = "_"+str(irng)
        tfoname = tfilenamein.replace(".root","_multiprocess_histograms"+sufx+".root")
        tfoname = os.path.expanduser(tfoname)
        if os.path.isfile(tfoname):
            os.remove(tfoname)
            print("file deleted:",tfoname)
        else:
            print("Error: %s file not found" % tfoname)


    #######################
    ### post processing ###
    #######################
    tfo.cd()
    ### cluster mean size vs position
    hname = "h_csize_vs_trupos"
    hnewname = hname.replace("csize","mean")
    hdenname = hname.replace("csize","ntrks")
    allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
    allhistos[hnewname].Divide(allhistos[hdenname])
    for det in cfg["detectors"]:
        tfo.cd(det)
        hname = "h_csize_vs_trupos_"+det
        hnewname = hname.replace("csize","mean")
        hdenname = hname.replace("csize","ntrks")
        allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
        allhistos[hnewname].Divide(allhistos[hdenname])
    for j in range(1,6):
        tfo.cd()
        strcsize = str(j) if(j<5) else "n"
        hname = "h_csize_"+strcsize+"_vs_trupos"
        hnewname = hname.replace("csize","mean")
        hdenname = hname.replace("csize","ntrks")
        allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
        allhistos[hnewname].Divide(allhistos[hdenname])
        for det in cfg["detectors"]:
            tfo.cd(det)
            hname = "h_csize_"+strcsize+"_vs_trupos_"+det
            hnewname = hname.replace("csize","mean")
            hdenname = hname.replace("csize","ntrks")
            allhistos.update( {hnewname:allhistos[hname].Clone(hnewname)} )
            allhistos[hnewname].Divide(allhistos[hdenname])

    
    # Save the histograms to a file
    tfo.Write()
    tfo.Close()
    
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
