#!/usr/bin/python
import multiprocessing as mp
# from multiprocessing.pool import ThreadPool
import time
import datetime
import os
import os.path
import math
import subprocess
import array
import numpy as np
import ROOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.optimize import curve_fit
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
import evtdisp
from evtdisp import *
import counters
from counters import *
import selections
from selections import *


ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)


###############################################################
###############################################################
###############################################################

if(cfg["isMC"]):
    # print("Building the classes for MC")
    ### declare the data tree and its classes
    ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
    ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
    ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
    ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
    ### declare the meta-data tree and its classes
    ROOT.gROOT.ProcessLine("struct run_meta_data  { Int_t run_number; Double_t run_start; Double_t run_end; };" )


###############################################################
###############################################################
###############################################################

### defined below as global
allhistos = {}

def dump_pixels(fpklname,pixels):
    fpkl = open(fpklname,"wb")
    data = {}
    for det in cfg["detectors"]:
        flat_pixels = []
        for pix in pixels[det]:
            flat_pixels.append( {"x":pix.x, "y":pix.y, "xmm":pix.xmm, "ymm":pix.ymm, "zmm":pix.zmm} )
        data.update( {det:flat_pixels} )
    pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()

def GetTree(tfilename):
    tfile = ROOT.TFile(tfilename,"READ")
    ttree = tfile.Get("MyTree")
    nevents = ttree.GetEntries()
    return tfile,ttree,nevents

def analyze(tfilenamein,irange,evt_range,masked,badtrigs):
    lock = mp.Lock()
    lock.acquire()
    
    ### important
    sufx = "_"+str(irange)
    
    ### the metadata:
    tfmeta = ROOT.TFile(tfilenamein,"READ")
    tmeta = tfmeta.Get("MyTreeMeta")
    runnumber = -1
    starttime = -1
    endtime   = -1
    duration  = -1
    if(tmeta is not None):
        try:
            nmeta = tmeta.GetEntries()
            tmeta.GetEntry(0)
            runnumber = tmeta.run_meta_data.run_number
            ts_start  = tmeta.run_meta_data.run_start
            starttime = get_human_timestamp(ts_start)
            if(nmeta>1): tmeta.GetEntry(nmeta-1)
            ts_end    = tmeta.run_meta_data.run_end
            endtime   = get_human_timestamp(ts_end)
            duration  = get_run_length(ts_start,ts_end)
        except:
            print("Problem with Meta tree.")
            runnumber = get_run_from_file(tfilenamein) #TODO: can also be taken from the event tree itself later
    meta = Meta(runnumber,starttime,endtime,duration)
    # tfmeta.Close()
    
    ### open the pickle:
    if(not cfg["skiptracking"]):
        picklename = tfilenamein.replace(".root","_"+str(irange)+".pkl")
        fpickle = open(os.path.expanduser(picklename),"wb")
    
    ### histos
    tfoname = tfilenamein.replace(".root",f'{cfg["hfilesufx"]}{sufx}.root')
    tfo = ROOT.TFile(tfoname,"RECREATE")
    tfo.cd()
    histos = book_histos(tfo)
    for name,hist in histos.items():
        hist.SetName(name+sufx)
        hist.SetDirectory(0)
    
    ### get the tree
    tfile,ttree,neventsall = GetTree(tfilenamein)
    # truth_tree = tfile.Get("MCParticle") if(cfg["isCVRroot"]) else None
    
    ### needed below
    hPixMatix = GetPixMatrix()
    
    ### start the event loop
    ievt_start = evt_range[0]
    ievt_end   = evt_range[-1]
    eventslist = []
    for ievt in range(ievt_start,ievt_end+1):
        ### get the event
        ttree.GetEntry(ievt)
        
        ### get the trigger number and time stamps
        trigger         = ttree.event.trg_n
        timestamp_begin = ttree.event.ts_begin
        timestamp_end   = ttree.event.ts_end

        ### append the envent no-matter-what:
        eventslist.append( Event(meta,trigger,timestamp_begin,timestamp_end) )

        ### all events...
        histos["h_events"].Fill(0.5)
        histos["h_cutflow"].Fill( cfg["cuts"].index("All") )

        ### skip bad triggers...
        if(not cfg["isMC"] and cfg["runtype"]=="beam"):
            if(int(trigger) in badtrigs): continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("BeamQC") )
        
        ### check event errors
        nerrors,errors = check_errors(ttree)
        eventslist[len(eventslist)-1].set_event_errors(errors)
        if(nerrors>0):
            wgt = 1./float(len(cfg["detectors"]))
            for det in cfg["detectors"]:
                for err in errors[det]:
                    b = ERRORS.index(err)+1
                    histos["h_errors"].AddBinContent(b,wgt)
                    histos["h_errors_"+det].AddBinContent(b)
            continue
        histos["h_cutflow"].Fill( cfg["cuts"].index("0Err") )
        
        
        # ### truth particles
        # mcparticles = get_truth_cvr(truth_tree,ievt) if(cfg["isCVRroot"] and truth_tree is not None) else {}
        # if(cfg["isCVRroot"] and truth_tree is not None):
        #     for det in cfg["detectors"]:
        #         xtru,ytru,ztru = getTruPos(det,mcparticles,cfg["pdgIdMatch"])
        #         histos["h_tru_3D"].Fill( xtru,ytru,ztru )
        #         histos["h_tru_occ_2D_"+det].Fill( xtru,ytru )

        ### get the pixels
        n_active_staves, n_active_chips, pixels = get_all_pixles(ttree,hPixMatix)
        sprnt = f"ievt={ievt}: active_chips={n_active_chips} -->"
        for det in cfg["detectors"]:
            sprnt += f" Npixels[{det}]={len(pixels[det])},"
            fillPixOcc(det,pixels[det],masked[det],histos) ### fill pixel occupancy
        print(sprnt)
        
        
        ### non-empty events
        if(n_active_chips==0): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("Non-empty") )
    
        
        ### all layers are active
        if(n_active_chips!=len(cfg["detectors"])): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/det}>0") )
        
        
        ### spatial ROI cut
        ROI = { "ix":{"min":cfg["cut_ROI_xmin"],"max":cfg["cut_ROI_xmax"]}, "iy":{"min":cfg["cut_ROI_ymin"],"max":cfg["cut_ROI_ymax"]} }
        n_active_staves, n_active_chips, pixels = get_all_pixles(ttree,hPixMatix,ROI)
        sprnt = f"ievt={ievt} in_ROI_chips={n_active_chips} -->"
        for det in cfg["detectors"]:
            sprnt += f" Npixels[{det}]={len(pixels[det])},"
        print(sprnt)
        if(n_active_chips!=len(cfg["detectors"])): continue  ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{hits/det}^{ROI}>0") )
        # dump_pixels(f"pixels_evt_{ievt}.pkl",pixels)
        
        
        ### get the non-noisy pixels but this will get emptied during clustering so also keep a duplicate
        pixels_save = {}
        for det in cfg["detectors"]:
            if(cfg["skipmasking"]):
                pixels_save.update({det:pixels[det].copy()})
            else:
                pixels[det] = getGoodPixels(det,pixels[det],masked[det],hPixMatix[det])
                pixels_save.update({det:goodpixels.copy()})
        eventslist[len(eventslist)-1].set_event_pixels(pixels_save)


        ### run clustering
        clusters = {}
        nclusters = 0
        sprnt = f"ievt={ievt}:"
        for det in cfg["detectors"]:
            det_clusters = BFS_GetAllClusters(pixels[det],det)
            clusters.update( {det:det_clusters} )
            fillClsHists(det,clusters[det],masked[det],histos)
            if(len(det_clusters)>0): nclusters += 1
            sprnt += f" Nclusters[{det}]={len(det_clusters)},"
        print(sprnt)
        eventslist[len(eventslist)-1].set_event_clusters(clusters)
        ### at least one cluster per layer
        if(nclusters<len(cfg["detectors"])): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{cls/det}>0") )


        #####################################
        if(cfg["skiptracking"]): continue ###
        #####################################

        
        ### run the seeding
        seeder = HoughSeeder(clusters,ievt)
        # #################
        
        nSeeds = seeder.nseeds
        histos["h_nSeeds"].Fill(nSeeds)
        histos["h_nSeeds_log"].Fill(nSeeds if(nSeeds>0) else 0.11)
        histos["h_nSeeds_full"].Fill(nSeeds)
        histos["h_nSeeds_mid"].Fill(nSeeds)
        histos["h_nSeeds_zoom"].Fill(nSeeds)
        if(nSeeds<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("N_{seeds}>0") )        
        
        ### prepare the clusters for the fit
        seeds = []
        for iseed,seed in enumerate(seeder.seeds):
            tunnelid = seeder.tnlid[iseed]
            trkseed = TrackSeed(seed,tunnelid,clusters)
            seeds.append(trkseed)
        del seeder
        eventslist[len(eventslist)-1].set_event_seeds(seeds)

        ### get the event tracks
        vtx  = [cfg["xVtx"],cfg["yVtx"],cfg["zVtx"]]    if(cfg["doVtx"]) else []
        evtx = [cfg["exVtx"],cfg["eyVtx"],cfg["ezVtx"]] if(cfg["doVtx"]) else []
        
        ### loop over all seeds:
        tracks = []
        n_tracks            = 0
        n_successful_tracks = 0
        n_goodchi2_tracks   = 0
        n_selected_tracks   = 0
        for seed in seeds:
            ### get the points
            xclserr = seed.xsize if(cfg["use_large_clserr_for_algnmnt"]) else seed.dx
            yclserr = seed.ysize if(cfg["use_large_clserr_for_algnmnt"]) else seed.dy
            points_SVD, errors_SVD  = SVD_candidate(seed.x,seed.y,seed.z,xclserr,yclserr,vtx,evtx)
            points_Chi2,errors_Chi2 = Chi2_candidate(seed.x,seed.y,seed.z,xclserr,yclserr,vtx,evtx)
            chisq     = None
            ndof      = None
            direction = None
            centroid  = None
            params    = None
            success   = None
            par_guess = None
            ### svd fit
            if("SVD" in cfg["fit_method"]):
                chisq,ndof,direction,centroid = fit_3d_SVD(points_SVD,errors_SVD)
                params = get_pars_from_centroid_and_direction(centroid,direction)
                par_guess = params
                success = True
            ### chi2 fit
            if("CHI2" in cfg["fit_method"]):
                chisq,ndof,direction,centroid,params,success = fit_3d_chi2err(points_Chi2,errors_Chi2,par_guess)
            ### prepae the track clusters
            trkcls = {}
            for idet,det in enumerate(cfg["detectors"]):
                icls = seed.clsids[idet]
                trkcls.update({det:clusters[det][icls]})
            ### set the track
            track = Track(trkcls,points_SVD,errors_SVD,chisq,ndof,direction,centroid,params,success)
            tracks.append(track)
            n_tracks += 1
            
            ### require good chi2, pointing to the pdc window, inclined up as a positron
            chi2ndof = chisq/ndof if(ndof>0) else 99999
            pass_fit       = (success and chi2ndof<=cfg["cut_chi2dof"])
            pass_selection = (pass_fit and pass_geoacc_selection(track))
            if(success):        n_successful_tracks += 1
            if(pass_fit):       n_goodchi2_tracks += 1
            if(pass_selection): n_selected_tracks += 1

            histos["h_3Dchi2err"].Fill(chi2ndof)
            histos["h_3Dchi2err_all"].Fill(chi2ndof)
            histos["h_3Dchi2err_full"].Fill(chi2ndof)
            histos["h_3Dchi2err_zoom"].Fill(chi2ndof)
            histos["h_3Dchi2err_0to1"].Fill(chi2ndof)
            histos["h_Chi2_phi"].Fill(track.phi)
            histos["h_Chi2_theta"].Fill(track.theta)
            if(abs(np.sin(track.theta))>1e-10): histos["h_Chi2_theta_weighted"].Fill( track.theta,abs(1/(2*np.pi*np.sin(track.theta))) )
            
            ### Chi2 track to cluster residuals
            fill_trk2cls_residuals(points_SVD,direction,centroid,chi2ndof,"h_Chi2fit_res_trk2cls",histos)
            fill_trk2cls_residuals(points_SVD,direction,centroid,chi2ndof,"h_Chi2fit_res_trk2cls_pass",histos,chi2threshold=cfg["cut_chi2dof"])
            ### response (residuals over cluster error)
            nxs = []
            nys = []
            for idet,det in enumerate(cfg["detectors"]):
                nxs.append(trkcls[det].nx)
                nys.append(trkcls[det].ny)
            fill_trk2cls_response(points_SVD,errors_SVD,direction,centroid,nxs,nys,chi2ndof,"h_response",histos,chi2threshold=cfg["cut_chi2dof"])
            ### fit points occupancy
            if(pass_selection): fillFitOcc(params,"h_trk_occ_2D", "h_trk_3D",histos)
            ### track to vertex residuals
            if(cfg["doVtx"]): fill_trk2vtx_residuals(vtx,direction,centroid,"h_Chi2fit_res_trk2vtx",histos)
            ### Chi2 track to truth residuals
            # if(cfg["isMC"]): fill_trk2tru_residuals(mcparticles,cfg["pdgIdMatch"],points_SVD,direction,centroid,"h_Chi2fit_res_trk2tru",histos)
        
        eventslist[len(eventslist)-1].set_event_tracks(tracks)
        
        histos["h_nTracks"].Fill( n_tracks )
        histos["h_nTracks_log"].Fill( n_tracks if(n_tracks>0) else 0.11 )
        histos["h_nTracks_mid"].Fill( n_tracks )
        histos["h_nTracks_full"].Fill( n_tracks )
        histos["h_nTracks_zoom"].Fill( n_tracks )
        histos["h_nTracks_success"].Fill( n_successful_tracks )
        histos["h_nTracks_success_log"].Fill( n_successful_tracks if(n_successful_tracks>0) else 0.11 )
        histos["h_nTracks_success_full"].Fill( n_successful_tracks )
        histos["h_nTracks_success_mid"].Fill( n_successful_tracks )
        histos["h_nTracks_success_zoom"].Fill( n_successful_tracks )
        histos["h_nTracks_goodchi2"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_goodchi2_log"].Fill( n_goodchi2_tracks if(n_goodchi2_tracks>0) else 0.11 )
        histos["h_nTracks_goodchi2_full"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_goodchi2_mid"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_goodchi2_zoom"].Fill( n_goodchi2_tracks )
        histos["h_nTracks_selected"].Fill( n_selected_tracks )
        histos["h_nTracks_selected_log"].Fill( n_selected_tracks if(n_selected_tracks>0) else 0.11 )
        histos["h_nTracks_selected_full"].Fill( n_selected_tracks )
        histos["h_nTracks_selected_mid"].Fill( n_selected_tracks )
        histos["h_nTracks_selected_zoom"].Fill( n_selected_tracks )
        print(f"eventid={ievt} Tracking: Seeds={nSeeds}, AllTracks={n_tracks}, Success={n_successful_tracks}, GoodChi2={n_goodchi2_tracks}, Selected={n_selected_tracks}\n")
        
        if(n_successful_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("Fitted") )
        
        ### plot everything which is fitted but the function will only put the track line if it passes the chi2 cut
        fevtdisplayname = tfilenamein.replace("tree_","event_displays/").replace(".root",f"_{trigger}.pdf")
        # seeder.plot_seeder(fevtdisplayname)
        plot_event(runnumber,starttime,duration,trigger,fevtdisplayname,clusters,tracks,chi2threshold=cfg["cut_chi2dof"])
        
        if(n_goodchi2_tracks<1): continue ### CUT!!!
        histos["h_cutflow"].Fill( cfg["cuts"].index("#chi^{2}/N_{DoF}#leqX") )
        
        ### fill the data and add to events --> TODO: this is deprecated
        # eventslist.append( Event(meta,trigger,pixels_save,clusters,tracks,mcparticles) )
        # eventslist.append( Event(meta,trigger,pixels_save,clusters,seeds,tracks) )
        
    ### end
    if(not cfg["skiptracking"]):
        pickle.dump(eventslist, fpickle, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
        fpickle.close()
        
    print(f"Worker {irange} is done!")
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
        print("export LD_LIBRARY_PATH=$PWD/DetectorEvent/20252302:$LD_LIBRARY_PATH")
        ROOT.gInterpreter.AddIncludePath('DetectorEvent/20252302/')
        ROOT.gSystem.Load('libtrk_event_dict.dylib')
    print("---- finish loading libs")
    
    # print config once
    show_config()
    
    ### make directories, copy the input file to the new basedir and return the path to it
    tfilenamein = make_run_dirs(cfg["inputfile"])
    fpkltrgname = tfilenamein.replace("tree_","beam_quality/tree_").replace(".root","_BadTriggers.pkl")
    fpkltrigger = open(fpkltrgname,'rb')
    badtriggers = pickle.load(fpkltrigger)
    print(f"Found {len(badtriggers)} bad triggers")
    
    masked = {}
    if(cfg["skipmasking"]):
        print("\n----------------------------")
        print("Skipping/ignoring noise mask")
        print("----------------------------\n")
        masked = GetNoiseMaskEmpty()
    else:
        tfnoisename = tfilenamein.replace(".root","_noise.root")
        masked = GetNoiseMask(tfnoisename)
    
    ### the output histos
    tfilenameout = tfilenamein.replace(".root",f'{cfg["hfilesufx"]}.root')
    tfo = ROOT.TFile(tfilenameout,"RECREATE")
    tfo.cd()
    allhistos = book_histos(tfo)
    
    ### meta data:
    tfmeta = ROOT.TFile(tfilenamein,"READ")
    tmeta = tfmeta.Get("MyTreeMeta")
    if(tmeta is not None):
        try:
            nmeta = tmeta.GetEntries()
            tmeta.GetEntry(0)
            ts_starttime = tmeta.run_meta_data.run_start
            print( f"\nRun start:  {get_human_timestamp(ts_starttime)}" )
            if(nmeta>1): tmeta.GetEntry(nmeta-1)
            ts_endtime = tmeta.run_meta_data.run_end
            print( f"Run end:    {get_human_timestamp(ts_endtime)}" )
            print( f"Run duration [h]: {get_run_length(ts_starttime,ts_endtime)}" )
        except:
            print("Problem with Meta tree, continuing without it.")
    
    
    # Parallelize the analysis
    ### architecture depndent
    nCPUs = mp.cpu_count()
    print(f'nCPUs available: {nCPUs}')
    print(f'nCPUs configured: {cfg["nCPU"]}')
    if(cfg["nCPU"]<1):
        print("nCPU config cannot be <1, quitting")
        quit()
    elif(cfg["nCPU"]>=1 and cfg["nCPU"]<=nCPUs):
        nCPUs = cfg["nCPU"]
    else:
        print(f"nCPU config cannot be greater than {nCPUs}, quitting")
        quit()

    ### Create a pool of workers
    pool = mp.Pool(nCPUs)
    
    ### start the loop
    print(f"\nStarting the loop with tree file {tfilenamein}:")
    tfile0,ttree0,nevents0 = GetTree(tfilenamein)
    firstevent  = cfg["first2process"]
    max2process = cfg["nmax2process"]
    print(f"Events in tree: {nevents0}, Starting in event: {firstevent}, Processing maximum {max2process} events")
    nevents = nevents0-firstevent
    if(max2process>0 and max2process<=nevents):
        nevents = max2process
        print(f"Going to analyze only {nevents} events out of the {nevents0} available in the tree")
    else:
        print(f'config nmax2process={max2process} --> will analyze all events in the tree:{nevents}')
    bundle = nCPUs
    fullrange = range(firstevent,firstevent+nevents)
    print(fullrange)
    ranges = np.array_split(fullrange,bundle) if(nevents>=bundle) else [range(firstevent,firstevent+nevents)]
    for irng,rng in enumerate(ranges): print(f"Range[{irng}]: {rng[0]},...,{rng[-1]}")
    
    
    for irng,rng in enumerate(ranges):
        print(f"Submitting range[{irng}]: {rng[0]},...,{rng[-1]}")
        if(debug):
            histos = analyze(tfilenamein,irng,rng,masked,badtriggers)
        else:
            pool.apply_async(analyze, args=(tfilenamein,irng,rng,masked,badtriggers), callback=collect_histos, error_callback=collect_errors)
    
    ### Wait for all the workers to finish
    pool.close()
    pool.join()
    
    
    ### remove worker root files (they are anyhow empty out of the worker scope)
    for irng,rng in enumerate(ranges):
        sufx = "_"+str(irng)
        tfoname = tfilenamein.replace(".root",f'{cfg["hfilesufx"]}{sufx}.root')
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
