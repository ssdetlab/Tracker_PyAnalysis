import ROOT

runname = "AallPix2_mc_prototype_beam_beryllium_window"
runnum  = 0
srunnum = "000"

### declare the data tree and its classes
ROOT.gROOT.ProcessLine("struct pixel  { Int_t ix; Int_t iy; };" )
ROOT.gROOT.ProcessLine("struct chip   { Int_t chip_id; std::vector<pixel> hits; };" )
ROOT.gROOT.ProcessLine("struct stave  { Int_t stave_id; std::vector<chip> ch_ev_buffer; };" )
ROOT.gROOT.ProcessLine("struct event  { Int_t trg_n; Double_t ts_begin; Double_t ts_end; std::vector<stave> st_ev_buffer; };" )
### declare the meta-data tree and its classes
ROOT.gROOT.ProcessLine("struct run_meta_data  { Int_t run_number; Double_t run_start; Double_t run_end; };" )

fIn = ROOT.TFile.Open(f"{runname}_run{srunnum}.root", "READ")
tIn = fIn.Get("MyTree")
tInMeta = fIn.Get("MyTreeMeta")

for entry in tIn:
    trg_n  = entry.event.trg_n
    staves = entry.event.st_ev_buffer
    for istv in range(staves.size()):
        staveid  = staves[istv].stave_id
        chips    = staves[istv].ch_ev_buffer
        for ichp in range(chips.size()):
            chipid   = chips[ichp].chip_id
            nhits    = chips[ichp].hits.size()
            for ipix in range(nhits):
                ix = chips[ichp].hits[ipix].ix
                iy = chips[ichp].hits[ipix].iy
                print(f"trg_n={trg_n}, staveid={staveid}, chipid={chipid} with {nhits} hits: ix={ix}, iy={iy}")


tInMeta.GetEntry(0)
runnumber = tInMeta.run_meta_data.run_number
ts_start  = tInMeta.run_meta_data.run_start
ts_end    = tInMeta.run_meta_data.run_end
print(f"From meta-data tree: runnumber={runnumber}, ts_start={ts_start}, ts_end={ts_end}")