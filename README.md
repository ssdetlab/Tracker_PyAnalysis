# Tracker_PyAnalysis

Prerequisits:
- have ROOT and python3 installed
- have the DetectorEvent header and lib installed (to read the EUDAQ data files)
  - it is also included in this rep for convenience, under DetectorEvent/

Setup:
- Setup ROOT
- `export LD_LIBRARY_PATH=$PWD/DetectorEvent/20240705:$LD_LIBRARY_PATH`
- put data files somewhere with enough space
  - there's a dir called `test_data` with example data already
- change config file as needed (see examples in the conf/ dir)

Run noise scan:
- if you want to process only part of the events, set the "nmax2process" parameter as needed
- to process all events check that the "nmax2process" parameter is larger than what you have in data
- change `doNoiseScan` in the config to 1
- `python3 serial_analyzer.py -conf conf/config_file_name.txt`
- change `doNoiseScan` in the config back to 0

Run analysis:
- run noise scan (see above)
- if you want to process only part of the events, set the "nmax2process" / "nmax2processMP" parameter as needed
- to process all events check that the "nmax2process" / "nmax2processMP" parameter is larger than what you have in data
- run analysis serially OR in parallel:
  - `python3 serial_analyzer.py -conf conf/config_file_name.txt`
  - `python3 multiproc_analyzer.py -conf conf/config_file_name.txt`
- to see event displays (fits...):
  - change `doplot` in the config to 1
  - run with `serial_analyzer.py` as above
  - kill the process after as many fits as desired
  - change `doplot` in the config back to 0

Run alignment with cosmics:
- run noise scan (see above)
- step 1: `python3 multiproc_analyzer.py -conf conf/config_file_name.txt` with all `misalignment` parameters set to 0 in the config file
- step 2: aligning wrt e.g. ALPIDE_0 or all at once after adjusting the parameters in the config: `maxchi2align`, `axes2align`, `naligniter`.
  - [option A.1] `python3 alignment_fitter.py -conf conf/config_file_name.txt -ref ALPIDE_0` or
  - [option A.2] `python3 alignment_fitter.py -conf conf/config_file_name.txt`
- step 3: choosing the fit strategy
  - [option B.1] if the `axes2align` parameter equals to `xytheta` then the fit will be FULLY-SIMULTANEOUS in 3D
  - [option B.2] if the `axes2align` parameter equals to `xy`, `xtheta` or `ytheta` then the fit will be SEMI-SIMULTANEOUS in 2D
  - [option B.3] if the `axes2align` parameter equals to `x`, `y` or `theta` then the fit will be SEQUENTIAL (i.e. non-simultaneous) in 1D
  - Notes:
    - if e.g. `-det ALPIDE_0` was used in option A.1 then you need to keep all `misalignment` parameters of `ALPIDE_0` fixed to 0 in the config file always
	 - it is advised to use one reference detector and align only the N-1 planes with respect to that with either the sequential or the simultaneous fit
	 - if the fit is SEMI-SIMULTANEOUS or SEQUENTIAL, you need to repeat steps 2-5 for all axes (e.g. `axes2align=x`->`axes2align=y`->`axes2align=theta`)
- step 4: put the non-zero resulting misalignment values in the config file for the relevant detectors
- step 5: run step 1 again, but with the new (non-zero wherever relevant) `misalignment` parameters in the config file (from step 3)
- step 6: check the residuals and the chi2 histograms

Plot some histos and fit the residuals:
- step 1: run the alignment as discussed above
- step 2: `python3 postproc_analyzer.py -conf conf/config_file_name.txt`
- step 3: look at the pdf file created in the dir where the root file is (as listed in the config file)
