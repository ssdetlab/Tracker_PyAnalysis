# Tracker_PyAnalysis

Prerequisits:
- have ROOT and python3 installed
- have the DetectorEvent header and lib installed (to read the EUDAQ data files)
  - it is also included in this rep for convenience, under DetectorEvent/

Setup:
- Setup ROOT
- `export LD_LIBRARY_PATH=DetectorEvent/libs:$LD_LIBRARY_PATH`
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
  - `python3 alignment_fitter.py -conf conf/config_file_name.txt -det ALPIDE_0` or
  - `python3 alignment_fitter.py -conf conf/config_file_name.txt`
  - Notes:
    - if e.g. `-det ALPIDE_0` was used then you need to keep all `misalignment` parameters of `ALPIDE_0` fixed to 0 in the config file always
    - if the `axes2align` parameter equals to `xytheta` then the fit will be FULLY-SIMULTANEOUS in 3D
    - if the `axes2align` parameter equals to `xy`, `xtheta` or `ytheta` then the fit will be SEMI-SIMULTANEOUS in 2D
    - if the `axes2align` parameter equals to `x`, `y` or `theta` then the fit will be SEQUENTIAL (i.e. non-simultaneous) in 1D
- step 3: put the non-zero resulting misalignment values in the config file for the relevant detectors
- step 4: run step 1 again, but with the new (non-zero wherever relevant) `misalignment` parameters in the config file (from step 3)
- step 5: check the residuals and the chi2 histograms
- step 6: if the fit is SEMI-SIMULTANEOUS or SEQUENTIAL, you need to repeat steps 2-5 for all axes (e.g. `axes2align=x`->`axes2align=y`->`axes2align=theta`)