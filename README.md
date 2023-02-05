# The TESSilator
An all-in-one program to get TESS lightcurves and rotation periods

Alexander Binks and Moritz Guenther, 2023

The TESSilator

Licence: MIT

This is a python3 program designed to provide an all-in-one module to measure
lightcurves and stellar rotation periods from the Transiting Exoplanet Survey
Satellite (TESS). Whilst there are many useful (and powerful) software tools
available for working with TESS data, they are mostly provided as various steps
in the data reduction process --- to our knowledge there are no programs that
automate the full process from downloading the data (start) to obtaining
rotation period measurements (finish). The software provided here fills this
gap. The user provides a table of targets with basic Gaia DR3 information
(source ID, sky positions, parallax and Gaia G magnitude) and simply allows the
TESSilator to do the rest! The steps are:

(1) download photometric time-series data from TESS.

(2) scan the Gaia DR3 catalogue to quantify the level of background
    contamination from nearby sources.

(3) clean the lightcurves for poor quality data caused by systematic and
    instrumental effects.

(4) normalize and detrend lightcurves over the whole sector of observations.

(5) measure stellar rotation periods using two processes: the Lomb-Scargle
    periodogram and the Auto-Correlation Function.

(6) quantify various data quality metrics from photometric time-series data
    which can be used by the user to assess data reliability

The calling sequence is:

python tess_cutouts.py flux_con LC_con make_plots file_ref t_filename

where
      "flux_con" is the toggle for applying the contamination calculation
       (yes=1, no=0).
      "LC_con" determines if lightcurve/periodogram analyses should be carried
       out for neighbouring contaminants (yes=1, no=0).
      "make_plots" gives the user the option to make plots (yes=1, no=0)
      "file_ref" is a string expression used to reference the files produced.
      "t_filename" is the name of the input file required for analysis.

In this module, the data is downloaded from TESSCut (Brasseur et al. 2019) -- a
service which allows the user to acquire a stack of n 20x20 pixel
"postage-stamp" image frames ordered in time sequence and centered using the
celestial coordinates provided by the user, where n represents the number of
TESS sectors available for download. It uses modules from the TesscutClass
(astroquery.readthedocs.io/en/latest/api/astroquery.mast.TesscutClass.html) - 
part of the astroquery.mast package) to download the data, then applies steps
2-6, sector-by-sector.

Since there are no requirements to download the full image frames (0.5-0.7TB
and 1.8-2.0TB per sector for 30 and 10 min cadence observations, respectively)
this software is recommended for users who require a relatively fast extraction
for a manageable number of targets (i.e., < 5000). With the correct
pre-requisite Python modules and an uninterrupted internet connection, for a
target with 5 sectors of TESS data, the processing time is approximately 1-2
minutes (depending whether or not the user wants to measure contamination
and/or generate plots). If the user is interested in formulating a much larger
survey, we recommend they obtain the bulk downloads of the TESS calibrated
full-frame images
(archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html) and
run the "tess_large_sectors.py" module, which has been used to measure rotation
periods for catalogues of >1 million targets.

The module "tess_functions.py" contains the functions called to run both
tess_cutouts.py and tess_large_sectors.py

Should there be any problems in using this software please contact Alex Binks
(lead author) at abinks@mit.edu

If this package is useful for research leading to publication we would
appreciate the following acknowledgement:

"The data from the Transiting Exoplanet Survey Satellite (TESS) was acquired
using the TESSilator software package (Binks et al. 2023)."
