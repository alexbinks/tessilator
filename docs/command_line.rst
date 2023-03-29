.. _command_line:
Running tessilator from the command line
========================================

The tessilator repository contains 2 example python modules which are executable from the
command line as shell scripts.

1) **run_tess_cutouts**
-----------------------
This module calls the ``all_sources_cutout`` function from the
``tessilator.py`` module. This program will download postage-stamp fits
files using ref:TESScut and runs the tessilator analysis for all the
sectors that are available for each target. To run this file simply enter::

   $ run_tess_cutouts

2) **run_tess_sectors**
-----------------------
This module calls the ``all_sources_sector`` function from the ``tessilator.py``
module. This program performs tessilator analyses using the full-frame calibrated
images (FFIC) that are stored locally on the user's machine. Bulk FFIC downloads
for each TESS sector are available at the `MAST website
<https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html>`_
as shell scripts which contain cURL commands to download all fits files in a given
sector. To run the tessilator in this way, it is important fits files are stored
such that all files in sector 'N' are in a subdirectory named './sectorNN/', where
NN represents a two digit integer (i.e., a trailing zero for sectors 1-9). To run
this program simply enter::

    $ run_tess_sector

Note that these programs will only run after a set of input parameters are passed.
This can be done by providing a set of :ref:`command-line arguments<input_parameters>`
after the script name, otherwise the tessilator will prompt the user for input.

Command line example
--------------------
An example of running the tessilator using the supplied shell scripts would be::

   $ run_tess_cutouts 1 1 1 cutouts targets.csv

| The command-line arguments here instruct the tessilator to:
| (a) Calculate the flux from contaminating sources
| (b) Run the tessilator analysis for neighbouring contaminants
| (c) Produce a plot for each lightcurve
| (d) Group the output files by the name ``cutouts``
| (e) Use ``cutout_targets.csv`` as the input table containing the targets

