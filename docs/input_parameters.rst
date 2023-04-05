.. _input_parameters:
Required input parameters
=========================

Whether running from the command line or as part of a module, the input parameters are:

| 1) "flux_con": the toggle for applying the contamination calculation
       (yes=1, no=0).

| 2) either "LC_con", if using the cutout functions or "sector_num" if sectors are needed.       
* "LC_con" determines if lightcurve/periodogram analyses should be carried out for neighbouring contaminants (yes=1, no=0).
* "sector_num" prompts the user to enter the sector number needed. If command line arguments are not used, the program will ask if a specific Camera and CCD is needed (1=yes, 0=no). If not required, the whole sector is analysed. If this is a command line argument, if the user enters just the sector number (maximum 2 digits) the whole sector will be analysed, and if the Camera and CCD number are given right after the sector number with no spaces, then a specific Camera and CCD configuration will be used. E.G: if "sector_num = 8", the entire sector 8 is analysed, whereas if "sector_num = 814" then the program will analyse only Camera 1 and CCD 4 in sector 8.

| 3) "make_plots" gives the user the option to make plots (yes=1, no=0)

| 4) "file_ref" is a string expression used to reference the files produced.

| 5) "t_filename" is the name of the input file required for analysis.

Table input
-----------

If a program is called from the command line without all five input parameters, a set of prompts are initiated to receive input. For example, if the user has a table of targets that are all members of the Hyades, a command line execution might look like this:

    >>> python run_tess_cutouts.py
    ... Do you want to search for contaminants? 1=yes, 0=no : 1
    ... Do you want to calculate period data for the contaminants? 1=yes, 0=no : 1
    ... Would you like to make some plots? 1=yes, 0=no : 1
    ... Enter the unique name for referencing the output files : hyades_stars
    ... Enter the file name of your input table or target.
    ... If this is a single target please enter a hash (#) symbol before the identifier : hyades_data.csv

If the user knows what they want, then these can all be entered as command line arguments like this:

    >>> python run_tess_cutouts.py 1 1 1 hyades_stars hyades_data.csv

If the user follows the set of prompts and gives a value either of the wrong format or out of range, the tessilator will ask again for input until it passes. If the inputs are all from the command line and one (or more) input are not readable by the tessilator, the program will return a warning message and exit.

Single target input
-------------------

To save time constructing tables, the tessilator can accept "t_filename" as a single target, provided it can be resolved by either Simbad or Gaia DR3. To run the tessilator in this way, the target must have a hash (#) symbol before the name. For example, if we want to run the tessilator for AB Doradus using the prompt method, we would replace the input for "t_filename" with: 

    >>> Enter the file name of your input table or target.
    ... If this is a single target please enter a hash (#) symbol before the identifier : #AB Doradus

Alternatively if we want to run this in one go from the command line we can put

    >>> python run_tess_cutouts.py 1 1 1 hyades_stars "#AB Doradus"
    
please note that in this case we need double quotation marks surrounding the parameter so it can be passed as a string.
