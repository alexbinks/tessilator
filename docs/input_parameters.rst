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

| If a program is called from the command line without all five input parameters, a set of prompts are initiated to receive input. If just one target is needed, then the user can simply supply either target name, or the sky coordinates as the final input.
| Otherwise, if the full set of command line parameters are supplied, the function will use these as the inputs, however, if they have the wrong format the program will return a warning message and exit.

