How to use
==========

From the command line
---------------------
Enter six inputs following the name of the python module as follows

  &> python run_tess_program.py 1 1 1 suffix name_file
  
Automatically run the program
-----------------------------
The tessilator will prompt you to enter these six inputs one-by-one

Inside Python
--------------

   >>> from tessilator import tessilator
   >>> out = tessialteor.all_sources_cutout(t_targets, period_file, LC_con, flux_con, con_file, make_plots, choose_sec=None)
   
API documentation
-----------------
.. automodapi:: tessilator.tessilator
.. automodapi:: tessilator.lc_analysis
.. automodapi:: tessilator.contaminants
.. automodapi:: tessilator.makeplots
.. automodapi:: tessilator.maketable
.. automodapi:: tessilator.fixedconstants

