# test_contaminants.py
import sys, os
import numpy as np
sys.path.append('..')
#from ..modules_to_import import *
from ..contaminants import flux_fraction_contaminant

def test_biser_millman():
    t_targ0 = 0.0**2/(2.0*0.65**(2))
    t_targ4 = 0.0**2/(2.0*0.65**(2))
    t_targ2 = 0.0**2/(2.0*0.65**(2))
    t_targ_inf = 1000.**(2.0*0.65**(2))
    s = 1.0**2/(2.0*0.65**(2))
    val_cont0 = flux_fraction_contaminant(t_targ0, s)
    assert np.isclose(val_cont0,1.0-np.exp(-s), rtol=0.0, atol=1e-4)
    

