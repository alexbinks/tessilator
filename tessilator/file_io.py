"""

Alexander Binks & Moritz Guenther, December 2024

Licence: MIT 2024
"""

import os
import logging
import sys

def logger_tessilator(name_log, log_ext='logging'):
    '''A function to set up the logging files from each tessilator module
    
    parameters
    ----------
    ref_name : `str`
        The reference name for each subdirectory which will connect all output
        files.
    name_log : `str`
        The name of the python module.
    log_ext : `str`, optional, default='logging'
        The name of the directory to save the logging files to.

    returns
    -------
    logger : `logging.getLogger`
        The logging object created by the function.
    '''
    logger = logging.getLogger(name_log)    
    if len(sys.argv)==6:
        log_dir = make_dir(log_ext, sys.argv[4])
    else:
        log_dir = make_dir(log_ext, 'target')
    

    f_handler = logging.FileHandler(f'{log_dir}/{name_log}.log')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - '
                                 '%(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    return logger


def make_dir(extn, ref):
    '''Create a directory to store various tessilator results.
    
    parameters
    ----------
    extn : `str`
        The name of the parent directory to store the files.
    ref : `str`
        The name of the sub-directory to store the specific set of data files.
    
    returns
    -------
    dir_name : `str`
        The name of the full directory extension AND creates the directory
        (if needed).
    '''
    dir_name = f'./{extn}/{ref}'
    dir_path_exist = os.path.exists(dir_name)
    if not dir_path_exist:
        os.makedirs(dir_name)
    return dir_name

