from ..aperture import get_xy_pos, aper_run
from ..logger import logger_tessilator

import numpy as np
import astropy.table as Table
from astropy.time import Time
from astropy.io import ascii, fits
from glob import glob

logger = logger_tessilator('xypos_sector_test')



fits_files = np.array(sorted(glob('./sector_tests/*ffic.fits')))
fits_tesscuts = np.array(sorted(glob('./sector_tests/*_*_*_*.fits')))

tess_fits_time_0 = '2014-12-08T12:00:00.000'
tess_fits_time_0 = Time(tess_fits_time_0, format='fits')
tess_jd0 = tess_fits_time_0.to_value('mjd')
time_fac = 0.0089
# the time fac is just time shift needed to match up the timestamps in both the cutout and sector data.

if len(fits_files) == 0:
    assert True
else:
    t = ascii.read('./sector_tests/sector_test.csv')
    Xpos, Ypos = t['Xpos'], t['Ypos']
    del t['Xpos']
    del t['Ypos']
    secs = [6,7,33]
    fits_secs = [fits_file.split('-')[1][1:] for fits_file in fits_files]
    tess_time = np.array([], dtype=np.float64)
    for f_n in fits_files:
        f_t = f_n.split('-')[0].split('/')[-1][4:]
        date_yday = ':'.join([f_t[:4],f_t[4:7],f_t[7:9],f_t[9:11],f_t[11:13]])
        tess_time = np.append(tess_time, Time(date_yday, format='yday').to_value(format='mjd')-tess_jd0)
    for i, sec in enumerate(secs):
        sec_str = f'{sec:04d}'
        matches = np.array([], dtype=int)
        for ind, fits_sec in enumerate(fits_secs):
            if fits_sec == sec_str:
                matches = np.append(matches, ind)


        tab_ffics = []
        t_spec = t[t["Sector"]==sec]
        fits_tesscut = fits_tesscuts[i]
        tab_cut, rad_cut = aper_run(fits_tesscut, t_spec)

        for fits_file, m in zip(fits_files[matches], matches):
            with fits.open(fits_file) as hdul:
                head = hdul[1].header
            pos_out = get_xy_pos(t_spec, head)
            
            tab_ffic, rad_ffic = aper_run(fits_file, t_spec)
            time_diff = np.abs(tab_cut["time"]-tess_time[m]-time_fac)
            t_cut = tab_cut[np.argsort(time_diff)]

            assert np.isclose(t_cut["flux"][0], tab_ffic["flux"][0], rtol=1.e-3)
            assert np.isclose(t_cut["reg_oflux"][0], tab_ffic["reg_oflux"][0], rtol=1.e-3)
