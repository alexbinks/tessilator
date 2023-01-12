'''

Alexander Binks & Moritz Guenther, January 2023

This module contains all the functions called upon by either the tess_cutouts.py
or tess_large_sectors.py modules. These are:

1)  get_TESS_XY - a function to return the X, Y positions on the Calibrated Full
    Frame Images. This function is needed only for tess_large_sectors.py.
   
2)  contamination - returns three measured quantities to assess the amount of
    incident flux in the TESS aperture that is from contaminating background
    sources. This is calculated by querying the Gaia DR3 catalogue within a 5-pixel
    distance (~105") from the sky position of the target.
   
3a) aper_run_cutouts - returns a numpy array containing the times, magnitudes and
    fluxes from aperture photometry. This is the function used for tess_cutouts.py.

3b) aper_run_sectors - similarly returns a numpy array, but for all stars that are
    located within a given sector/ccd/camera. The resulting array has approximately,
    but no more than n * m rows, where n is the number of stars and m is the number
    of observations in a given sector (some data are excluded from the aperture
    photometry because of quality screening). This function is designed to be used
    by tess_large_sectors.
    
4)  gtd/make_detrends/make_lc - these 3 functions are made to remove spurious data
    points, ensure that only "strings" of contiguous data are being processed, and
    each string is detrended using either a linear (1st) or quadratic (2nd) polynomial
    (chosen by Aikake Inference Criterion) using only data contained in each string.
    Finally, the lightcurve is pieced together and make_lc returns a table containing
    the data ready for periodogram analysis.
    
5)  run_LS - function to conduct Lomb-Scargle periodogram analysis. Returns a table
    with period measurements, plus several data quality flags. If required a plot of
    the lightcurve, periodogram and phase-folded lightcurve is provided.

'''
from __main__ import *
from tess_stars2px import tess_stars2px_function_entry
t_contam = Table(names=['source_id', 'log_tot_bg', 'log_max_bg', 'num_tot_bg'],
                  dtype=(str, float, float, int))




# USE THE TESS SOFTWARE "TESS_STARS2PX_FUNCTION_ENTRY" TO RETURN A TABLE CONTAINING THE INSTRUMENTAL DATA
def get_TESS_XY(t_in):
    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, \
           outColPix, outRowPix, scinfo = tess_stars2px_function_entry(
           t_in['source_id'], t_in['ra'], t_in['dec'])
    return QTable([outID, outSec, outCam, outCcd, outColPix, outRowPix], names=('source_id', 'Sector', 'Camera', 'CCD', 'Xpos', 'Ypos'))





def contamination(t_targets):
    con1, con2, con3 = [], [], []
    for i in range(len(t_targets["Gmag"])):
        query = f"SELECT source_id, ra, dec, phot_g_mean_mag,\
        DISTANCE(\
        POINT({t_targets['ra'][i]}, {t_targets['dec'][i]}),\
        POINT(ra, dec)) AS ang_sep\
        FROM gaiadr3.gaia_source\
        WHERE 1 = CONTAINS(\
        POINT({t_targets['ra'][i]}, {t_targets['dec'][i]}),\
        CIRCLE(ra, dec, {5.0*pixel_size/3600.})) \
        AND phot_g_mean_mag < {t_targets['Gmag'][i]+3.0} \
        ORDER BY phot_g_mean_mag ASC"

        try:
            job = Gaia.launch_job(query)
        except Exception:
            traceback.print_exc(file=log)
            job = Gaia.launch_job_async(query)

        r = job.get_results()  
        r["ang_sep"] = r["ang_sep"]*3600.
        rx = r[r["source_id"] != t_targets["source_id"][i]]

        # calculate the fraction of flux from the source object that falls into the aperture
        # using the Rayleigh formula P(x) = 1 - exp(-[R^2]/[2*sig^2])
        s = Rad**(2)/(2.0*exprf**(2)) # measured in pixels
        fg_star = (1.0-np.exp((-s)))*10**(-0.4*t_targets["Gmag"][i])
        rx["fG"] = 10**(-0.4*rx["phot_g_mean_mag"])
        rx["t"] = (rx["ang_sep"]/pixel_size)**2/(2.0*exprf**(2)) # measured in pixels
        tx = []
        d_th= 0.0005
        for j in range(len(rx["t"])):
            n, z, n_z = 0, 0, 0
            while z < 1:
                sk = np.sum([(s**(k)/np.math.factorial(k)) for k in range(0,n+1)])
                sx = 1.0 - (np.exp(-s)*sk)
                n_0 = ((rx["t"][j])**n/np.math.factorial(n))*sx
                n_z = n_z + n_0
                if np.abs(n_0) > d_th:
                    n = n+1
                if np.abs(n_0) < d_th:
                    break
            tx.append(n_z*np.exp(-rx["t"][j])*10**(-0.4*rx["phot_g_mean_mag"][j]))
            
        if np.sum(tx) == 0:
            con1.append(-999)
            con2.append(-999)
        else:
            con1.append(np.log10(np.sum(tx)/fg_star))
            con2.append(np.log10(max(tx)/fg_star))
        con3.append(len(tx))
        with open(store_file, 'a') as file1:
            file1.write("Gmag = "+str(t_targets["Gmag"][i])+', log_tot_bg = '+
                        str(con1[i])+', log_max_bg = '+
                        str(con2[i])+', num_tot_bg = '+
                        str(con3[i])+', '+
                        str(i+1)+'/'+str(len(t_targets["Gmag"]))+
                        '\n')

        t_contam.add_row([str(t_targets["source_id"][i]), con1[i], con2[i], con3[i]])
    t_targets["log_tot_bg"] = con1
    t_targets["log_max_bg"] = con2
    t_targets["num_tot_bg"] = con3

    return t_targets
    
    
    






def aper_run_cutouts(f_file):
    '''
    Function to return the initial table of aperture photometry
    measured using photutils.
    There two try/except clauses, one to check if the fits file
    can be read, and the second to ensure data is present within
    the file.

    Inputs are,
       f_file: the fits file to be opened
    '''
#    print(j+1, len(manifest))
    try:
        with fits.open(f_file) as hdul:
            with open(store_file, "a") as log:
                try:
                    data = hdul[1].data
                    head = hdul[0].header
                    error = hdul[2].data
                except Exception:
                    traceback.print_exc(file=log)
                    return
            f_out = []
            for i in range(data.shape[0]):
                if data["QUALITY"][:][:][i] == 0:
                    flux_vals = data["FLUX"][:][:][i]
                    xc, yc = centroid_com(flux_vals)
                    positions = tuple([xc, yc])
                    if np.isfinite(positions[0]) == False or np.isfinite(positions[1]) == False:
                        if i > 0:
                            flux_vals_replace = data["FLUX"][:][:][i-1]
                            xc, yc = centroid_com(flux_vals_replace)
                            positions = tuple([xc, yc])
                        elif i == 0:
                            flux_vals_replace = data["FLUX"][:][:][i+1]
                            xc, yc = centroid_com(flux_vals_replace)
                            positions = tuple([xc, yc])
#define a circular aperture around all objects
                    aperture = CircularAperture(positions, Rad)
#select a background annulus
                    annulus_aperture = CircularAnnulus(positions, SkyRad[0], SkyRad[1])
#fit the background using the median flux in the annulus
                    aperstats = ApertureStats(flux_vals, annulus_aperture)
#obtain the raw (source+background) flux
                    phot_table = aperture_photometry(flux_vals, aperture, error=error)
#calculate the background contribution to the aperture
                    aperture_area = aperture.area_overlap(flux_vals)
#print out the data to "phot_table"
                    phot_table['bkg'] = aperstats.mode
                    phot_table['total_bkg'] = phot_table['bkg'] * aperture_area
                    phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['total_bkg']
                    phot_table['mag'] = -2.5*np.log10(phot_table['aperture_sum_bkgsub'])+Zpt
                    phot_table['mag_err'] = np.abs((-2.5/np.log(10))*phot_table['aperture_sum_err']/phot_table['aperture_sum'])
                    phot_table['time'] = data["TIME"][:][:][i]
                    phot_table['qual'] = data["QUALITY"][:][:][i]
                    for col in phot_table.colnames:
                        phot_table[col].info.format = '%.6f'
                    for j in range(len(phot_table)):
                        f_out.append(phot_table.as_array()[j])
            return f_out
    except OSError as e:
        return




    
def aper_run_sectors(f_file, objects, zr, lenzr):
    try:
        with fits.open(f_file) as hdul:
            with open(store_file, "a") as log:
                try:
                    data = hdul[1].data
                    head = hdul[1].header
                    error = hdul[2].data
                except Exception:
                    traceback.print_exc(file=log)
                    return
            if (head["DQUALITY"] != 0) or (head["NAXIS"] != 2):
                return
            print(objects["Sector"][0], objects["Camera"][0], objects["CCD"][0], zr+1, lenzr)
            with open(store_file, 'a') as file1:
                file1.write(str(f_file)+', '+
                            str(zr+1)+'/'+str(lenzr)+
                            '\n')
            w = WCS(head)
            c = SkyCoord(objects['ra'], objects['dec'], unit=u.deg, frame='icrs')
            with open(store_file, "a") as log:
                try:
                    x_obj, y_obj = w.world_to_pixel(c)
                except Exception:
                    traceback.print_exc(file=log)
                    return
            positions = tuple(zip(x_obj, y_obj))
      #define a circular aperture around all objects
            aperture = CircularAperture(positions, Rad)
      #select a background annulus
            annulus_aperture = CircularAnnulus(positions, SkyRad[0], SkyRad[1])
      #fit the background using the median flux in the annulus
            aperstats = ApertureStats(data, annulus_aperture)
            bkg_mode = 3.*aperstats.median - 2.*aperstats.mean
      #obtain the raw (source+background) flux
            phot_table = aperture_photometry(data, aperture, error=error)
            aperture_area = aperture.area_overlap(data)
      #print out the data to "phot_table"
            phot_table['source_id'] = objects['source_id']
            phot_table['bkg'] = aperstats.median
            phot_table['total_bkg'] = bkg_mode * aperture_area
            phot_table['aperture_sum_bkgsub'] = phot_table['aperture_sum'] - phot_table['total_bkg']
            phot_table['mag'] = -2.5*np.log10(phot_table['aperture_sum_bkgsub'])+Zpt
            phot_table['mag_err'] = np.abs((-2.5/np.log(10))*phot_table['aperture_sum_err']/phot_table['aperture_sum'])
            phot_table['time'] = (head['TSTART'] + head['TSTOP'])/2.
            phot_table['qual'] = head['DQUALITY']
            g_match = (phot_table["total_bkg"] > 0) & (phot_table["aperture_sum_bkgsub"] > 0)
            phot_table['qual'][~g_match] = 999
            for col in phot_table.colnames[:5]:
                phot_table[col].info.format = '%.6f'
            for col in phot_table.colnames[6:]:
                phot_table[col].info.format = '%.6f'
            phot_table['source_id'].info.format = '%s'
            f_out = []
            for i in range(len(phot_table)):
                f_out.append(list(phot_table.as_array()[i]))
            return f_out
    except OSError as e:
        return


def gtd(t, f):
    tm, fm = np.median(t), np.median(f)
    f_MAD  = median_abs_deviation(f, scale='normal')
    td     = np.zeros(len(t))
    td[1:] = np.diff(t)

    A0 = (np.abs(f-fm) < 1.*f_MAD).astype(int)
    A1 = (td < 10.*np.median(td)).astype(int)
    B  = (A0+A1).astype(int)
    gs, gf = [], []
    i = 0    
    l = 0
    while i < len(f)-1:
        if B[i] == 2:
            gs.append(i)
            j = i+1
            while (A1[j]) == 1 and (j < len(f)-1):
                j = j+1
            if j == len(f)-1:
                l = l+1
                break
            else:
                k = j
                j = j-1
                while A0[j] == 0:
                    j = j-1
                gf.append(j)
                i = k + 1
        else:
            i = i+1
    if l == 1:
        k = len(f)-1
        while B[k] != 2:
            k = k-1
        gf.append(k)
    
    gs, gf = np.array(gs), np.array(gf)
    return gs[gf-gs>50], gf[gf-gs>50]



def make_detrends(ds,df,t,f,err):
    t_fin, f_no, f_fin, e_fin, s_fin = [], [], [], [], []
    for i in range(len(ds)):
        p1, r1, _,_,_ = np.polyfit(t[ds[i]:df[i]+1], f[ds[i]:df[i]+1], 1, full=True)
        p2, r2, _,_,_ = np.polyfit(t[ds[i]:df[i]+1], f[ds[i]:df[i]+1], 2, full=True)
        chi1 = np.sum((np.polyval(p1, t[ds[i]:df[i]+1])-f[ds[i]:df[i]+1])**2/(err[ds[i]:df[i]+1])**2)
        chi2 = np.sum((np.polyval(p2, t[ds[i]:df[i]+1])-f[ds[i]:df[i]+1])**2/(err[ds[i]:df[i]+1])**2)
        AIC1, AIC2 = 2.*(2. - np.log(chi1/2.)), 2.*(3. - np.log(chi2/2.))
        if AIC1 < AIC2:
            fn = f[ds[i]:df[i]+1]/np.polyval(p2, t[ds[i]:df[i]+1])
            s_fit = 1
        else:
            fn = f[ds[i]:df[i]+1]/np.polyval(p2, t[ds[i]:df[i]+1])
            s_fit = 2
        t_fin.append(t[ds[i]:df[i]+1])
        f_no.append(f[ds[i]:df[i]+1])
        f_fin.append(fn)
        e_fin.append(err[ds[i]:df[i]+1])
        s_fin.append(s_fit)
    return t_fin, f_no, f_fin, e_fin, s_fin


def make_lc(phot_table):
    g = (phot_table["flux"][:] - np.median(phot_table["flux"][:]) < 20.0*median_abs_deviation(phot_table["flux"][:], scale='normal'))
    time = np.array(phot_table["time"][:][g])
    mag = np.array(phot_table["mag"][:][g])
    flux = np.array(phot_table["flux_corr"][:][g])
    eflux = np.array(phot_table["flux_err"][:][g])
    ds, df = gtd(time, flux)
# 1st: normalise the flux by dividing by the median value
    nflux  = flux/np.median(flux)

    neflux = eflux/flux
# 2nd: detrend each lightcurve sections by either a straight-line fit or a parabola. The choice is selected using AIC.
    t_detrend, f_orig, f_detrend, e_detrend, s_detrend = make_detrends(ds, df, time, nflux, neflux)

    original_data = dict()
    original_data["time"] = np.array([time])
    original_data["nflux"] = np.array([nflux])
    if len(t_detrend) > 0:
        if len(t_detrend[0][:]) > 50:
            clean_data = Table()
            clean_data['time']  = np.concatenate(t_detrend)
            clean_data['time0'] = np.concatenate(t_detrend)-time[0]
            clean_data['oflux'] = np.concatenate(f_orig)
            clean_data['nflux'] = np.concatenate(f_detrend)
            clean_data['enflux'] = np.concatenate(e_detrend)
            return clean_data, original_data
        else:
            return
    else:
        return


def get_second_peak(power):
### Get the left side of the peak
    p_m = np.argmax(power)
    x = p_m
    while (power[x-1] < power[x]) and (x > 0):
        x = x-1
    p_l = x
    p_lx = 0
    while (power[p_l] > 0.85*power[p_m]) and (p_l > 1):
        p_lx = 1
        p_l = p_l - 1
    if p_lx == 1:
        while (power[p_l] > power[p_l-1]) and (p_l > 0):
            p_l = p_l - 1
    if p_l < 0:
        p_l = 0

### Get the right side of the peak
    x = p_m
    if x < len(power)-1:
        while (power[x+1] < power[x]) and (x < len(power)-2):
            x = x+1
        p_r = x
        p_rx = 0
        while (power[p_r] > 0.85*power[p_m]) and (p_r < len(power)-2):
            p_rx = 1
            p_r = p_r + 1
        if p_rx == 1:
           while (power[p_r] > power[p_r+1]) and (p_r < len(power)-2):
                p_r = p_r + 1
        if p_r > len(power)-1:
            p_r = len(power)-1
        a = np.arange(len(power))
        a_g = a[p_l:p_r+1]
        a_o = a[np.setdiff1d(np.arange(a.shape[0]), a_g)] 
    return a_g, a_o


def filter_out_none(files):
    '''
    Simple function to remove list items if they are empty.
    >>> filter_out_none([1, 4, None, 6, 9])
    [1, 4, 6, 9]
    '''
    return list(filter(lambda item: item is not None, files))


def gauss_fit(x, a0, x_mean, sigma):
    '''
    Returns the best-fit Gaussian parameters: amplitude (A),
    mean (x_mean) and uncertainty (sigma) for a distribution
    of x values
    '''
    return a0*np.exp(-(x-x_mean)**2/(2*sigma**2))

def sine_fit(x, y0, A, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase)
    to a regular sinusoidal function.
    '''
    return y0 + A*np.sin(2.*np.pi*x + phi)



def run_LS(clean):
    '''
    Runs a Lomb-Scargle periodogram on the cleaned lightcurve
    and returns a dictionary of results.
    '''
    
    LS_dict = dict()
    med_f, MAD_f = np.median(clean["nflux"]), median_abs_deviation(clean["nflux"], scale='normal')
    ls = LombScargle(clean["time0"], clean["nflux"])
    frequency, power = ls.autopower(minimum_frequency=1./30,
                                    maximum_frequency=1./0.1,
                                    samples_per_peak=10)
    FAP = ls.false_alarm_probability(power.max())
    probabilities = [0.1, 0.05, 0.01]
    FAP_test = ls.false_alarm_level(probabilities)
    p_m = np.argmax(power)
    y_fit = ls.model(clean["time0"], frequency[p_m])
    period_best = 1.0/frequency[p_m]
    power_best = power[p_m]

    period = 1./frequency[::-1]
    power = power[::-1]
    
    a_g, a_o = get_second_peak(power) # a_g returns the datapoints that form the Gaussian
                                           # around the highest power
                                           # a_o returns the array for all other values 
    with open(store_file, "a") as log:
        try:
            popt, _ = curve_fit(gauss_fit, period[a_g], power[a_g], bounds=(0, [1., 30., 30.]))
        except Exception:
            traceback.print_exc(file=log)
            popt = np.array([1, 15, 15])
            pass

    ym = gauss_fit(period[a_g], *popt)

    per_a_o, power_a_o = period[a_o], power[a_o]
    per_2 = per_a_o[np.argmax(power[a_o])]
    pow_2 = power_a_o[np.argmax(power[a_o])]
    pow_pow2 = 1.0*power_best/pow_2
    x, c = np.modf(clean["time0"]/period_best)
    x = np.array(x)
    xg = np.sort(x)

    with open(store_file, "a") as log:
        try:
            pops, popsc = curve_fit(sine_fit, x, clean["nflux"].data, bounds=(0, [2., 2., 2.*np.pi]))

        except Exception:
            traceback.print_exc(file=log)
            pops = np.array([1., 0.001, 0.5])
            pass
    Ndata = len(clean)
    yp = sine_fit(x, *pops)
    phase_scatter = median_abs_deviation(yp - clean["nflux"], scale='normal')
    fdev = 1.*np.sum(np.abs(clean["nflux"] - yp) > 3.0*phase_scatter)/Ndata

    LS_dict['median_MAD_nLC'] = [med_f, MAD_f]
    LS_dict['period'] = period
    LS_dict['power'] = power
    LS_dict['period_best'] = period_best
    LS_dict['power_best'] = power_best 
    LS_dict['y_fit_LS'] = y_fit 
    LS_dict['FAPs'] = FAP_test
    LS_dict['Gauss_fit_peak_parameters'] = popt
    LS_dict['Gauss_fit_peak_y_values'] = ym
    LS_dict['period_around_peak'] = period[a_g]
    LS_dict['power_around_peak'] = power[a_g]
    LS_dict['period_not_peak'] = period[a_o] 
    LS_dict['power_not_peak'] = power[a_o] 
    LS_dict['period_second'] = per_2
    LS_dict['power_second'] = pow_2
    LS_dict['phase_fit_x'] = x[np.argsort(x)]
    LS_dict['phase_fit_y'] = sine_fit(x, *pops)[np.argsort(x)]
    LS_dict['phase_fit_cols'] = c[np.argsort(x)]
    LS_dict['pops_vals'] = pops    
    LS_dict['pops_cov'] = popsc
    LS_dict['phase_scatter'] = phase_scatter
    LS_dict['frac_phase_outliers'] = fdev
    LS_dict['Ndata'] = Ndata
    return LS_dict
    
def make_LC_plots(clean, orig, LS_dict, scc, t_table):    
    fsize = 22.
    lsize = 0.9*fsize
    fig, axs = plt.subplots(2,2, figsize=(20,15))
        
    axs[0,0].set_position([0.1,0.6,0.8,0.3])
    axs[0,0].set_xlim([0, 30])
    axs[0,0].set_xlabel("Time", fontsize=fsize)
    axs[0,0].set_ylim([LS_dict['median_MAD_nLC'][0]-(8.*LS_dict['median_MAD_nLC'][1]),
                       LS_dict['median_MAD_nLC'][0]+(8.*LS_dict['median_MAD_nLC'][1])])
    axs[0,0].set_ylabel("normalised flux", fontsize=fsize)
    axs[0,0].scatter(orig["time"]-orig["time"][0], orig["nflux"], s=0.5, alpha=0.3)
    axs[0,0].scatter(clean["time0"], clean["oflux"],s=0.5, c='r', alpha=0.5)
    axs[0,0].scatter(clean["time0"], clean["nflux"],s=1.2, c='g', alpha=0.7)
    axs[0,0].plot(clean["time0"], LS_dict['y_fit_LS'], c='b', linewidth=2)
    axs[0,0].text(0.01,0.94, f"Gaia DR3 {t_table['source_id']}", fontsize=lsize, transform=axs[0,0].transAxes)
    axs[0,0].text(0.01,0.88, f"Gmag = {t_table['Gmag']:.3f}", fontsize=lsize, transform=axs[0,0].transAxes)
    axs[0,0].text(0.01,0.82, "$\log (f_{\\rm bg}/f_{*})$ = " + f"{t_table['log_tot_bg']:.3f}", fontsize=lsize, transform=axs[0,0].transAxes)

    axs[1,0].set_position([0.1,0.1,0.35,0.4])
    axs[1,0].set_xlim([0, 30])
    axs[1,0].grid(True)
    axs[1,0].set_xlabel("Period", fontsize=fsize)
    axs[1,0].set_ylabel("Power", fontsize=fsize)
    axs[1,0].plot(LS_dict['period'], LS_dict['power'])
    [axs[1,0].axhline(y=i, linestyle='--', color='grey', alpha=0.8) for i in LS_dict['FAPs']]
    axs[1,0].text(0.99,0.94, "$P_{\\rm rot} $(max) = " + f"{LS_dict['period_best']:.3f} days, power = {LS_dict['power_best']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.82, "$P_{\\rm rot} (2) $= " + f"{LS_dict['period_second']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.76, f"power ratio = {LS_dict['power_best']/LS_dict['power_second']:.3f}",
                  fontsize=lsize,horizontalalignment='right', 
                  transform=axs[1,0].transAxes)

    if LS_dict['Gauss_fit_peak_parameters'][1] != 15:
        axs[1,0].plot(LS_dict['period_around_peak'], LS_dict['Gauss_fit_peak_y_values'], c='r', label='Best fit')
        axs[1,0].text(0.99,0.88, "$P_{\\rm rot} {\\rm Gauss} $= " + f"{LS_dict['Gauss_fit_peak_parameters'][1]:.3f}" + "$\\pm$" + f"{LS_dict['Gauss_fit_peak_parameters'][2]:.3f}",
                      fontsize=lsize, horizontalalignment='right',
                      transform=axs[1,0].transAxes)        

    axs[0,1].set_position([0.55,0.1,0.35,0.4])
    axs[0,1].set_xlim([0,1])
    axs[0,1].set_xlabel("phase", fontsize=fsize)
    axs[0,1].set_ylabel("normalised flux, $y$", fontsize=fsize)
    axs[0,1].plot(LS_dict['phase_fit_x'], LS_dict['phase_fit_y'], c='b')
    axs[0,1].scatter(LS_dict['phase_fit_x'], clean["nflux"], c=LS_dict['phase_fit_cols'], cmap='cool')
        
    axs[1,1].remove()

    plt.savefig('_'.join([str(t_table['source_id']), str(scc[0]), str(scc[1]), str(scc[2])])+'.png', bbox_inches='tight')
    plt.close('all')

