import sys, os
from glob import glob
import numpy as np
import random
from astropy.table import Table
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import json
import itertools

print("THIS IS THE TESSIMULATION VERSION!")
from tessilator import tessilator
from tessilator import aperture
from tessilator import lc_analysis
from tessilator import periodogram





def get_lc_data(files, directory, n_bin=10):
    '''A function that makes the final noisy lightcurve
    
    For a given number of lightcurves, this function
    calculates the median flux at each time step if there are
    more than 2 measurements (or the mean if 2 or less).
    
    parameters
    ----------
    files : `list`
        a list of lightcurves in a given directory
        
    directory : `str`
        the name of the directory containing the lightcurves
        
    n_bin : `int`, optional, default=10
        the maximum number of lightcurves to be used in the analysis.
        
    returns
    -------
    t_fin : `astropy.table.Table`
        a table containing the data for the final noisy lightcurve
    '''
    
    dir_out = files[0]
    num, time, flux, eflux = [], [], [], []
    d_points, n_diff = [], []
    if len(files) > n_bin:
        chosen_indices = np.random.choice(len(files), n_bin)
        files_chosen = [files[n] for n in chosen_indices]
    else:
        files_chosen = files
    for f, file in enumerate(files_chosen):
        tab = ascii.read(file)
        tab['nflux_err'][np.where(tab['nflux_err'] < 0)] = .01
        d_points.append(len(tab))
        n_diff.append(np.median(np.diff(tab['time'])))
        for t in tab:
            if t['nflux_dtr'] > 0:
                num.append(f+1)
                time.append(t['time'])
                flux.append(t['nflux_dtr'])
                eflux.append(t['nflux_err'])
    t_uniq = np.unique(np.array(time))

    t_fin = Table(names=('time', 'nflux_dtr', 'nflux_err', 'lc_part', 'pass_clean_scatter', 'pass_clean_outlier', 'pass_full_outlier'), dtype=(float, float, float, int, bool, bool, bool))
    
    for t in t_uniq:
        g = np.where(time == t)[0]
        flux_med = np.median(np.array(flux)[g])
        eflux_med = np.median(np.array(eflux)[g])
        flux_mean = np.mean(np.array(flux)[g])
        eflux_mean = np.mean(np.array(eflux)[g])
        num_lc = len(g)
        if num_lc > 2:
            t_fin.add_row([t, flux_med, eflux_med, num_lc, True, True, True])
            
        else:
            t_fin.add_row([t, flux_mean, eflux_mean, num_lc, True, True, True])
        t_fin.write(f'{directory}/flux_fin.csv', overwrite=True)
    return t_fin
    
    
def make_lc_plots(files, med_lc, directory):
    '''Make a plot of the final normalized lightcurve
    
    Also plotted are the individual lightcurves that compose
    the final noisy lightcurve.
    
    parameters
    ----------
    files : `list`
        a list of lightcurves in a given directory
    med_lc : `astropy.table.Table`
        a table containing the data for the final noisy lightcurve
    directory : `str`
        the name of the directory containing the lightcurves
    
    returns
    -------
    None. The plot is saved in the directory provided
    '''
    fig, ax = plt.subplots(figsize=(15,5))
    plt.rcParams.update({'font.size': 20})
    for file in files:
        tab = ascii.read(file)
        ax.scatter(tab['time'], tab['nflux_dtr'])
    ax.scatter(med_lc['time'], med_lc['nflux_dtr'], s=40, c='black')
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Normalised flux')
    fig.savefig(f'{directory}/med_lc.png', bbox_inches='tight')
    fig.clf()
    
    
def make_sine(amp, period, phase, time):
    '''Make a sinusoidal function
    
    parameters
    ----------
    amp : `float`
        The amplitude of the sine wave
    period : `float`
        The period of the sine wave
    phase : `phase`
        The phase of the sine wave
    time : `iterable`
        The time coordinates.

    returns
    -------
    sine_wave : `iterable`
        The y-coordinates that comprise the sine wave
    '''
    sine_wave = amp*np.sin(phase + 2.*np.pi*time/period)
    return sine_wave
    

def make_sim_plots(new_lc, directory, period, amp):
    '''Make a plot of the simulated lightcurve.
    
    This is the combination of noisy lightcurve and the simulated
    sine wave.
    
    parameters
    ----------
    new_lc : `astropy.table.Table`
        The data for the simulated lightcurve
    directory : `str`
        the name of the directory containing the lightcurves
    period : `float`
        the period of the sine wave
    amp : `float`
        the amplitude of the sine wave
        
    returns
    -------
    None. The plot is saved in the directory provided.
    '''
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(15,5))
    sort_time = np.argsort(new_lc['time'])
    ax.set_xlabel('time [days]')
    ax.set_ylabel('normalised flux')
    ax.scatter(new_lc['time'][sort_time], new_lc['nflux_dtr'][sort_time])
    amp_per_label = f'log(amp)={np.log10(amp)}, log(period)={np.log10(period)}'
    ax.text(0.01, 0.01, f'{amp_per_label}',
            horizontalalignment='left',
            transform=ax.transAxes)
    figname = f'log_amp_{np.log10(amp)}_log_per_{np.log10(period)}.png'
    fig.savefig(f'{directory}/{figname}', bbox_inches='tight')
    plt.close()


def run_tessimulate(dir_base, amps, periods, mag_1s, mag_2s,
                    jitter_frac=.05, create_lc_plot=True, n_bin=10,
                    num_phase_shifts=50, create_sim_plots=True):
    '''Run the simulations for each noisy lightcurve.
    
    For a set of amplitudes and periods, and a given number of (random)
    phase shifts, a simulated lightcurve is made. A periodogram analysis
    is performed, and if the period measured by the periodogram is within
    one error bar of the injected period, this is classed as a match.
    
    parameters
    ----------
    dir_base : `str`
        The name of the directory with the noisy lightcurve for a given
        sector, camera and CCD configuration.
    amps : `iterable`
        The list of amplitudes to generate simulated lightcurves
    periods : `iterable`
        The list of periods to generate simulated lightcurves
    jitter_frac : `float`, optional, default=.05
        The fraction by which to randomly jitter the amplitude and period
        coordinates.
    create_lc_plot : `bool`, optional, default=True
        Choose to make plots of the individual lightcurves and the final
        noisy lightcurve used for the simulation
    n_bin : `int`, optional, default=10
        The number of lightcurves to form the noisy lightcurve
    num_phase_shifts : `int`, optional, default=50
        The number of repeat measurements to make for each amplitude and
        period configuration, where each time the sine wave is shifted by
        a random phase.
    create_sim_plots : `bool`, optional, default=True
        Choose to make plots of the simulated lightcurves.

    returns
    -------
    sim_res : `dict`
        A dictionary of results from the simulations.
    '''
    sim_res = defaultdict(list)
    for mag_1, mag_2 in zip(mag_1s, mag_2s):
        mag_dir = f'{dir_base}/mag_{mag_1:02d}_{mag_2:02d}'
        files = sorted(glob(f'{mag_dir}/lc*.csv'))
        if len(files) > 0:
            print(files, dir_base)
            med_lc = get_lc_data(files, mag_dir, n_bin=n_bin)
            
            print(med_lc)
            if create_lc_plot:
                make_lc_plots(files, med_lc, mag_dir)
            LS_dict = periodogram.run_ls(med_lc)
            sim_res['orig_periodLS'].append(LS_dict['period_1'])
            sim_res['orig_e_periodLS'].append(LS_dict['Gauss_1'][2])

            for amp in amps:
                print(f'amplitude: {amp}')
                for period in periods:
                    print(f'period: {period}')
                    n_match = 0
                    for n_ps in range(num_phase_shifts):

                        new_lc = Table(names=('time', 'nflux_dtr', 'nflux_err', 'lc_part', 'pass_clean_scatter', 'pass_clean_outlier', 'pass_full_outlier'), dtype=(float, float, float, int, bool, bool, bool))


                        jitter_x = np.random.normal(0, period*jitter_frac, size=len(med_lc))
                        jitter_y = np.random.normal(0, amp*jitter_frac, size=len(med_lc))

                        phase = np.random.uniform(low=0., high=2.*np.pi)
                        error = [np.random.normal(0, sigma) for sigma in med_lc['nflux_err']]

                        sine_arr = make_sine(amp, period, phase, med_lc['time'])
                        for i in range(len(med_lc)):
                            new_lc.add_row([med_lc['time'][i] + jitter_x[i],
                                            med_lc['nflux_dtr'][i] + error[i] + jitter_y[i] + sine_arr[i],
                                            med_lc['nflux_err'][i],
                                            1, True, True, True])
                        if create_sim_plots:
                            if (period == periods[-1]) & (n_ps == num_phase_shifts-1):
                                make_sim_plots(new_lc, mag_dir, period, amp)
                            
                        LS_dict = tessilator.run_ls(new_lc)
                        periodLS = LS_dict['period_1']
                        e_periodLS = LS_dict['Gauss_1'][2]
                        if abs(period-periodLS) < e_periodLS:
                            n_match += 1
                    sim_res['mag_1'].append(str(mag_1))
                    sim_res['mag_2'].append(str(mag_2))
                    sim_res['amp'].append(amp)
                    sim_res['period'].append(period)
                    sim_res['n_frac'].append(1.*n_match/num_phase_shifts)
    print(sim_res)
    with open(f'{dir_base}/sim_file.dat', "w") as fp:
        json.dump(sim_res, fp)  # encode dict into JSON
    print(f'finished TESSIMULATIONS for {dir_base}')
    return sim_res

def make_heatmap_plot(dir_base, sim_res, periods, amps):
    '''Make a heatmap of the fraction of period matches as a function of amplitude and period
    
    parameters
    ----------
    dir_base : `str`
        The name of the directory with the noisy lightcurve for a given
        sector, camera and CCD configuration.
    
    periods : `iterable`
        The list of periods to generate simulated lightcurves
    amps : `iterable`
        The list of amplitudes to generate simulated lightcurves
    sim_res : `dict`
        A dictionary of results from the simulations.
    
    returns
    -------
    None. A plot of the heatmap is saved to dir_base
    '''
    mags = np.unique(sim_res['mag_1'])
    fig, ax = plt.subplots(nrows=len(mags), ncols=1, figsize=(30,20*len(mags)))
    plt.rcParams.update({'font.size': 40})
    for i, mag in enumerate(mags):
        o_ps = sim_res['orig_periodLS'][i]
        o_eps = sim_res['orig_e_periodLS'][i]
        xx, yy = np.meshgrid(np.log10(periods), np.log10(amps))
        g = np.where(np.array(sim_res['mag_1']) == mag)[0]
        n_frac = np.array(sim_res['n_frac'])[g].reshape(len(amps), len(periods))
        n_frac = n_frac[::1][::-1]
        z_min, z_max = min(n_frac.ravel()), max(n_frac.ravel())

        ax[i].set_title(f'{int(mag)}<$G_{{\\rm mag}}$<{int(mag)+1}')
        # set the limits of the plot to the limits of the data
        if i == len(mag)-1:
            ax[i].set_xlabel('log Period [d]', fontsize=30)
        ax[i].set_ylabel('log Amplitude', fontsize=30)
        im = ax[i].imshow(n_frac, cmap='viridis',
                          interpolation='bicubic', 
                          vmin=z_min, vmax=z_max,
                          aspect='auto',
                          extent=[xx.min(), xx.max(), yy.min(), yy.max()])
        rect = patches.Rectangle((np.log10(o_ps-o_eps),yy.min()),
                                  np.log10(2.*o_eps),
                                  yy.max()-yy.min(),
                                  linewidth=1, edgecolor='r',
                                  facecolor='r', alpha=0.1)
        ax[i].add_patch(rect)
        ax[i].tick_params(labelsize=25)
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='fraction of matches')
        plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9)
        cbar.ax.tick_params(labelsize=20)
    fig.subplots_adjust(wspace=0, hspace=0.5)    
    fig.savefig(f'{dir_base}/amp_mag_heatmap.png', bbox_inches='tight')
    plt.close()
    
    
    
    
    
def get_fits(target, Sector, cutout_size=20, fits_dir='./tesssim/fits'):
    name_target = target['name'].replace(" ", "_")
    name_spl = name_target.split("_")
    if name_spl[0] == 'Gaia':
        name_target = name_spl[-1]
    coord = SkyCoord(target["ra"], target["dec"], unit="deg")
    file_in = tessilator.cutout_onesec(coord, cutout_size, name_target, Sector, tot_attempts=3, cap_files=None, fits_dir=fits_dir)[0]
    
    f_sp = file_in.split('/')[-1].split('-')
    if (len(f_sp) >=3) & (f_sp[0] == 'tess'):
        f_new = f'{fits_dir}/'+'_'.join([name_target, f_sp[1][1:], f_sp[2],\
                                         f_sp[3][0]])+'.fits'
        os.rename(f'./{file_in}', f_new)
        return f_new
    else:
        return file_in
    



def run_full_tessimulation(Sectors, Cameras, CCDs, mag_1s, mag_2s, n_bin=10, num_phase_shifts=50,
                           amps=np.logspace(-6, -1, num=9), periods=np.logspace(-1, 1.2, num=23),
                           create_heatmap_plot=True):
    '''The full monty. Run the tessimulations from start to finish!
    
    parameters
    ----------
    Sectors : `iterable`
        The list of sectors to run the tessimulator
    Cameras : `iterable`
        The list of cameras to run the tessimulator
    CCDs : `iterable`
        The list of CCDs to run the tessilator
    mag_1s : `iterable`
        The list of the lower-limits to each magnitude bin 
    mag_2s : `iterable`
        The list of the upper-limits to each magnitude bin
    tab_per : `astropy.table.Table`
        The table of targets which the tessimulator will select from.
        These were all calculated using the tessilator.
    n_bin : `int`, optional, default=10
        The maximum number of lightcurves to make a noisy lightcurve.
    num_phase_shifts : `int`, optional, default=50
        The number of random phase shifts for the tessimulation
    amps : `iterable`, optional, default=np.logspace(-6, -1, num=9)
        The list of amplitudes for the tessimulation
    periods : `iterable`, optional, default=np.logspace(-1, 1.2, num=23)
        The list of periods for the tessimulation
    create_heatmap_plot : `bool`, optional, default=True    
        Choose to make heatmaps from the tessimulation.
    returns
    -------
    None. This function calls all the other necessary functions in this module to
    run the tessimulations.
    '''
    
    for Sector in Sectors:
        tab_per = ascii.read(f'./LargeRun/Mar2024_4SYS/sector{Sector:02d}/periods_sector{Sector:02d}.csv')
        for Camera, CCD in itertools.product(Cameras, CCDs):
            Sector, Camera, CCD = int(Sector), int(Camera), int(CCD)
            print(f'running the tessimulations for sector {Sector}, Camera {Camera}, CCD {CCD}') 
        # collect the subsample bins, for targets that are
        # best fit with a straight line rather than sinusoidal
        # e.g., flat, noisy lightcurves
            extn_scc = f'{Sector:02d}_{Camera}_{CCD}'
            fits_base = f'./tesssim/{extn_scc}/fits'
            lc_base = f'./tesssim/{extn_scc}/lc'
            if not os.path.exists(fits_base):
                print(fits_base)
                os.makedirs(fits_base)
            for mag_1, mag_2 in zip(mag_1s, mag_2s):
                extn_mag = f'{mag_1}_{mag_2}'
                lcmag_base=f'{lc_base}/mag_{extn_mag}'
                if not os.path.exists(lc_base):
                    os.makedirs(lc_base)
                lcmag_files = glob(f'{lcmag_base}/lc*.csv')
                if not lcmag_files:
                    g = tab_per[
                                (tab_per['Sector']==Sector) & \
                                (tab_per['Camera']==Camera) & \
                                (tab_per['CCD']==CCD) & \
                                (tab_per['Gmag'] > mag_1) & \
                                (tab_per['Gmag'] < mag_2) & \
                                (tab_per['AIC_sine'] - tab_per['AIC_line'] > 1.5)
                               ]
                    # if there are more than n_bin matches, take a random subsample of "n_bin" targets
                    if len(g) > n_bin:
                        sub_sample = g[np.random.choice(len(g), n_bin)]
                    else:
                        sub_sample = g
                    source_ids = [s for s in sub_sample['source_id'].data]
                    if source_ids:
                        tab_info = dict()
                        tab_info['source_id'] = source_ids
                        tab_in = Table(tab_info)
                        tTargets = tessilator.read_data(tab_in)
                        print('running the TESSILATOR now')
                        for target in tTargets:
                            fits_file = get_fits(target, Sector, fits_dir=fits_base)
                            full_phot_table, Rad = aperture.aper_run(fits_file, target)
                            name_lc = f'lc_{target["name"].astype(str)}'
                            lc = lc_analysis.make_lc(full_phot_table, name_lc=name_lc, store_lc=True, lc_dir=lcmag_base)

 
            sim_file = f'{lc_base}/sim_file.dat'
            if os.path.exists(sim_file):
                with open(f'{sim_file}', "r") as fp:
                    sim_res = json.load(fp)
            else:
                print(f'running TESSIMULATIONS for {len(amps)} amplitudes and {len(periods)} periods...')
                sim_res = run_tessimulate(lc_base, amps, periods, mag_1s, mag_2s, n_bin=n_bin, num_phase_shifts=num_phase_shifts)
                if sim_res is None:
                    continue
            if create_heatmap_plot:
                print('here comes the heatmap!')
                make_heatmap_plot(lc_base, sim_res, periods, amps)

if __name__ == "__main__":
    # choose the magnitude bins and number of targets per bin
    mag_b, mag_d, mag_n = 10, 1, 8
    mag_1s = mag_b + mag_d*np.arange(mag_n)
    mag_2s = mag_b + mag_d + mag_d*np.arange(mag_n)
    n_bin=10
    num_phase_shifts=3
    amps=np.logspace(-4, -1, num=4)
    periods=np.logspace(-1.0, 1.0, num=5)
#    num_phase_shifts=2
#    amps=np.logspace(-1, -1, num=2)
#    periods=np.logspace(-1, 1, num=2)
    Sectors, Cameras, CCDs = np.array([27]), np.array([2]), np.array([3])
    run_full_tessimulation(Sectors, Cameras, CCDs, mag_1s, mag_2s,
                           n_bin=n_bin, num_phase_shifts=num_phase_shifts,
                           amps=amps, periods=periods)
