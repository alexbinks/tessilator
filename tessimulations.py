import sys, os
from glob import glob
import numpy as np
import random
from astropy.table import Table
from astropy.io import ascii
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import json
import itertools
print("THIS IS THE TESSIMULATION VERSION!")
from tessilator import tessilator





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
    num, time, flux, eflux = [], [], [], []
    d_points, n_diff = [], []
    if len(files) > n_bin:
        chosen_indices = np.random.choice(len(files), n_bin)
        files_chosen = [files[n] for n in chosen_indices]
    else:
        files_chosen = files
    for f, file in enumerate(files_chosen):
        tab = ascii.read(file)
        tab['enflux'][np.where(tab['enflux'] < 0)] = .01
        d_points.append(len(tab))
        n_diff.append(np.median(np.diff(tab['time'])))
        for t in tab:
            num.append(f+1)
            time.append(t['time'])
            flux.append(t['nflux'])
            eflux.append(t['enflux'])
    t_uniq = np.unique(np.array(time))

    t_fin = Table(names=('time', 'nflux', 'enflux', 'n_lc'), dtype=(float, float, float, int))
    for t in t_uniq:
        g = np.where(time == t)[0]
        flux_med = np.median(np.array(flux)[g])
        eflux_med = np.median(np.array(eflux)[g])
        flux_mean = np.mean(np.array(flux)[g])
        eflux_mean = np.mean(np.array(eflux)[g])
        num_lc = len(g)
        if num_lc > 2:
            t_fin.add_row([t, flux_med, eflux_med, num_lc])
        else:
            t_fin.add_row([t, flux_mean, eflux_mean, num_lc])
#    print(len(np.unique(np.array(time))))
#    t0, t1, tn, t_diff = min(time), max(time), np.median(d_points), np.median(n_diff)
#    t_arr = np.linspace(t0, t1, num=int(tn))
#    num, time, flux, eflux = np.array(num), np.array(time), np.array(flux), np.array(eflux)
#    dict_lc = defaultdict(list)
#    for t in t_arr:
#        for ind in 1+np.arange(len(files_chosen)):
#            g = np.where(num == ind)[0]
#            t_min = min(abs(time[g] - t))
#            t_test = t_min < 2.0*t_diff
#            if t_test:
#                dict_lc[f'{t}'].append(np.interp(t, time[g], flux[g]))
#    t_fin = Table(names=('time', 'nflux', 'enflux', 'n_lc'), dtype=(float, float, float, int))
#    for n, (k, v) in enumerate(dict_lc.items()):
#        if len(files) > 2:
#            t_fin.add_row([k, np.median(v), eflux[n], len(v)])
#        else:
#            t_fin.add_row([k, np.mean(v), eflux[n], len(v)])                
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
    fig, ax = plt.subplots(figsize=(10,7))
    for file in files:
        tab = ascii.read(file)
        ax.scatter(tab['time'], tab['nflux'])
    ax.scatter(med_lc['time'], med_lc['nflux'], s=40, c='black')
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
    fig, ax = plt.subplots(figsize=(30,15))
    sort_time = np.argsort(new_lc['time'])
    ax.set_xlabel('time [days]')
    ax.set_ylabel('normalised flux')
    ax.plot(new_lc['time'][sort_time], new_lc['nflux'][sort_time])
    amp_per_label = f'log(amp)={np.log10(amp)}, log(period)={np.log10(period)}'
    ax.text(0.01, 0.01, f'{amp_per_label}',
            horizontalalignment='left',
            transform=ax.transAxes)
    figname = f'log_amp_{np.log10(amp)}_log_per_{np.log10(period)}.png'
    fig.savefig(f'{directory}/{figname}', bbox_inches='tight')
    plt.close()


def run_tessimulate(dir_base, amps, periods,
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
    directories = sorted(glob(f'{dir_base}/mag_*'))
    if not directories:
        print(f"No fits files found inside {dir_base}")
        return None
    sim_res = defaultdict(list)
    for directory in directories:
        print(directory)
        d_spl = directory.split("_")
        mag_1, mag_2 = d_spl[-2], d_spl[-1]
        files = glob(f'{directory}/lc*')

        med_lc = get_lc_data(files, directory, n_bin=n_bin)
        if create_lc_plot:
            make_lc_plots(files, med_lc, directory)
        LS_dict = tessilator.run_ls(med_lc)
        sim_res['orig_periodLS'].append(LS_dict['Gauss_fit_peak_parameters'][1])
        sim_res['orig_e_periodLS'].append(LS_dict['Gauss_fit_peak_parameters'][2])

        for amp in amps:
            print(f'amplitude: {amp}')
            for period in periods:
                print(f'period: {period}')
                n_match = 0
                for n_ps in range(num_phase_shifts):
                    new_lc = dict()
                    jitter_x = np.random.normal(0, period*jitter_frac, size=len(med_lc))
                    jitter_y = np.random.normal(0, amp*jitter_frac, size=len(med_lc))

                    new_lc['enflux'] = med_lc['enflux']
                    phase = np.random.uniform(low=0., high=2.*np.pi)
                    error = [np.random.normal(0, sigma) for sigma in new_lc['enflux']]

                    new_lc['time'] = med_lc['time'] + jitter_x

                    new_lc['nflux'] = med_lc['nflux'] + \
                                      error + \
                                      jitter_y + \
                                      make_sine(amp, period, phase, med_lc['time'])
                    if create_sim_plots:
                        if (period == periods[9]) & (n_ps == num_phase_shifts-1):
                            make_sim_plots(new_lc, directory, period, amp)
                        
                    LS_dict = tessilator.run_ls(new_lc)
                    periodLS = LS_dict['Gauss_fit_peak_parameters'][1]
                    e_periodLS = LS_dict['Gauss_fit_peak_parameters'][2]
                    if abs(period-periodLS) < e_periodLS:
                        n_match += 1
                sim_res['mag_1'].append(mag_1)
                sim_res['mag_2'].append(mag_2)
                sim_res['amp'].append(amp)
                sim_res['period'].append(period)
                sim_res['n_frac'].append(1.*n_match/num_phase_shifts)
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

        ax[i].set_title(f'magnitude {int(mag)}-{int(mag)+1}')
        # set the limits of the plot to the limits of the data
        ax[i].set_xlabel('log Period')
        ax[i].set_ylabel('log Amplitude')
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

        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, label='fraction of matches')
        plt.subplots_adjust(bottom=0.4, right=0.8, top=0.9)
        cbar.ax.tick_params(labelsize=25)
    fig.subplots_adjust(wspace=0, hspace=0.5)    
    fig.savefig(f'{dir_base}/amp_mag_heatmap.png', bbox_inches='tight')
    plt.close()



def run_full_tessimulation(Sectors, Cameras, CCDs, mag_1s, mag_2s, tab_per, n_bin=10, num_phase_shifts=50,
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
    for Sector, Camera, CCD in itertools.product(Sectors, Cameras, CCDs):
        Sector, Camera, CCD = int(Sector), int(Camera), int(CCD)
        print(f'running the tessimulations for sector {Sector}, Camera {Camera}, CCD {CCD}') 
    # collect the subsample bins, for targets that are
    # best fit with a straight line rather than sinusoidal
    # e.g., flat, noisy lightcurves
        extn_scc = f'{Sector:02d}_{Camera}_{CCD}'
        dir_base = f'./tesssim/lc/{extn_scc}'

        for mag_1, mag_2 in zip(mag_1s, mag_2s):
            extn_mag = f'{mag_1}_{mag_2}'
            lc_dir=f'tesssim/lc/{extn_scc}/mag_{extn_mag}'
            lc_files = glob(f'{lc_dir}/lc*.csv')
            if not lc_files:
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
                    tessilator.all_sources_cutout(tTargets, f'tesssim_{extn_scc}_{extn_mag}', 0, 0, '_', 0,
                                                  choose_sec=Sector, store_lc=True, tot_attempts=2, cap_files=1,
                                                  lc_dir=lc_dir)

 
        sim_file = f'{dir_base}/sim_file.dat'
        if os.path.exists(sim_file):
            with open(f'{sim_file}', "r") as fp:
                sim_res = json.load(fp)
        else:
            print(f'running TESSIMULATIONS for {len(amps)} amplitudes and {len(periods)} periods...')
            sim_res = run_tessimulate(dir_base, amps, periods, n_bin=n_bin, num_phase_shifts=num_phase_shifts)
            if sim_res is None:
                continue
        if create_heatmap_plot:
            make_heatmap_plot(dir_base, sim_res, periods, amps)

if __name__ == "__main__":
    # choose the magnitude bins and number of targets per bin
    mag_b, mag_d, mag_n = 10, 1, 6
    mag_1s = mag_b + mag_d*np.arange(mag_n)
    mag_2s = mag_b + mag_d + mag_d*np.arange(mag_n)
    n_bin=10
    num_phase_shifts=10
    amps=np.logspace(-6, -1, num=6)
    periods=np.logspace(-1, 1.2, num=12)
    tab_per = ascii.read('./4SYS/TESS_4SYS_FOR_SIMULATION.csv')
    Sectors, Cameras, CCDs = 20+np.arange(10), 1+np.arange(4), 1+np.arange(4)
    run_full_tessimulation(Sectors, Cameras, CCDs, mag_1s, mag_2s, tab_per,
                           n_bin=n_bin, num_phase_shifts=num_phase_shifts,
                           amps=amps, periods=periods)
