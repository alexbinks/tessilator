'''

Alexander Binks & Moritz Guenther, 2024

Licence: MIT 2024

Generate pixel images, light-curves and periodogram plots

'''

###############################################################################
####################################IMPORTS####################################
###############################################################################
# Internal
import sys
import inspect

# Third party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections.abc import Iterable


# Local application
from .file_io import logger_tessilator
###############################################################################
###############################################################################
###############################################################################



# initialize the logger object
logger = logger_tessilator(__name__) 

def create_plot(im_plot, clean, LS, scc, t_table, name_target, plot_dir,
                xy_ctr=(10,10), xy_contam=None, p_min_thresh=0.1,
                p_max_thresh=50., ap_rad=1.0, sky_ann=(6.,8.), nc='nc'):
    '''Produce a plot of tessilator results.

    | This module produces a 4-panel plot displaying information from the
      tessilator analysis. These are:
    | 1) An TESS cut-out image of the target, with aperture and sky annulus.
    | 2) A power vs period plot from the Lomb-Scargle periodogram analysis.
    | 3) A lightcurve of the normalised flux.
    | 4) The phase-folded lightcurve.

    parameters
    ----------
    im_plot : `astropy.nddata.Cutout2D`
        The cut-out image of the target
    clean : `dict`
        The modified (cleaned) lightcurve after processing
    LS : `dict`
        The dictionary of parameters calculated by the Lomb-Scargle periodogram
    scc : `list`, size=3
        List containing the sector number, camera and CCD
    t_table : `astropy.table.Table`
        Table containing the input data for the target
    name_target : `str`
        The name of the target
    plot_dir : `str`
        The directory to save the plots.
    XY_ctr : `tuple`, optional, default=(10,10)
        The centroid (in pixels) of the target in the TESS image.
    XY_contam : `Iterable` or `None`, optional, default = `None`
        The pixel positions of the strongest contaminants
    p_min_thresh : `float`, optional, default=0.1
        The shortest period calculated in the Lomb-Scargle periodogram
    p_max_thresh : `float`, optional, default=50.
        The longest period calculated in the Lomb-Scargle periodogram
    ap_rad : `float`, optional, default=1.0
        The aperture radius from the aperture photometry
    sky_ann : `Iterable`, size=2, optional, default=[6.,8.]
        The inner and outer background annuli from aperture photometry  
    nc : `str`, optional, default='nc'
        Describes the type of noise correction applied to the lightcurve.

    returns
    -------
    Nothing returned. The resulting plot is saved to file.
    '''
    t_0 = clean["time"][0]
    c_1 = clean["pass_sparse"].data
    c_2 = clean["pass_clean_outlier"].data
    cln_cond = np.logical_and.reduce([
                   clean["pass_clean_scatter"],
                   clean["pass_clean_outlier"],
                   clean["pass_full_outlier"]
                   ])

    clean_orig_time = clean["time"]-t_0
    clean_orig_flux = clean["nflux_ori"]
    clean_orig_mag = clean["mag"]
    
    clean_norm_time = clean["time"][c_1]-t_0
    clean_norm_flux = clean["nflux_ori"][c_1]

    clean_detr_time = clean["time"][cln_cond]-t_0
    clean_detr_flux = clean["nflux_dtr"][cln_cond]
    mpl.rcParams.update({'font.size': 14})
    if LS["AIC_line"]+1. < LS["AIC_sine"]:
        best_fit_type = 'linear'
    else:
        best_fit_type = 'sine'
    fsize = 22.
    lsize = 0.9*fsize
    fig, axs = plt.subplots(2,2, figsize=(20,15))

    axs[0,0].set_position([0.05,0.55,0.40,0.40])
    axs[0,1].set_position([0.55,0.55,0.40,0.40])
    axs[1,0].set_position([0.05,0.3,0.90,0.2])
    axs[1,1].set_position([0.05,0.05,0.90,0.2])

    circ_aper = Circle(xy_ctr, ap_rad, linewidth=1.2, fill=False, color='r')
    circ_ann1 = Circle(xy_ctr, sky_ann[0], linewidth=1.2, fill=False, color='b')
    circ_ann2 = Circle(xy_ctr, sky_ann[1], linewidth=1.2, fill=False, color='b')

    print(t_table)
    with np.errstate(all='ignore'):
        log_im_plot = np.log10(im_plot.data)
        image_plot = np.ma.array(log_im_plot, mask=np.isnan(log_im_plot))
    im_fig = axs[0,0].imshow(image_plot, cmap='binary')
    Gaia_name = f"{t_table['source_id'][0]}"
    targ_name = t_table['name'][0]
    fig.text(0.5,0.96,
             f"{targ_name}, Sector {scc[0]}, "
             f"Camera {scc[1]}, "
             f"CCD {scc[2]}",
             fontsize=lsize*2.0,
             horizontalalignment='center')
    axs[0,0].set_xlabel("x pixel", fontsize=fsize)
    axs[0,0].set_ylabel("y pixel", fontsize=fsize)
    axs[0,0].add_patch(circ_aper)
    axs[0,0].add_patch(circ_ann1)
    axs[0,0].add_patch(circ_ann2)
    axs[0,0].text(0.01,0.94, "$r_{\\rm ap}$ = "
                  f"{ap_rad:.2f}", color='red',
                  fontsize=lsize,horizontalalignment='left', 
                  transform=axs[0,0].transAxes)
    if isinstance(xy_contam, Iterable):
        axs[0,0].scatter(xy_contam[:, 0], xy_contam[:, 1], marker='X',
                         s=400, color='orange')
    divider = make_axes_locatable(axs[0,0])
    cax = divider.new_horizontal(size='5%', pad=0.4)
    fig.add_axes(cax)
    cbar = fig.colorbar(im_fig, cax=cax)
    cbar.set_label('log$_{10}$ counts (e$^-$/s)', rotation=270, labelpad=+15)

    axs[0,1].set_xlim([p_min_thresh, p_max_thresh])
    axs[0,1].grid(True)
    axs[0,1].set_xlabel("Period (days)", fontsize=fsize)
    axs[0,1].set_ylabel("Power", fontsize=fsize)
    axs[0,1].semilogx(LS['period_a_1'], LS['power_a_1'])
    [axs[0,1].axhline(y=i, linestyle='--', color='grey', alpha=0.8) \
     for i in LS['FAPs']]
    axs[0,1].text(0.01,0.94, f"Best fit: {best_fit_type}",
                  fontsize=lsize,horizontalalignment='left', 
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.94, "$P_{\\rm rot}^{\\rm (max)}$ = "
                  f"{LS['period_1']:.3f} days, "
                  f"power = {LS['power_1']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.82, "$P_{\\rm rot}^{\\rm (2nd)}$ = "
                  f"{LS['period_2']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.76, f"power ratio = "
                  f"{LS['power_1']/LS['power_2']:.3f}",
                  fontsize=lsize,horizontalalignment='right', 
                  transform=axs[0,1].transAxes)

    if (LS['Gauss_1'][1] != 15) & \
       (isinstance(LS['period_around_1'], Iterable)):
        axs[0,1].plot(LS['period_around_1'],
                      LS['Gauss_y_1'],
                      c='r', label='Best fit')
        axs[0,1].text(0.99,0.88, "$P_{\\rm rot}^{\\rm (Gauss)}$ = "
                      f"{LS['Gauss_1'][1]:.3f} $\\pm$"
                      f"{LS['Gauss_1'][2]:.3f}",
                      fontsize=lsize, horizontalalignment='right',
                      transform=axs[0,1].transAxes)    
    if LS['shuffle_flag'] > 0:
        axs[0,1].axvline(x=LS['period_1'], color='red', linewidth=3, alpha=0.3)
    axs[1,0].set_xlim([0, 30])
    axs[1,0].set_xlabel("Time (days)", fontsize=fsize)
    axs[1,0].set_ylim(
        [LS['median_MAD_nLC'][0]-(8.*LS['median_MAD_nLC'][1]),
        LS['median_MAD_nLC'][0]+(8.*LS['median_MAD_nLC'][1])])
    axs[1,0].set_ylabel("normalised flux", c='g', fontsize=fsize)
    axs[1,0].plot(LS["time"]-t_0, LS['y_fit_LS'], c='orange',
                  linewidth=1.5, label='LS best fit')
    axs[1,0].scatter(clean_orig_time, clean_orig_flux, s=1.0, c='pink',
                     alpha=0.5, label='raw, normalized')
    axs[1,0].scatter(clean_norm_time, clean_norm_flux, s=1.0, c='r', 
                     alpha=0.5, label='cleaned, normalized')
    axs[1,0].scatter(clean_detr_time, clean_detr_flux, s=1.2, c='g',
                     alpha=0.7, label='cleaned, normalized, detrended')
    if LS['jump_flag']:
        axs[1,0].text(0.01,0.90, 'Jumps detected', fontsize=lsize,
                      horizontalalignment='left',
                      transform=axs[1,0].transAxes)
    if LS["CBV_flag"] == 1:
        axs[1, 0].text(
            0.01,
            0.01,
            "best fit: original flux",
            fontsize=lsize,
            horizontalalignment="left",
            transform=axs[1, 0].transAxes,
        )
    if LS["CBV_flag"] == 2:
        axs[1, 0].text(
            0.01,
            0.01,
            "best fit: CBV corrected flux",
            fontsize=lsize,
            horizontalalignment="left",
            transform=axs[1, 0].transAxes,
        )
    axs[1,0].text(0.99,0.90, Gaia_name, fontsize=lsize,
                  horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.80, f"Gmag = {float(t_table['Gmag']):.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1, 0].text(
        0.99,
        0.70,
        f"$\\log (f_{{\\rm bg}}/f_{{*}})$ = {float(t_table['log_tot_bg']):.3f}",
        fontsize=lsize,
        horizontalalignment="right",
        transform=axs[1, 0].transAxes,
    )
    axs[1, 0].legend(loc="lower right", markerscale=10)
    ax2=axs[1,0].twinx()
    ax2.set_position([0.05,0.3,0.90,0.2])
    ax2.invert_yaxis()

    if not np.all(clean_orig_mag.data == -999.):
        ax2.scatter(clean_orig_time[clean_orig_mag>-999],
                    clean_orig_mag[clean_orig_mag>-999],
                    s=0.3, alpha=0.3, color="b", marker="x")
        ax2.set_ylabel("TESS magnitude", c="b",fontsize=fsize)

    axs[1,1].set_xlim([0,1])
    axs[1,1].set_xlabel("phase", fontsize=fsize)
    axs[1,1].set_ylabel("normalised flux", fontsize=fsize)
    axs[1,1].plot(LS['phase_fit_x'], LS['phase_fit_y'], c='b')
    LS["phase_col"] += 1
    N_cyc = int(max(LS["phase_col"]))
    cmap_use = plt.get_cmap('rainbow', N_cyc)
    s = axs[1,1].scatter(LS['phase_x'], LS["phase_y"],
                         c=LS['phase_col'], cmap=cmap_use, vmin=0.5,
                         vmax=N_cyc+0.5)
#    axs[1,1].text(0.01, 0.90, f"Amplitude = {LS['amp']:.3f}, "
#                  f"Scatter = {LS['scatter']:.3f}, "
#                  f"$\chi^{2}$ = {LS['chisq_phase']:.3f}, "
#                  "$f_{\\rm dev}$"+ f"= {LS['fdev']:.3f}",
#                  fontsize=lsize, horizontalalignment='left',
#                  transform=axs[1,1].transAxes)

#    cbaxes = inset_axes(axs[1,1], width="100%", height="100%",
#                        bbox_to_anchor=(0.79, 0.92, 0.20, 0.05),
#                        bbox_transform=axs[1,1].transAxes)
#    cbar = plt.colorbar(s, cax=cbaxes, orientation='horizontal',
#                        label='cycle number')
    plot_name = '_'.join([name_target, f"{scc[0]:04d}",
                          f"{scc[1]}", f"{scc[2]}", f"{nc}"])+'.png'
    fig.savefig(f"./{plot_dir}/{plot_name}", bbox_inches="tight")
    plt.close('all')


__all__ = [item[0] for item in inspect.getmembers(sys.modules[__name__],
           predicate = lambda f: inspect.isfunction(f)
           and f.__module__ == __name__)]
