'''Generate pixel images, light-curves and periodogram plots

'''
from astropy.table import Table
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections.abc import Iterable

__all__ = ['make_plot']

def make_plot(im_plot, clean, orig, LS_dict, scc, t_table, XY_ctr=(10,10),
              XY_contam=None, p_min_thresh=0.1, p_max_thresh=50., Rad=1.0,
              SkyRad = [6.,8.]):
    '''Produce a plot of tessilator results.

    | This module produces a 4-panel plot displaying information from the tessilator analysis. These are:
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
    orig : `dict`
        The original normalised lightcurve before processing.
    LS_dict : `dict`
        The dictionary of parameters calculated by the Lomb-Scargle periodogram
    scc : `list`, size=3
        List containing the sector number, camera and CCD
    t_table : `astropy.table.Table`
        Table containing the input data for the target
    XY_ctr : `tuple`, optional, default=(10,10)
        The centroid (in pixels) of the target in the TESS image.
    XY_contam : `Iterable` or `None`, optional, default = `None`
        The pixel positions of the strongest contaminants
    p_min_thresh : `float`, optional, default=0.1
        The shortest period calculated in the Lomb-Scargle periodogram
    p_max_thresh : `float`, optional, default=50.
        The longest period calculated in the Lomb-Scargle periodogram
    Rad : `float`, optional, default=1.0
        The aperture radius from the aperture photometry
    SkyRad : `Iterable`, size=2, optional, default=[6.,8.]
        The inner and outer background annuli from aperture photometry  

    returns
    -------
    Nothing returned. The plot produced is saved to file.
    '''
    mpl.rcParams.update({'font.size': 14})

    fsize = 22.
    lsize = 0.9*fsize
    fig, axs = plt.subplots(2,2, figsize=(20,15))

    axs[0,0].set_position([0.05,0.55,0.40,0.40])
    axs[0,1].set_position([0.55,0.55,0.40,0.40])
    axs[1,0].set_position([0.05,0.3,0.90,0.2])
    axs[1,1].set_position([0.05,0.05,0.90,0.2])

    t_orig0 = orig["time"]-orig["time"][0]
    circ_aper = Circle(XY_ctr, Rad, linewidth=1.2, fill=False, color='r')
    circ_ann1 = Circle(XY_ctr, SkyRad[0], linewidth=1.2, fill=False, color='b')
    circ_ann2 = Circle(XY_ctr, SkyRad[1], linewidth=1.2, fill=False, color='b')
    f = axs[0,0].imshow(im_plot.data, cmap='binary')
    name_underscore = t_table['name'][0].replace(" ", "_")

    Gaia_name = f"Gaia DR3 {t_table['source_id'][0]}"
    targ_name = t_table['name'][0]
    fig.text(0.5,0.96,
             f"{targ_name}, Sector {str(scc[0])}, "
             f"Camera {str(scc[1])}, "
             f"CCD {str(scc[2])}",
             fontsize=lsize*2.0,
             horizontalalignment='center')
    axs[0,0].set_xlabel("X pixel", fontsize=fsize)
    axs[0,0].set_ylabel("Y pixel", fontsize=fsize)
    axs[0,0].add_patch(circ_aper)
    axs[0,0].add_patch(circ_ann1)
    axs[0,0].add_patch(circ_ann2)
    if isinstance(XY_contam, Iterable):
        axs[0,0].scatter(XY_contam[:, 0], XY_contam[:, 1], marker='X',
                         s=150, color='y')

    divider = make_axes_locatable(axs[0,0])
    cax = divider.new_horizontal(size='5%', pad=0.4)
    fig.add_axes(cax)
    cbar = fig.colorbar(f, cax=cax)
    cbar.set_label('log$_{10}$ counts (e$^-$/s)', rotation=270, labelpad=+15)

    axs[0,1].set_xlim([p_min_thresh, p_max_thresh])
    axs[0,1].grid(True)
    axs[0,1].set_xlabel("Period (days)", fontsize=fsize)
    axs[0,1].set_ylabel("Power", fontsize=fsize)
    axs[0,1].semilogx(LS_dict['period'], LS_dict['power'])
    [axs[0,1].axhline(y=i, linestyle='--', color='grey', alpha=0.8) \
     for i in LS_dict['FAPs']]
    axs[0,1].text(0.99,0.94, "$P_{\\rm rot}^{\\rm (max)}$ = "
                  f"{LS_dict['period_best']:.3f} days, "
                  f"power = {LS_dict['power_best']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.82, "$P_{\\rm rot}^{\\rm (2nd)}$ = "
                  f"{LS_dict['period_second']:.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[0,1].transAxes)
    axs[0,1].text(0.99,0.76, f"power ratio = "
                  f"{LS_dict['power_best']/LS_dict['power_second']:.3f}",
                  fontsize=lsize,horizontalalignment='right', 
                  transform=axs[0,1].transAxes)
    if LS_dict['Gauss_fit_peak_parameters'][1] != 15:
        axs[0,1].plot(LS_dict['period_around_peak'],
                      LS_dict['Gauss_fit_peak_y_values'],
                      c='r', label='Best fit')
        axs[0,1].text(0.99,0.88, "$P_{\\rm rot}^{\\rm (Gauss)}$ = "
                      f"{LS_dict['Gauss_fit_peak_parameters'][1]:.3f} $\\pm$"
                      f"{LS_dict['Gauss_fit_peak_parameters'][2]:.3f}",
                      fontsize=lsize, horizontalalignment='right',
                      transform=axs[0,1].transAxes)        
    axs[1,0].set_xlim([0, 30])
    axs[1,0].set_xlabel("Time (days)", fontsize=fsize)
    axs[1,0].set_ylim(
        [LS_dict['median_MAD_nLC'][0]-(8.*LS_dict['median_MAD_nLC'][1]),
        LS_dict['median_MAD_nLC'][0]+(8.*LS_dict['median_MAD_nLC'][1])])
    axs[1,0].set_ylabel("normalised flux", c='g', fontsize=fsize)
    axs[1,0].plot(clean["time0"], LS_dict['y_fit_LS'], c='orange',
                  linewidth=0.5, label='LS best fit')
    axs[1,0].scatter(t_orig0, orig["nflux"], s=0.5, alpha=0.3,
                     label='raw, normalized')
    axs[1,0].scatter(clean["time0"], clean["oflux"],s=0.5, c='r', alpha=0.5,
                     label='cleaned, normalized')
    axs[1,0].scatter(clean["time0"], clean["nflux"],s=1.2, c='g', alpha=0.7,
                     label='cleaned, normalized, detrended')
    axs[1,0].text(0.99,0.90, Gaia_name, fontsize=lsize,
                  horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.80, f"Gmag = {float(t_table['Gmag']):.3f}",
                  fontsize=lsize, horizontalalignment='right',
                  transform=axs[1,0].transAxes)
    axs[1,0].text(0.99,0.70, "$\log (f_{\\rm bg}/f_{*})$ = "
                  f"{float(t_table['log_tot_bg']):.3f}", fontsize=lsize,
                  horizontalalignment='right', transform=axs[1,0].transAxes)
    axs[1,0].legend(loc='lower right')
    ax2=axs[1,0].twinx()
    ax2.set_position([0.05,0.3,0.90,0.2])
    ax2.invert_yaxis()
    ax2.scatter(t_orig0, orig["mag"], s=0.3, alpha=0.3, color="b", marker="x")
    ax2.set_ylabel("TESS magnitude", c="b",fontsize=fsize)

    axs[1,1].set_xlim([0,1])
    axs[1,1].set_xlabel("phase", fontsize=fsize)
    axs[1,1].set_ylabel("normalised flux", fontsize=fsize)
    axs[1,1].plot(LS_dict['phase_fit_x'], LS_dict['phase_fit_y'], c='b')
    LS_dict["phase_col"] += 1
    N_cyc = int(max(LS_dict["phase_col"]))
    cmap_use = plt.get_cmap('rainbow', N_cyc)
    s = axs[1,1].scatter(LS_dict['phase_x'], LS_dict["phase_y"],
                         c=LS_dict['phase_col'], cmap=cmap_use, vmin=0.5,
                         vmax=N_cyc+0.5)
    axs[1,1].text(0.01, 0.90, f"Amplitude = {LS_dict['pops_vals'][1]:.3f}, "
                  f"Scatter = {LS_dict['phase_scatter']:.3f}", fontsize=lsize,
                  horizontalalignment='left', transform=axs[1,1].transAxes)

    cbaxes = inset_axes(axs[1,1], width="100%", height="100%",
                        bbox_to_anchor=(0.79, 0.92, 0.20, 0.05),
                        bbox_transform=axs[1,1].transAxes)
    cbar = plt.colorbar(s, cax=cbaxes, orientation='horizontal',
                        label='cycle number')

    plot_name = '_'.join([name_underscore, f"{scc[0]:04d}",
                          str(scc[1]), str(scc[2])])+'.png'
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close('all')

