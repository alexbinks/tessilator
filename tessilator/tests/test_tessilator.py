# Licensed under MIT license - see LICENSE file
import os
import glob
import numpy as np
import pytest
from astropy.table import Table, join
from tessilator import tessilator
from contextlib import chdir

ABDor = Table(
    {
        "name": ["AB Dor"],
        "source_id": ["4660766641264343680"],
        "ra": [82.18736509782727],
        "dec": [-65.44796165891661],
        "parallax": [67.33308290357063],
        "Gmag": [6.690593],
        "BPmag": [7.1454396],
        "RPmag": [6.0462117],
    }
)


def test_xy_pixel_data():
    """Test the xy_pixel_data function."""
    out = tessilator.get_tess_pixel_xy(ABDor)
    assert len(out) >= 35  # might change when more sectors are added
    assert out["Sector"][0] == 1
    assert out["Xpos"][0] == pytest.approx(1655.3008491588157)
    # Don't want to hardcode all results, but check some general properties
    assert np.all(np.isin(out["Camera"], [1, 2, 3, 4]))
    assert np.all(np.isin(out["CCD"], [1, 2, 3, 4]))
    assert np.min(out["Xpos"]) >= 0
    assert np.min(out["Ypos"]) >= 0
    assert np.max(out["Xpos"]) <= 2048
    assert np.max(out["Ypos"]) <= 2048


@pytest.mark.remote_data
def test_all_sources_cutout(tmpdir):
    """Test the all_sources_cutout function."""
    with chdir(tmpdir):
        tessilator.all_sources_cutout(
            ABDor,
            period_file="period",
            lc_con=False,
            flux_con=False,
            make_plots=True,
            ref_name="ABDor",
            choose_sec=33,
        )
    assert os.path.isfile(f"{tmpdir}/fits/ABDor/AB_Dor_0033_4_2.fits")
    # Checking how a matplotlib file looks is more complicated,
    # so just check is exists.
    assert os.path.isfile(f"{tmpdir}/plots/ABDor/AB_Dor_0033_4_2_nc.png")
    out_csv = glob.glob(f"{tmpdir}/results/ABDor/*.csv")
    out = Table.read(out_csv[0])
    assert len(out) == 1

    for col in ABDor.colnames[2:]:
        if col in ["log_tot_bg", "log_max_bg", "num_tot_bg"]:
            assert col not in out.colnames
        else:
            assert out[col] == pytest.approx(ABDor[col])
    # This is special, because it uses a different column name
    assert out["original_id"][0] == ABDor["name"]
    # Just test a few representative columns
    assert out["Sector"][0] == 33
    assert out["num_tot_bg"].mask[0]
    assert (
        out["ap_rad"][0] == 1.926
    )  # The csv writer sets the precision, so this is exact
    assert out["Ndata"] == 3444
    assert out["period_shuffle"] == -999.0
    assert out["period_1"][0] == pytest.approx(0.514, rel=1e-3)


@pytest.mark.remote_data
def test_all_sources_cutout_no_data(tmpdir):
    """Test the all_sources_cutout function."""
    xy_pixel_data = tessilator.get_tess_pixel_xy(ABDor)
    # For GAIA numbers, the source_id comes out as int64
    xy_pixel_data["source_id"] = np.asarray(xy_pixel_data["source_id"], dtype=str)
    tTargets = join(ABDor, xy_pixel_data, keys="source_id")
    with chdir(tmpdir):
        tessilator.all_sources_sector(
            tTargets,
            scc=[16, 4, 3],
            make_plots=False,
            period_file="period",
            file_ref="ABDor",
            keep_data=False,
        )

    assert not os.path.exists(f"{tmpdir}/fits")
    assert not os.path.exists(f"{tmpdir}/plots")
    assert not os.path.exists(f"{tmpdir}/results")
