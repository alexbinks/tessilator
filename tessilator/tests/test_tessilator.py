# Licensed under MIT license - see LICENSE file

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
def test_all_sources_sector_no_data(tmpdir):
    """Test the all_sources_sector function."""
    xy_pixel_data = tessilator.get_tess_pixel_xy(ABDor)
    # For GAIA numbers, the source_id comes out as int64
    xy_pixel_data["source_id"] = np.asarray(xy_pixel_data["source_id"], dtype=str)
    tTargets = join(ABDor, xy_pixel_data, keys="source_id")
    with chdir(tmpdir):
        tessilator.all_sources_sector(
            tTargets,
            scc=[34, 4, 3],
            make_plots=True,
            period_file="period",
            file_ref="ABDor",
            keep_data=True,
        )
    out = Table.read(f"{tmpdir}/period_4_3.csv")
    assert len(out) == 1

    for col in ABDor.colnames[2:]:
        assert out[col] == pytest.approx(ABDor[col])
    # This is special, because it uses a different column name
    assert out["original_id"][0] == ABDor["name"]
    # Just test a few representative columns
    assert out["Sector"][0] == 34
    assert out["n_conts"].mask[0]
    assert out["ap_rad"][0] == 1.0
    assert out["power_4"].mask[0]
    assert out["period_shuffle"].mask[0]


# This is the example in Fig 9 in the tessilator draft paper
gaia_src = Table(
    {
        "name": ["6131349562059299712"],
        "source_id": ["6131349562059299712"],
        "ra": [185.9092860303274],
        "dec": [-46.1056960870971],
        "parallax": [43.79235599149152],
        "Gmag": [12.425029],
        "BPmag": [13.979581],
        "RPmag": [11.20509],
    }
)


@pytest.mark.remote_data
def test_all_sources_sector(tmpdir):
    """Test the all_sources_sector function."""
    xy_pixel_data = tessilator.get_tess_pixel_xy(gaia_src)
    # For GAIA numbers, the source_id comes out as int64
    xy_pixel_data["source_id"] = np.asarray(xy_pixel_data["source_id"], dtype=str)
    tTargets = join(gaia_src, xy_pixel_data, keys="source_id")
    with chdir(tmpdir):
        tessilator.all_sources_sector(
            tTargets,
            scc=[64, 2, 4],
            make_plots=True,
            period_file="period",
            file_ref="Fig9",
            keep_data=True,
        )
    out = Table.read(f"{tmpdir}/period_4_3.csv")
    assert len(out) == 1

    for col in gaia_src.colnames[2:]:
        assert out[col] == pytest.approx(gaia_src[col])
    # This is special, because it uses a different column name
    assert out["original_id"][0] == gaia_src["name"]
    # Just test a few representative columns
    assert out["Sector"][0] == 64
    assert out["n_conts"].mask[0]
    assert out["ap_rad"][0] == 1.0
    assert out["power_4"].mask[0]
    assert out["period_shuffle"].mask[0]
    assert False