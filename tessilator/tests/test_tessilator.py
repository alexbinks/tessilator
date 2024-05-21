# Licensed under MIT license - see LICENSE file

import numpy as np
import pytest
from astropy.table import Table, join
from tessilator import tessilator

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
def test_all_sources_sector():
    """Test the all_sources_sector function."""
    xy_pixel_data = tessilator.get_tess_pixel_xy(ABDor)
    # For GAIA numbers, the source_id comes out as int64
    xy_pixel_data["source_id"] = np.asarray(xy_pixel_data["source_id"], dtype=str)
    tTargets = join(ABDor, xy_pixel_data, keys="source_id")
    tessilator.all_sources_sector(
        tTargets,
        scc=[34, 4, 3],
        make_plots=True,
        period_file="temp/period",
        file_ref="ABDor",
        keep_data=True,
    )
    # Put some real tests here. But that requires me to be able to run
    # it first...
    assert False