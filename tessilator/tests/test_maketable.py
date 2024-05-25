# Licensed under MIT license - see LICENSE file
import pytest
from astropy.table import Table
from tessilator import maketable


def check_ABDor(tab):
    """Check the AB Dor table.

    This check might fail when the next GAIA data release occurs because
    the magnitude will probably change. In that case, the test will need to
    be updated.

    parameters
    ----------
    tab : `astropy.table.Table`
        The table to check.
    """
    assert len(tab) == 1
    assert len(tab.colnames) == 8
    assert str(tab["source_id"][0]) == "Gaia DR3 4660766641264343680"
    assert tab["Gmag"][0] == pytest.approx(6.690593)


@pytest.mark.remote_data
def test_table_from_simbad():
    """Get a table from SIMBAD and check the output."""
    input = Table({"sadf": ["AB Dor"]})
    out = maketable.table_from_simbad(input)
    assert out["name"][0] == "AB Dor"
    check_ABDor(out)


@pytest.mark.remote_data
def test_table_from_coords():
    """Get a table from SIMBAD and check the output."""
    input = Table({"col1": [82.18736509782727], "col2": [-65.44796165891661]})
    out = maketable.table_from_coords(input)
    check_ABDor(out)
