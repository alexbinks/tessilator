# Licence: MIT

import pytest
from tessilator.acf_functions import process_LightCurve


def test_process_LightCurve_prot_prior_value_error():
    """Test we error out with invalid prot_prior value. The other parameters don't matter in this case."""
    with pytest.raises(ValueError):
        process_LightCurve(None, prot_prior="invalid")