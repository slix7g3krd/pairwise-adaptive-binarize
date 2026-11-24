import pytest
import imaging.adaptive_binarize as mod

CASES = [
    ("window_by_glyph", (1,)),
    ("integral_32bit", (1,)),
    ("params_per_page", (1,)),
    ("background_normalize", (1,)),
    ("window_cross_tile", (1,)),
    ("check_stitch", (1,)),
    ("load", (1,)),
    ("binarize", (1,)),
    ("stitch", (1,)),
    ("export", (1,)),
    ("validate_binarize", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
