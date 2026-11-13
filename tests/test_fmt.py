import math

from betterybalance.core.fmt import eu


def test_eu_formatting():
    assert (
        eu(1234.5, 1) == "1.234,5"
        and eu(-0.25, 2) == "-0,25"
        and eu(math.nan) == "n.v.t."
        and eu(1e6, 0) == "1.000.000"
    )


def test_eu_never_prints_a_negative_zero():
    assert eu(-0.001) == "0,00" and eu(-0.0) == "0,00" and eu(-0.4, 0) == "0"
    assert eu(-0.004, 2) == "0,00" and eu(-0.005001, 2) == "-0,01" and eu(-1e-12, 1) == "0,0"
    assert eu(-1234.0, 0) == "-1.234" and eu(-0.25, 2) == "-0,25"
