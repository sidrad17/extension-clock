"""Tests for src/bonds.py: par price, textbook duration, positive convexity."""
import numpy as np
import pandas as pd
import pytest

from src.bonds import Curve, accrued, convexity, mod_duration, price


@pytest.mark.parametrize("coupon", [0.5, 4.5, 8.0])
@pytest.mark.parametrize("years", [0.5, 2.0, 10.0, 30.0])
def test_par_bond_prices_at_100(coupon, years):
    assert price(coupon, coupon, years) == pytest.approx(100.0, abs=1e-9)
    assert price(coupon, coupon, years, clean=True) == pytest.approx(100.0, abs=1e-9)


def test_textbook_duration_10y_8pct():
    # 10-year 8% semiannual bond at an 8% yield: Macaulay 7.0670y, modified 6.7951y (closed form for a par bond:
    # D_mac = (1+i)/i * (1 - (1+i)^-n) half-years, i = 4%, n = 20).
    i, n = 0.04, 20
    mac = (1 + i) / i * (1 - (1 + i) ** -n) / 2
    assert mac == pytest.approx(7.0670, abs=1e-4)
    assert mod_duration(8.0, 8.0, 10.0) == pytest.approx(mac / (1 + i), abs=1e-10)
    assert mod_duration(8.0, 8.0, 10.0) == pytest.approx(6.7951, abs=1e-4)


def test_zero_coupon_duration_is_maturity():
    assert mod_duration(0.0, 5.0, 7.0) == pytest.approx(7.0 / 1.025, abs=1e-12)


def test_convexity_positive_and_matches_finite_difference():
    for c, y, T in [(4.0, 4.0, 10.0), (2.0, 5.5, 29.7), (6.0, 1.0, 1.3)]:
        h = 1e-3  # percent
        p0, pu, pd_ = price(c, y, T), price(c, y + h, T), price(c, y - h, T)
        dy = h / 100.0
        assert convexity(c, y, T) > 0
        assert convexity(c, y, T) == pytest.approx((pu - 2 * p0 + pd_) / (p0 * dy * dy), rel=1e-4)
        assert mod_duration(c, y, T) == pytest.approx(-(pu - pd_) / (2 * dy) / p0, rel=1e-6)


def test_fractional_first_period_accrued_and_clean():
    c, T = 4.0, 9.9  # 20 coupons left, first period 0.8 of a half-year
    assert accrued(c, 4.0, T) == pytest.approx(c / 2 * 0.2, abs=1e-12)
    assert price(c, 4.0, T) - price(c, 4.0, T, clean=True) == pytest.approx(accrued(c, 4.0, T), abs=1e-12)
    assert price(c, c, T, clean=True) == pytest.approx(100.0, abs=0.01)


def test_vectorized_matches_scalar_and_invalid_is_nan():
    c = np.array([1.0, 3.0, 5.0, 7.0])
    y = np.array([2.0, 3.5, 4.0, 6.5])
    T = np.array([1.2, 5.0, 9.75, 30.0])
    vec = price(c, y, T)
    assert np.allclose(vec, [price(*a) for a in zip(c, y, T)])
    assert np.allclose(mod_duration(c, y, T), [mod_duration(*a) for a in zip(c, y, T)])
    assert np.isnan(price(4.0, 4.0, 0.0)) and np.isnan(price(4.0, np.nan, 5.0))


def test_curve_linear_flat_ends_and_skips_missing_knots():
    d = pd.Timestamp("2005-06-30")
    frame = pd.DataFrame({"DGS1": [3.0], "DGS2": [3.5], "DGS5": [4.0], "DGS10": [4.5], "DGS20": [5.0],
                          "DGS30": [np.nan]}, index=[d])
    c = Curve(frame)
    assert c.yields(d, 1.5) == pytest.approx(3.25)
    assert c.yields(d, 7.5) == pytest.approx(4.25)
    assert c.yields(d, 0.25) == pytest.approx(3.0)      # flat below the shortest knot
    assert c.yields(d, 28.0) == pytest.approx(5.0)      # DGS30 missing: flat beyond DGS20
    assert np.allclose(c.yields(d, [2, 15]), [3.5, 4.75])
