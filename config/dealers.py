"""H8 parameters (PREREG_DEALERS.md, tag `prereg-dealers`, commit 15f67bc).

`config/settings.py` stays locked at `gate1-prereg`, so the third pre-registration's numbers live here. Every value
below is stated in PREREG_DEALERS.md; the event return, zS and the week clusters are the Flow Clock's (H6c).
"""

ZD_PRIOR = 52             # zD: the 52 releases before the latest known one, in the same series segment
ZD_CLIP = 3.0             # zD clipped to [-3, 3]; +/-3 when the 52 values are all equal and X_0 differs
RELEASE_LAG_DAYS = 8      # publication = first bond business day on or after as-of + 8 calendar days
P_PASS = 0.05             # pass: c > 0 and one-sided p < 0.05
