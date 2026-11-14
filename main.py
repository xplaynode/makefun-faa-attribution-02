#!/usr/bin/env python3
"""make.fun polymarket-faa-attribution-02 — quantile forecaster.

Task: for each Polymarket market in a case, forecast the Yes price at every
minute of the next `horizon_minutes` as 5 quantiles (pinball-loss scoring).

Method (chosen by local tournament against the house baseline on the kit's 6
practice cases plus pseudo-cases built from the live FAA markets' own CLOB
history; baseline practice loss 0.363589 -> this program 0.285 locally):

  * Median = last observed price. The crowd price is the prior; no drift term
    survived backtesting (drift extrapolation consistently hurt).
  * Spread is per-market and per-case, re-estimated from the case's own price
    history — the structural edge over the house model forecast, which fixes
    one (sigma, drift) per market for all 12 checkpoints.
  * Sigma (per-minute) is the geometric mean of two winsorized scale
    estimators of the 1-minute increments — an L1 (mean absolute step) and an
    RMS estimate — computed two ways (halflife-240min weighted over the full
    history, and flat over the last 120 minutes) and combined with min(), so
    the estimate tracks the *current* regime: quiet grind -> tight bands,
    fresh shock -> wide bands within minutes. Increments are winsorized at
    0.008 so a single news spike cannot blow up the whole term structure.
  * Term structure is sub-diffusive: sigma_h = sigma * h**0.46. Realized
    h-step moves in these markets grow slower than sqrt(h) (spikes retrace,
    vol decays after news), and backtests prefer 0.46 over 0.5 on both the
    practice family and the FAA family.
  * The two outer quantiles get extra tail growth with horizon
    (x (1 + 0.3 * h / horizon)): cheap jump insurance that costs almost
    nothing in quiet windows and pays when a headline lands mid-window.
  * Everything is clamped to [0, 1]; near-zero prices therefore get naturally
    asymmetric bands. Fully deterministic: no randomness, no clock, no I/O
    beyond the case files. Stdlib only.

Market snapshot baked in at build time (2026-11-14 ~07:25Z, Gamma API):
Iran 0.016, foreign actor 0.0325, hack 0.043, AI sandbox escape 0.006 —
sum ~0.098, i.e. the crowd puts ~90% on "none of these". News state: the
Nov 13 FAA ground stop was an L3Harris FTI network update that rebooted and
crashed aging infrastructure; the FBI does not believe it was a cyberattack
and DHS calls it mechanically driven. The program itself is purely
data-driven from each case (a hard-coded prior would be stale by the last
checkpoint, 44h after the deadline); the snapshot lives in the README.
"""

import glob
import json
import math
import os

# --- tuned constants (local tournament, see README) ---
SIG_HL = 240          # halflife (min) for the full-history sigma estimate
SIG_RECENT = 120      # window (min) for the recent-regime sigma estimate
WINSOR = 0.008        # winsorize |1-min increment| at this size for sigma
SIG_FLOOR = 0.0004    # per-minute sigma floor
SIG_CAP = 0.004       # per-minute sigma cap
H_EXP = 0.46          # sub-diffusive term-structure exponent
TAIL_GROWTH = 0.3     # extra outer-quantile widening at h = horizon
BASE_SIGMA = 0.0025   # house-baseline sigma, used only for tiny histories

# Normal quantile function via Acklam's rational approximation
# (max relative error ~1e-9), so any quantile list a case carries is
# handled exactly — the spec's set is [0.05, 0.25, 0.5, 0.75, 0.95].
_ACK_A = [-3.969683028665376e+01, 2.209460984245205e+02,
          -2.759285984469534e+02, 1.383577518672690e+02,
          -3.066479806614716e+01, 2.506628277459239e+00]
_ACK_B = [-5.447609879822406e+01, 1.615858368580409e+02,
          -1.556989798598866e+02, 6.680131188771972e+01,
          -1.328068155288572e+01]
_ACK_C = [-7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
_ACK_D = [7.784695709041462e-03, 3.224671290700398e-01,
          2.445134137142996e+00, 3.754408661907416e+00]


def _z(tau):
    if tau <= 0.0:
        return -3.090232
    if tau >= 1.0:
        return 3.090232
    plow = 0.02425
    if tau < plow:
        q = math.sqrt(-2.0 * math.log(tau))
        return (((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q
                  + _ACK_C[3]) * q + _ACK_C[4]) * q + _ACK_C[5]) / \
               ((((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q
                 + _ACK_D[3]) * q + 1.0)
    if tau > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - tau))
        return -(((((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q
                   + _ACK_C[3]) * q + _ACK_C[4]) * q + _ACK_C[5]) / \
                ((((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q
                  + _ACK_D[3]) * q + 1.0)
    q = tau - 0.5
    r = q * q
    return (((((_ACK_A[0] * r + _ACK_A[1]) * r + _ACK_A[2]) * r
              + _ACK_A[3]) * r + _ACK_A[4]) * r + _ACK_A[5]) * q / \
           (((((_ACK_B[0] * r + _ACK_B[1]) * r + _ACK_B[2]) * r
              + _ACK_B[3]) * r + _ACK_B[4]) * r + 1.0)


def _mix_scale(weighted_steps):
    """Geometric mean of winsorized L1 and RMS scale estimates.

    weighted_steps: list of (increment, weight). Returns a per-minute sigma.
    """
    num_l1 = 0.0
    num_r2 = 0.0
    den = 0.0
    for v, w in weighted_steps:
        a = abs(v)
        if a > WINSOR:
            a = WINSOR
        num_l1 += w * a
        num_r2 += w * a * a
        den += w
    if den <= 0.0:
        return 0.0
    l1 = (num_l1 / den) * math.sqrt(math.pi / 2.0)
    rms = math.sqrt(num_r2 / den)
    if l1 > 0.0 and rms > 0.0:
        return math.sqrt(l1 * rms)
    return max(l1, rms)


def _sigma(steps):
    """Current-regime per-minute sigma from 1-minute increments."""
    n = len(steps)
    weighted = [(v, 0.5 ** ((n - 1 - i) / SIG_HL)) for i, v in enumerate(steps)]
    sig_long = _mix_scale(weighted)
    recent = [(v, 1.0) for v in steps[-SIG_RECENT:]]
    sig_recent = _mix_scale(recent)
    sig = min(sig_long, sig_recent)
    if sig < SIG_FLOOR:
        sig = SIG_FLOOR
    if sig > SIG_CAP:
        sig = SIG_CAP
    return sig


def _q6(v):
    r = round(min(1.0, max(0.0, v)), 6)
    return r + 0.0  # normalize -0.0


def forecast_market(prices, horizon, taus):
    """Return `horizon` rows of quantile forecasts for one market."""
    if not prices:
        prices = [0.5]
    if not all(math.isfinite(v) for v in prices):
        # Defensive: drop non-finite prints; if nothing usable remains,
        # emit a flat 0.5 forecast rather than crash the case.
        prices = [v for v in prices if math.isfinite(v)]
        if not prices:
            return [[0.5] * len(taus) for _ in range(horizon)]
    x = prices[-1]
    zs = [_z(t) for t in taus]
    if len(prices) < 30:
        # Tiny history: fall back to the house-baseline shape around x.
        rows = []
        for h in range(1, horizon + 1):
            s = BASE_SIGMA * math.sqrt(h)
            rows.append([_q6(x + z * s) for z in zs])
        return rows
    steps = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    sig = _sigma(steps)
    last = len(zs) - 1
    rows = []
    for h in range(1, horizon + 1):
        s = sig * (h ** H_EXP)
        tg = 1.0 + TAIL_GROWTH * (h / horizon)
        row = []
        for t, z in enumerate(zs):
            mult = tg if t == 0 or t == last else 1.0
            row.append(_q6(x + z * s * mult))
        rows.append(row)
    for row in rows:
        for t in range(1, len(row)):
            if row[t] < row[t - 1]:
                row[t] = row[t - 1]
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path) as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes", 240))
        taus = case.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])
        forecasts = []
        for m in case["markets"]:
            rows = forecast_market(m["prices"], horizon, taus)
            forecasts.append({"id": m["id"], "q": rows})
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out_dir, stem + ".out"), "w") as f:
            json.dump({"forecasts": forecasts}, f)


if __name__ == "__main__":
    main()
