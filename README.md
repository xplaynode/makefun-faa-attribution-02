# FAA attribution forecaster — polymarket-faa-attribution-02

Entry for make.fun (@0xsfh). `python3 main.py` reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `<case>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (else `/output`), in the `makefun/polymarket-case/v1`
format: for each market, 240 rows (one per minute) of 5 quantiles
[0.05, 0.25, 0.5, 0.75, 0.95] for the Yes price. Python 3 stdlib only, no
network, fully deterministic (same input -> byte-identical output).

## How it works (3 lines)

1. Median = the market's last price — the crowd price is the prior; drift
   terms were tested and rejected (they consistently hurt in backtests).
2. Spread is re-estimated per market, per case from that case's own history:
   a winsorized, recency-weighted per-minute sigma (geometric mean of L1 and
   RMS step scales; min of a halflife-weighted full-history estimate and a
   last-120-minutes estimate), grown sub-diffusively as `sigma * h**0.46`.
3. Outer quantiles widen extra with horizon (`x (1 + 0.3 * h/240)`) as cheap
   jump insurance; everything is clamped to [0, 1], so near-zero markets get
   naturally asymmetric bands.

The structural edge over the house entries: the house baseline uses one
fixed sigma (0.0025) for every market and case, and the house model forecast
fixes one (sigma, drift) per market from a snapshot 1h before the deadline
and applies it to all 12 checkpoints. This program re-reads each case's full
history at each checkpoint and adapts to the regime that is actually in the
data — tight in quiet grinds, wide within minutes of a fresh shock.

## Validation (local, 2026-11-14)

Scored with an exact replica of the house grid + pinball scorer (the replica
reproduces the house's published practice baseline loss to 6 decimals):

| set | house baseline | this program |
| --- | --- | --- |
| kit practice cases p00–p05 (NVIDIA "Next CEO" family) | 0.363589 | **0.285231** |
| FAA pseudo-cases (12 checkpoints from the live markets' own history) | 0.259050 | **0.151678** |

It beats the baseline on every practice case and is never worse than the
baseline on any FAA pseudo-case. Runtime: the full 6-case practice set
(128 markets) completes in ~1.1 s total, vs the 8 s/case limit; outputs are
~215–280 KB vs the 1 MiB cap. Constants were picked by a tournament over
~40 variants on both sets, preferring robust (both-set) winners over
single-set optima.

## Market snapshot baked in at build time (2026-11-14 ~07:25Z, Gamma API)

The four attribution markets for the Nov 13, 2026 FAA nationwide ground stop
(event created 2026-11-13 ~21:35–22:20Z; first prints were wild — Iran 0.515,
foreign 0.255, AI 0.215, hack 0.155 — and collapsed within hours):

| market | Yes price |
| --- | --- |
| Attributed to Iran | 0.016 |
| Attributed to a foreign actor | 0.0325 |
| Attributed to a hack | 0.043 |
| Attributed to an AI sandbox escape | 0.006 |

Sum ≈ 0.098 → the crowd puts ~90% on "none of these". News state at build
time: the outage was caused by an update to the L3Harris-operated FTI
private comms network that rebooted and crashed, with aging switches/routers
overloaded (Transportation Sec. Duffy: legacy infrastructure); the FBI does
not believe it was a cyberattack; DHS Sec. Mullin: no indication of a
hostile attack, "mechanically driven", nothing ruled out; the FAA's AI
advisory tool SMART was denied any role by its maker and by Duffy.

The snapshot informs the *design* (calibration over directional bets — the
edge is volatility + tail calibration, not picking a culprit), but the
program itself is purely data-driven from each case: a hard-coded prior
would be stale by checkpoint h11, 44 hours after the deadline.

## Files

- `main.py` — the forecaster (only entrypoint).
- `LICENSE` — MIT, copied from the kit starter.
- `.makefun/owner.json` — claim wallet for this entry.
