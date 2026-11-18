# GTA VI Twitch CCU forecast — make.fun `gta6-twitch-ccu-forecast-01`

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alim-14-code/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alim-14-code/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry: `main.py` (Python 3.13, standard library only). Run by the grader as
`python3 /bot/main.py`; locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input dir, writes `<id>.out` (23 rows, hours
2–24, five quantiles each at 0.05 / 0.25 / 0.5 / 0.75 / 0.95) to the output
dir. No network, no randomness, no state — two runs are byte-identical.

## Model

Forecasts launch-day shapes in log space, relative to hour 1's print.
For a past launch, `r_h = viewers(h) / viewers(1)`. Hour 1 (00:00–01:00 UTC)
only catches regions whose midnight unlock already happened, so the shape
is dominated by the size of hour 1: across the 37 usable comparables,
`corr(log viewers(1), log r_h)` is ~0 at hour 2 and about −0.94 at hours
18–24. Huge hour-1 launches stay flat with a mild US-evening rise; tiny
hour-1 launches (unlock later in the day) explode later, up to 1000×.

Each hour's quantiles blend two estimates in log space (0.5 / 0.5):

1. **Kernel-weighted empirical quantiles** of `log r_h` — Gaussian kernel on
   `log viewers(1)` (bandwidth 1.0), so a huge hour 1 borrows its shape from
   the biggest past launches (ELDEN RING, Cyberpunk 2077, Diablo IV…).
   Launches whose unlock type differs (rolling midnight local vs one
   simultaneous global unlock) get weight ×0.14.
2. **Per-hour QUADRATIC OLS** of `log r_h` on `log viewers(1)` (centred,
   with a squared term — the log_v1 → shape curve bends) with Gaussian
   residual quantiles, clipped to the donors' range so nothing extrapolates
   past the biggest launch.

The blended deviation from the median is scaled by 0.9 and the whole curve
shifted up by +0.05 log units (the LOO median is biased low; a +0.05 shift
minimises pinball loss). If a case can't be forecast from its comparables,
it falls back to the house baseline (persistence + 0.15/√h random walk)
instead of crashing.

## Validation (leave-one-out on the 37 kit comparables, same pinball loss
on the log scale as the grader)

| model | mean loss | big hour-1 (≥40k) |
|---|---|---|
| this entry (v2, quadratic OLS + tuned blend/shift) | **0.383** | **~0.24** |
| v1 (linear OLS, 0.7/0.3 blend) | 0.396 | 0.250 |
| house baseline | 0.930 | 0.295 |
| unconditional empirical quantiles | 0.715 | 0.373 |

Tuning notes: a full grid search over bandwidth / blend weight / roll
penalty / spread / shift found the shipped config (bw 1.0, kernel 0.5,
roll 0.5, spread 0.9, shift +0.05). A PCA functional model (0.509) and
pre-launch history features (residual corr −0.12…−0.27) did not help and
were not used. Live practice rank before v2: 5066/8,256 (loss 0.222).

Also verified: output format on all 4 practice cases (strict checker:
exactly one `hours` key, 23 rows, hours 2–24, 5 finite quantiles each in
(0, 1e9], non-decreasing, ≤65,536 bytes), byte-identical reruns, ~0.15 s
per case (limit 8 s), and edge cases (hour-1 = 2,000,000; hour-1 = 164;
hour-1 = 1; no comparables; no hour-1 print — all still produce valid
output via the model or the baseline fallback).

## Notes

- The practice cases' answers are not in the kit (server-graded), so model
  selection used leave-one-out over the comparables themselves; the final
  case adds the 4 practice launches as extra donors (42 comparables).
- `streams` / `categories_hourly` (present only in the final case) are not
  used: they carry no signal the comparables don't already give, and the
  practice cases can't validate them.
- GTA VI's own pre-release category history is a comparable but has no
  release day, so it is automatically excluded as a shape donor.
