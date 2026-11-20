# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rian0net/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Rian0net/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the earn rates of Grand Theft Auto VI's PlayStation 5
trophies on Exophase, built for the make.fun bounty
gta6-trophy-rarity-forecast-01.

`main.py` (Python 3.13, standard library only) reads a case file from the
input directory — every read of the game's trophy page before the case's
checkpoint — and writes five quantiles (5%, 25%, 50%, 75%, 95%) of each
trophy's earn rate at the case's target time to the output directory.

How it works:

- Consecutive identical reads are collapsed to the distinct snapshots the
  series really contains, because Exophase refreshes its numbers only now
  and then while the page is read every few minutes.
- Each trophy's logit earn rate is fitted with a weighted least-squares
  line against ln(1 + hours since the first snapshot); the line is read at
  the target time, damped, and clamped.
- Slopes are partially pooled toward the median slope of the trophy's tier
  (platinum, gold, silver, bronze), so one noisy trophy cannot run away
  with the forecast.
- The quantile spread combines pooled step volatility, the slope's
  standard error projected over the log-time gap, and the fit's residual
  scatter.

Run it the way the grader does:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same case write the same bytes.
