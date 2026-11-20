"""Trophy rarity forecaster for gta6-trophy-rarity-forecast-01.

Reads one case file (every Exophase read of the GTA VI PS5 trophy page made
before the case's checkpoint) and writes five quantiles per trophy for its
earn rate at the case's target time.

Model, in plain words:

* The house reads the page every few minutes, but Exophase only refreshes
  its numbers now and then, so most consecutive reads are identical.  The
  series is first collapsed to the distinct snapshots it really contains.
* On the logit scale a trophy's rate drifts roughly linearly in log-time
  (fast early movement that flattens out), so each trophy gets a weighted
  least-squares line of logit(rate) against ln(1 + hours since the first
  snapshot).  The line is read at the target time.
* One trophy's fitted slope is noisy, so slopes are partially pooled: each
  slope is shrunk toward the median slope of its own tier (platinum, gold,
  silver, bronze), by an amount that depends on how many distinct snapshots
  support it, and the total predicted move is damped and clamped.
* The starting point is smoothed the same way: the median sits halfway
  between the last read's logit and the fitted line's value at the last
  snapshot, so one stale or jumpy final refresh cannot drag the whole
  forecast with it.
* The spread combines three honest sources of uncertainty: the day-to-day
  volatility of logit rates between distinct snapshots (pooled across
  trophies, blended with the trophy's own), the slope's standard error
  projected over the forecast gap in log-time, and the fit's residual
  scatter.  Quantiles are the median plus z * sigma on the logit scale.

Standard library only, no network, no randomness: two runs on the same case
write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# How far back (in hours of snapshots) the trend line is fitted on.  The
# final case holds about sixty hours of history, so this keeps all of it
# while still letting a longer history forget the launch-night chaos.
FIT_WINDOW_HOURS = 96.0
# Partial pooling: a trophy with POOL_K or more effective snapshots keeps
# most of its own slope; fewer and it leans on its tier's median slope.
POOL_K = 6.0
# A log-time trend extrapolated over days is damped a little: real trophy
# drift saturates as the tracked player pool matures.
TREND_DAMPING = 0.9
# The predicted logit move over the whole horizon is never allowed past
# this, however steep the recent line looks.
MAX_LOGIT_MOVE = 2.2
# Pooled daily logit volatility is clamped into this band.
SIGMA_DAY_MIN = 0.25
SIGMA_DAY_MAX = 0.95
# The total quantile spread (logit units) is clamped into this band.
SIGMA_TOTAL_MIN = 0.15
SIGMA_TOTAL_MAX = 2.4


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_of(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def pct_of_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, for positive x; whole numbers as int."""
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def z_for_level(level):
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if level < plow:
        q = math.sqrt(-2 * math.log(level))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if level > phigh:
        q = math.sqrt(-2 * math.log(1 - level))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = level - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def distinct_snapshots(history):
    """Collapse consecutive reads with identical players and rates."""
    snaps = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _, read in snaps]


def fit_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, resid_sd, b_se)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0), 0.0, 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    resid_sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / max(1e-9, (sw / len(xs)) * dof))
    b_se = resid_sd / math.sqrt(sxx) if resid_sd > 0 else 0.0
    return a, b, resid_sd, b_se


def forecast_case(case):
    levels = case["levels"]
    zs = [z_for_level(float(t)) for t in levels]
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        return {"trophies": [{"id": t["id"], "q": [1.0] * len(levels)} for t in trophies]}

    snaps = distinct_snapshots(history)
    t_first = parse_time(snaps[0]["at"])
    times_h = [(parse_time(s["at"]) - t_first).total_seconds() / 3600.0 for s in snaps]
    last_read = history[-1]
    horizon_days = float(case["horizon_minutes"]) / 1440.0
    x_last = math.log1p(times_h[-1])
    x_target = math.log1p(times_h[-1] + float(case["horizon_minutes"]) / 60.0)
    dx = max(0.0, x_target - x_last)

    # Per-trophy series over the fit window, on distinct snapshots only.
    series = []
    for i in range(len(trophies)):
        pts = []
        for s, th in zip(snaps, times_h):
            if th < times_h[-1] - FIT_WINDOW_HOURS:
                continue
            rate = (s.get("rates") or [None] * len(trophies))[i]
            if rate is None:
                continue
            pts.append((math.log1p(th), logit_of(rate)))
        last_rate = (last_read.get("rates") or [None] * len(trophies))[i]
        if last_rate is None:
            for s in reversed(snaps):
                r = (s.get("rates") or [None] * len(trophies))[i]
                if r is not None:
                    last_rate = r
                    break
        series.append((pts, last_rate))

    # First pass: raw slopes, residual scatter and step volatility.
    raw = []
    step_vols = []
    for pts, _ in series:
        if len(pts) >= 3 and pts[-1][0] - pts[0][0] > 0.05:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [0.5 + 0.5 * (j / (len(pts) - 1)) for j in range(len(pts))]
            a, b, rsd, bse = fit_line(xs, ys, ws)
        else:
            a, b, rsd, bse = (pts[-1][1] if pts else 0.0), 0.0, 0.0, 0.0
        raw.append((a, b, rsd, bse, len(pts)))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            # x is ln(1+hours); a step's clock gap in days is recovered from
            # the x positions of its two snapshots on that curve.
            hours_gap = math.expm1(x1) - math.expm1(x0)
            step_vols.append(abs(y1 - y0) / math.sqrt(max(1e-6, hours_gap / 24.0)))
    pooled_vol = median(step_vols) if step_vols else 0.5
    pooled_vol = min(SIGMA_DAY_MAX, max(SIGMA_DAY_MIN, pooled_vol / 0.6745 if step_vols else 0.5))
    pooled_resid = median([r[2] for r in raw if r[2] > 0]) if any(r[2] > 0 for r in raw) else 0.3

    # Tier medians of the raw slopes, for partial pooling.
    tier_slopes = {}
    for t, r in zip(trophies, raw):
        tier_slopes.setdefault(t.get("tier", "?"), []).append(r[1])
    tier_median = {k: median(v) for k, v in tier_slopes.items()}
    global_median = median([r[1] for r in raw]) if raw else 0.0

    rows = []
    for t, (pts, last_rate), (a, b, rsd, bse, npts) in zip(trophies, series, raw):
        base_logit = logit_of(last_rate if last_rate is not None else 1.0)
        if npts >= 3:
            # Smooth the starting point: halfway between the last read and
            # the fitted line's value at the last snapshot.
            base_logit = 0.5 * base_logit + 0.5 * (a + b * x_last)
        tm = tier_median.get(t.get("tier", "?"), global_median)
        keep = npts / (npts + POOL_K) if npts else 0.0
        slope = tm + keep * (b - tm)
        move = TREND_DAMPING * slope * dx
        if last_rate is not None and (last_rate >= 99.9 or last_rate <= 0.02):
            move *= 0.5  # rates pinned to a boundary move less freely
        move = min(MAX_LOGIT_MOVE, max(-MAX_LOGIT_MOVE, move))
        med_logit = base_logit + move

        own_vol = 0.0
        vols = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            hours_gap = math.expm1(x1) - math.expm1(x0)
            vols.append(abs(y1 - y0) / math.sqrt(max(1e-6, hours_gap / 24.0)))
        if vols:
            own_vol = min(SIGMA_DAY_MAX, max(SIGMA_DAY_MIN, (median(vols) / 0.6745)))
        sigma_day = 0.5 * pooled_vol + 0.5 * own_vol if own_vol else pooled_vol
        resid = rsd if rsd > 0 else pooled_resid
        sigma = math.sqrt(sigma_day ** 2 * horizon_days + (bse * dx) ** 2 + 0.25 * resid ** 2)
        sigma = min(SIGMA_TOTAL_MAX, max(SIGMA_TOTAL_MIN, sigma))

        qs = []
        for z in zs:
            qs.append(round4(clamp_pct(pct_of_logit(med_logit + z * sigma))))
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
