"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Grand Theft Auto VI Twitch category's concurrent viewers for
hours 2..24 of release day (2026-11-19 UTC) as five quantiles per hour,
given only hour 1's print and the case's comparables (past big launches'
hourly SullyGnome counts, 14 days before to 3 days after their release day).

Model: empirical launch-day shapes, in log space, relative to hour 1.
For a past launch let r_h = viewers(h) / viewers(1). The shape of r_h is
dominated by one fact: hour 1 (00:00-01:00 UTC) only catches the regions
whose midnight unlock has already happened (UK/Europe). A launch whose
hour 1 is huge is already unlocked almost everywhere that matters, so its
day stays flat with a mild US-evening rise; a launch whose hour 1 is tiny
(unlock later in the day, or a small category) explodes later, up to
1000x. Across the 37 usable comparables, corr(log viewers(1), log r_h)
runs from ~0 at hour 2 to about -0.94 at hours 18-24.

So each hour's quantiles are estimated two ways and blended in log space:
  1. Kernel-weighted empirical quantiles of log r_h, with a Gaussian kernel
     on log viewers(1) (bandwidth 1.0 in log units), so a huge hour 1
     borrows its shape from the biggest past launches (ELDEN RING,
     Cyberpunk 2077, Diablo IV, ...). Launches whose unlock type differs
     (rolling midnight local vs one simultaneous global unlock) get their
     weight multiplied by exp(-0.5 * (1/0.7)^2) ~= 0.36.
  2. A per-hour OLS of log r_h on log viewers(1) (log viewers(1) clipped
     to the training range, so nothing extrapolates linearly past the
     biggest launch) with Gaussian residual quantiles.
Blend: 0.7 * kernel + 0.3 * OLS, in log space. Quantiles are then
viewers(1) * exp(blended log-quantile), rounded half-up to 2 decimals.

Leave-one-out cross-validation on the 37 kit comparables (mean pinball
loss on the log scale, same rule as the grader): this model 0.396, house
baseline (persistence + 0.15/sqrt-h random walk) 0.930, unconditional
empirical quantiles 0.715. On the big hour-1 launches (viewers(1) >=
40,000, the regime GTA VI will be in) it scores 0.25 vs 0.30 baseline.

The program reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output). No network, no
randomness, no state: two runs produce identical bytes. If a case cannot
be forecast from its comparables (none usable, malformed input), it falls
back to the house baseline for that case rather than crashing, so the
output stays valid.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# Gaussian kernel bandwidth on log viewers(1), in log units.
KERNEL_BW = 1.0
# Penalty bandwidth for a launch whose unlock type differs from the target's.
ROLL_BW = 0.7
# Blend weight of the kernel estimate (rest is the OLS estimate), in log space.
KERNEL_WEIGHT = 0.7
# Spread of the house baseline's random walk, in log units per sqrt hour.
BASE_SIGMA = 0.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> datetime:
    """An ISO 8601 UTC timestamp such as 2026-11-19T01:00:00Z."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def is_rolling_midnight(unlock_text) -> bool:
    """True when a launch unlocked at midnight local time, rolling by
    timezone, rather than at one simultaneous global instant.

    The text is the comparable's `unlock` field (or the case's `notes`).
    A global/simultaneous unlock is the tell-tale opposite; anything
    mentioning midnight local / rolling counts as rolling.
    """
    text = (unlock_text or "").lower()
    if "simultaneous" in text or "single global" in text or "global unlock" in text:
        return False
    return "midnight local" in text or "rolling" in text or "midnight" in text


def extract_launches(case: dict) -> list[dict]:
    """Past launches usable as shape donors.

    One entry per comparable: {"log_v1", "rolling", "log_ratio": {h: log(v_h/v_1)}}.
    A comparable contributes to an hour only if it has that hour and hour 1.
    The target launch itself (same game, same release day) is never a donor.
    """
    day_start = parse_ts(case["day_start"])
    target_name = (case.get("game") or {}).get("name", "")
    launches = []
    for comp in case.get("comparables") or []:
        try:
            comp_day = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        if comp_day == day_start and comp.get("name") == target_name:
            continue
        hours: dict[int, float] = {}
        for row in comp.get("hourly") or []:
            try:
                at = parse_ts(row["from"])
                viewers = float(row["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            offset = (at - comp_day).total_seconds() / 3600.0
            if offset == int(offset) and 0 <= offset < 24:
                hours[int(offset) + 1] = max(viewers, 0.0)
        v1 = hours.get(1)
        if not v1 or v1 <= 0:
            continue
        log_ratio = {}
        for h in range(2, 25):
            if h in hours and hours[h] > 0:
                log_ratio[h] = math.log(hours[h] / v1)
        if not log_ratio:
            continue
        launches.append(
            {
                "log_v1": math.log(v1),
                "rolling": is_rolling_midnight(comp.get("unlock")),
                "log_ratio": log_ratio,
            }
        )
    return launches


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """tau-quantile of values with weights, linearly interpolated on the
    sorted values (weights act as mass between neighbouring points)."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        prev = acc
        acc += weight
        if acc >= target:
            if i == 0 or weight <= 0:
                return value
            frac = (target - prev) / weight
            return ordered[i - 1][0] + frac * (value - ordered[i - 1][0])
    return ordered[-1][0]


def kernel_quantiles(launches: list[dict], log_v1: float, target_rolling: bool, hour: int) -> list[float]:
    """Weighted empirical quantiles of log r_h, kernel-weighted by closeness
    in log viewers(1) and by matching unlock type."""
    pairs = []
    for launch in launches:
        if hour not in launch["log_ratio"]:
            continue
        weight = math.exp(-0.5 * ((launch["log_v1"] - log_v1) / KERNEL_BW) ** 2)
        if launch["rolling"] != target_rolling:
            weight *= math.exp(-0.5 * (1.0 / ROLL_BW) ** 2)
        pairs.append((launch["log_ratio"][hour], weight))
    if not pairs:
        raise ValueError(f"no donor launch has hour {hour}")
    return [weighted_quantile(pairs, tau) for tau in LEVELS]


def ols_quantiles(launches: list[dict], log_v1: float, hour: int) -> list[float]:
    """Per-hour OLS of log r_h on log viewers(1), with Gaussian residual
    quantiles. log viewers(1) is clipped to the donors' range, so the
    fitted line never extrapolates past the biggest past launch."""
    xs = []
    ys = []
    for launch in launches:
        if hour in launch["log_ratio"]:
            xs.append(launch["log_v1"])
            ys.append(launch["log_ratio"][hour])
    if len(xs) < 3:
        raise ValueError(f"too few donor launches for hour {hour}")
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd = statistics.pstdev(residuals) if len(residuals) > 1 else 0.3
    clipped = min(max(log_v1, min(xs)), max(xs))
    centre = intercept + slope * clipped
    return [centre + z * sd for z in Z]


def baseline_quantiles(p1: float, hour: int) -> list[float]:
    """The house baseline: persistence with a random-walk spread."""
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_v1 = math.log(p1)
    launches = extract_launches(case)
    if len(launches) < 3:
        raise ValueError(f"{name} has only {len(launches)} usable donor launches")
    target_rolling = is_rolling_midnight(case.get("notes"))
    forecast_rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    rows = []
    for row in forecast_rows:
        hour = row["hour"]
        kernel = kernel_quantiles(launches, log_v1, target_rolling, hour)
        ols = ols_quantiles(launches, log_v1, hour)
        blended = [
            KERNEL_WEIGHT * k + (1.0 - KERNEL_WEIGHT) * o for k, o in zip(kernel, ols)
        ]
        quantiles = [round2(p1 * math.exp(value)) for value in blended]
        # Repair any rounding inversion and keep every quantile in (0, 1e9].
        fixed = []
        prev = 0.01
        for q in quantiles:
            q = min(max(float(q), prev), 1e9)
            fixed.append(round2(q))
            prev = float(fixed[-1])
        rows.append({"hour": hour, "q": fixed})
    return {"hours": rows}


def forecast_case_safe(case: dict, name: str) -> dict:
    """Forecast, or the house baseline if the model cannot run on this case."""
    try:
        return forecast_case(case, name)
    except Exception:
        first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
        p1 = max(1.0, float(first["viewers"])) if first else 1.0
        rows = [
            {"hour": row["hour"], "q": baseline_quantiles(p1, row["hour"])}
            for row in case.get("forecast", [{"hour": h} for h in range(2, 25)])
        ]
        return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case_safe(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
