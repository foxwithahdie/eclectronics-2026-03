"""
What component tolerances do to the oscillator's period.

Every resistor and capacitor in the timing loop is given a Gaussian spread
inside its tolerance band, and the period is measured for each combination.
Four ways of choosing the combinations:

  sweep       one part at a time across its Gaussian, the rest nominal.
              6 parts x 100 levels = 600 runs. Shows what each part does.
  montecarlo  all parts drawn together at random. This is the one that gives
              a yield, because it is the only mode that samples reality.
  factorial   every combination of every level. See the note below.
  corners     all 2^6 extremes. A hard bound, not a sample.

A note on "factorial": fully crossing 6 parts at 100 levels each is 100^6 =
10^12 runs. At the ~7 runs/s this gets out of LTspice that is about 4500 years,
so "factorial" is capped by MAX_RUNS and is only usable at a handful of levels
(5 levels = 15625 runs, about 40 minutes). For the real question -- how often
does a built board land inside +/-10% -- "montecarlo" with a few thousand draws
is both feasible and statistically the right answer, because a grid spends
almost all its points in corners that essentially never occur.

Two engines: "analytic" uses the closed-form period (instant, ideal op-amp) and
"spice" runs LTspice for real. Running both and comparing is the point; they
should agree to a fraction of a percent.

Everything you would want to change is in the SETTINGS block below the imports.
"""

from __future__ import annotations

import csv
import itertools
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

import analyze_vout

# =========================================================================== #
# SETTINGS -- edit these, then run:  python tolerance_simulation.py
# =========================================================================== #

# Which builds to simulate.
#   "sweep"       one part at a time across its Gaussian, the rest nominal.
#                 LEVELS per part x 6 parts. This is the per-part table.
#   "montecarlo"  all six parts drawn together at random, RUNS times.
#                 The only mode that gives a meaningful yield.
#   "corners"     all 2**6 = 64 tolerance extremes. A hard bound, not a sample.
#   "factorial"   every combination of every level: LEVELS**6 runs. See the
#                 note in the module docstring -- 100 levels is 10**12 runs, so
#                 this is only usable at about 5 levels. MAX_RUNS enforces that.
MODE: str = "montecarlo"

# "spice" runs LTspice for real. "analytic" uses the closed-form period: instant,
# but assumes an ideal op-amp. Running both and comparing is the point.
ENGINE: str = "spice"

# Gaussian levels per part, for "sweep" and "factorial".
LEVELS: int = 100

# Random draws, for "montecarlo".
RUNS: int = 2000

# Refuse any plan bigger than this, so a stray zero cannot start a run that
# would take years. Raise it deliberately if you mean to.
MAX_RUNS: int = 20000

# How the tolerance band is turned into a distribution.
#   "truncated"  Gaussian, resampled so nothing lands outside the band. Most
#                defensible: real shape, and the manufacturer's bound respected.
#   "gauss"      plain Gaussian, tails included. Lets through parts the
#                manufacturer promises do not exist.
#   "uniform"    flat across the band. The pessimistic reading.
DISTRIBUTION: str = "truncated"

# How many sigma the marked tolerance is worth. 3.0 is the industry reading
# (99.7% of parts inside the band). 2.0 is the conservative alternative, and
# it is worth running the whole thing both ways.
SIGMAS: float = 3.0

# Marked tolerances. Every resistor on the parts list is 1%; the 0603 X7R is 10%.
RESISTOR_TOLERANCE: float = 0.01
CAPACITOR_TOLERANCE: float = 0.10

# The design spec.
TARGET_PERIOD: float = 1.0
TOLERANCE_BAND: float = 0.10

# Where the nominal values come from. Anything set here is read out of the
# netlist, so the values are never typed in twice.
NETLIST: Path = Path("../mp1_sim.net")

# The .tran directive used for every run. "10 startup" gives about 9 cycles,
# which is plenty; lengthen it if you want more cycles averaged per run.
TRAN: str = "10 startup"

# Simulations per LTspice invocation. A wine start costs ~4 s and a solve ~0.1 s,
# so batching is what makes this tolerable. 200 is comfortable.
CHUNK: int = 200

# Seed for the random draws. Fixed so the numbers in the writeup reproduce.
SEED: int = 0

# How many rows to print. 0 prints all of them; the CSV always has everything.
SHOW_ROWS: int = 40

# Output files. Set PLOT_PATH to None to skip plotting.
CSV_PATH: Path = Path("tolerance_results.csv")
PLOT_PATH: Optional[Path] = Path("tolerance.png")

# Scratch directory for the LTspice runs. Cleaned up as it goes.
WORKROOT: Path = Path("_sweep")

# =========================================================================== #

# The parts in the timing loop, each varied independently. R6 and D1 are the
# LED branch and cannot move the period, so they are left out.
TIMING_PARTS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R7", "C1")

# Which of those add up to the timing resistance. One name if R5 is a single
# resistor, several if it is split across a series string -- the parts list is
# a sparse ladder and the value that is wanted falls in a hole, so two in
# series get much closer than any single stocked part. Each one carries its own
# independent tolerance, which is why they are sampled separately and summed
# rather than treated as one part.
TIMING_RESISTOR: tuple[str, ...] = ("R5", "R7")

# The timing capacitor.
TIMING_CAPACITOR: str = "C1"

# The two dividers, in the order the period formula wants them.
DIVIDER_PARTS: tuple[str, ...] = ("R1", "R2", "R3", "R4")


class Result(NamedTuple):
    """One simulated build."""

    values: dict[str, float]
    period: float
    duty_cycle: float
    label: str
    error: Optional[str] = None

    @property
    def error_percent(self) -> float:
        """How far the period is from target, in percent."""
        return (self.period / TARGET_PERIOD - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# The closed-form model (README section 3.6), vectorised
# --------------------------------------------------------------------------- #


def period_from_values(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    r4: np.ndarray,
    r5: np.ndarray,
    c1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge and discharge half-periods, in seconds. Broadcasts over arrays.

    Kept as two halves rather than the collapsed 2*R5*C1*ln3, because that
    shortcut assumes R1 == R2 and R3 == R4 and so hides exactly the mismatch
    this script exists to measure. VDD cancels and does not appear.
    """
    weight_reference = r4 / (r3 + r4)
    weight_output = r3 / (r3 + r4)
    divider_ratio = r2 / (r1 + r2)
    tau = r5 * c1

    charge = tau * np.log(
        (1.0 - weight_reference * divider_ratio)
        / (weight_reference * (1.0 - divider_ratio))
    )
    discharge = tau * np.log(
        (weight_reference * divider_ratio + weight_output)
        / (weight_reference * divider_ratio)
    )
    return charge, discharge


def timing_resistance(sample: dict[str, float]) -> float:
    """Total timing resistance: the series group added up."""
    present = [name for name in TIMING_RESISTOR if name in sample]
    if not present:
        raise KeyError(f"none of {TIMING_RESISTOR} is in the sample")
    return sum(sample[name] for name in present)


def evaluate_analytic(samples: Sequence[dict[str, float]]) -> list[tuple[float, float]]:
    """Period and duty cycle for every sample, from the closed form."""
    column = {
        part: np.array([sample[part] for sample in samples]) for part in DIVIDER_PARTS
    }
    r5 = np.array([timing_resistance(sample) for sample in samples])
    c1 = np.array([sample[TIMING_CAPACITOR] for sample in samples])

    divider = [column[part] for part in DIVIDER_PARTS]
    charge, discharge = period_from_values(
        divider[0], divider[1], divider[2], divider[3], r5, c1
    )
    total = charge + discharge
    return list(zip(total.tolist(), (charge / total).tolist()))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def levels_for(
    nominal: float,
    tolerance: float,
    count: int,
    sigmas: float = 3.0,
    distribution: str = "truncated",
) -> np.ndarray:
    """
    `count` values spanning one part's tolerance band.

    Quantiles of the distribution, not a linear ramp, so the levels crowd
    together near nominal exactly as the population does -- which is what
    "100 values on the Gaussian" has to mean if the levels are to be
    representative rather than uniformly spaced.
    """
    if count == 1:
        return np.array([nominal])

    # Midpoints of `count` equal-probability bins, so no level sits on the
    # infinite tail at p = 0 or p = 1.
    probabilities = (np.arange(count) + 0.5) / count

    if distribution == "uniform":
        spread = (2 * probabilities - 1) * tolerance
    else:
        sigma = tolerance / sigmas
        # numpy has no inverse normal CDF, and the stdlib does.
        normal = NormalDist(0.0, sigma)
        spread = np.array([normal.inv_cdf(p) for p in probabilities])
        if distribution == "truncated":
            spread = np.clip(spread, -tolerance, tolerance)

    return nominal * (1.0 + spread)


def draw(
    nominal: dict[str, float],
    tolerances: dict[str, float],
    count: int,
    rng: np.random.Generator,
    sigmas: float = 3.0,
    distribution: str = "truncated",
) -> list[dict[str, float]]:
    """Draw `count` builds with every part varied independently."""
    columns: dict[str, np.ndarray] = {}
    for part in nominal:
        tolerance = tolerances[part]
        if distribution == "uniform":
            spread = rng.uniform(-tolerance, tolerance, count)
        else:
            spread = rng.normal(0.0, tolerance / sigmas, count)
            if distribution == "truncated":
                # Resample rather than clip: clipping piles probability onto
                # the two band edges, which is not what a real reel looks like.
                for _ in range(64):
                    bad = np.abs(spread) > tolerance
                    if not bad.any():
                        break
                    spread[bad] = rng.normal(0.0, tolerance / sigmas, int(bad.sum()))
                spread = np.clip(spread, -tolerance, tolerance)
        columns[part] = nominal[part] * (1.0 + spread)

    return [{part: float(columns[part][i]) for part in nominal} for i in range(count)]


def build_plan(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    mode: str,
    nominal: dict[str, float],
    tolerances: dict[str, float],
    levels: int,
    runs: int,
    seed: int,
    sigmas: float,
    distribution: str,
    max_runs: int,
) -> tuple[list[dict[str, float]], list[str]]:
    """Produce the list of builds to simulate, and a label for each."""
    rng = np.random.default_rng(seed)

    if mode == "montecarlo":
        samples = draw(nominal, tolerances, runs, rng, sigmas, distribution)
        return samples, [f"draw {i + 1}" for i in range(len(samples))]

    if mode == "sweep":
        samples, labels = [], []
        for part in TIMING_PARTS:
            for value in levels_for(
                nominal[part], tolerances[part], levels, sigmas, distribution
            ):
                samples.append({**nominal, part: float(value)})
                labels.append(f"{part} {value / nominal[part] - 1:+.3%}")
        return samples, labels

    if mode == "corners":
        samples, labels = [], []
        for signs in itertools.product((-1, 1), repeat=len(TIMING_PARTS)):
            build = {
                part: nominal[part] * (1 + sign * tolerances[part])
                for part, sign in zip(TIMING_PARTS, signs)
            }
            samples.append(build)
            labels.append("".join("+" if sign > 0 else "-" for sign in signs))
        return samples, labels

    if mode == "factorial":
        total = levels ** len(TIMING_PARTS)
        if total > max_runs:
            affordable = int(max_runs ** (1 / len(TIMING_PARTS)))
            raise SystemExit(
                f'MODE = "factorial" at LEVELS = {levels} is '
                f"{levels}**{len(TIMING_PARTS)} = {total:,} runs, over "
                f"MAX_RUNS ({max_runs:,}).\n"
                f"At this cap you can afford LEVELS = {affordable} "
                f"({affordable ** len(TIMING_PARTS):,} runs). For a yield "
                'number set MODE = "montecarlo" instead: a few thousand joint '
                "draws answer the question a grid cannot afford to."
            )
        grids = [
            levels_for(nominal[part], tolerances[part], levels, sigmas, distribution)
            for part in TIMING_PARTS
        ]
        samples, labels = [], []
        for combination in itertools.product(*grids):
            samples.append(dict(zip(TIMING_PARTS, map(float, combination))))
            labels.append(
                " ".join(
                    f"{part}{value / nominal[part] - 1:+.1%}"
                    for part, value in zip(TIMING_PARTS, combination)
                )
            )
        return samples, labels

    raise SystemExit(
        f"unknown MODE {mode!r}; pick sweep, montecarlo, factorial or corners"
    )


# --------------------------------------------------------------------------- #
# The SPICE engine
# --------------------------------------------------------------------------- #


def evaluate_spice(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    samples: Sequence[dict[str, float]],
    netlist_path: Path,
    workroot: Path,
    tran: str = TRAN,
    chunk: int = 200,
    skip_cycles: float = 2.0,
    progress: bool = True,
) -> list[tuple[float, float, Optional[str]]]:
    """Run every build through LTspice and measure its waveform."""
    # pylint: disable=import-outside-toplevel  # analytic engine needs no wine
    import ltspice

    netlist = ltspice.Netlist.from_file(netlist_path)
    out: list[tuple[float, float, Optional[str]]] = [
        (float("nan"), float("nan"), "not run")
    ] * len(samples)

    for index, wave, failure in ltspice.run_sweep(
        netlist,
        samples,
        workroot,
        tran=tran,
        save=["V(vout)"],
        chunk=chunk,
        progress=progress,
    ):
        if wave is None:
            out[index] = (float("nan"), float("nan"), failure)
            continue
        try:
            timing = analyze_vout.measure(
                wave.time, wave.trace("V(vout)"), skip_cycles=skip_cycles
            )
        except (analyze_vout.NotOscillating, KeyError) as error:
            out[index] = (float("nan"), float("nan"), str(error))
            continue
        out[index] = (timing.period, timing.duty_cycle, None)

    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise(results: Sequence[Result]) -> dict[str, float]:
    """Yield and distribution over the runs that produced a period."""
    periods = np.array([r.period for r in results if math.isfinite(r.period)])
    if periods.size == 0:
        raise SystemExit("no run produced a measurable period")

    inside = np.abs(periods - TARGET_PERIOD) <= TOLERANCE_BAND * TARGET_PERIOD
    return {
        "runs": float(periods.size),
        "failed": float(len(results) - periods.size),
        "mean": float(periods.mean()),
        "sigma": float(periods.std(ddof=1)) if periods.size > 1 else 0.0,
        "min": float(periods.min()),
        "max": float(periods.max()),
        "p0.1": float(np.percentile(periods, 0.1)),
        "p1": float(np.percentile(periods, 1)),
        "p50": float(np.percentile(periods, 50)),
        "p99": float(np.percentile(periods, 99)),
        "p99.9": float(np.percentile(periods, 99.9)),
        "yield": float(inside.mean()),
    }


def print_summary(stats: dict[str, float], mode: str, engine: str) -> None:
    """Headline numbers."""
    runs = int(stats["runs"])
    print(f"\n{'=' * 72}")
    print(f"{mode} / {engine}: {runs:,} runs measured", end="")
    if stats["failed"]:
        print(f", {int(stats['failed']):,} failed", end="")
    print(f"\n{'=' * 72}")
    print(
        f"  period     mean {stats['mean']:.6f} s   sigma {stats['sigma']:.6f} s "
        f"({stats['sigma'] / stats['mean']:.3%})"
    )
    print(f"  range      {stats['min']:.6f} .. {stats['max']:.6f} s")
    print(
        f"  percentiles  0.1% {stats['p0.1']:.4f}   1% {stats['p1']:.4f}   "
        f"50% {stats['p50']:.4f}   99% {stats['p99']:.4f}   "
        f"99.9% {stats['p99.9']:.4f}"
    )
    print(
        f"  inside +/-{TOLERANCE_BAND:.0%} of {TARGET_PERIOD:g} s: "
        f"{stats['yield']:.3%}"
    )

    # The rule of three: zero failures in n samples still only bounds the
    # failure rate at about 3/n, so a clean sweep is not proof of 100% yield.
    if stats["yield"] >= 1.0:
        print(
            f"    every run passed, which bounds the true failure rate at "
            f"about {3 / runs:.2%} (95% confidence), not at zero."
        )


def print_table(results: Sequence[Result], limit: int, sort: bool) -> None:
    """The component values and what period they gave."""
    rows = list(results)
    if sort:
        rows.sort(key=lambda r: abs(r.error_percent), reverse=True)

    shown = rows[:limit] if limit and len(rows) > limit else rows
    offset = f"% from {TARGET_PERIOD:g}s"
    print(
        f"\n{'build':>18} "
        + " ".join(f"{p:>9}" for p in TIMING_PARTS)
        + f" {'period':>10} {'duty':>7} {offset:>10}"
    )
    print("-" * (19 + 10 * len(TIMING_PARTS) + 30))

    for row in shown:
        if not math.isfinite(row.period):
            print(
                f"{row.label:>18} "
                + " ".join("        -" for _ in TIMING_PARTS)
                + f" {'FAILED':>10}  {(row.error or '')[:28]}"
            )
            continue
        print(
            f"{row.label:>18} "
            + " ".join(f"{row.values[p]:>9.4g}" for p in TIMING_PARTS)
            + f" {row.period:>10.6f} {row.duty_cycle:>6.2%} {row.error_percent:>+9.3f}%"
        )

    if len(rows) > len(shown):
        print(
            f"  ... {len(rows) - len(shown):,} more rows (all of them are in the CSV)"
        )


def _value_of(part: str) -> Callable[["Result"], float]:
    """Sort key for one part's value, as a named function so mypy can type it."""
    return lambda row: row.values[part]


def print_sweep_summary(results: Sequence[Result]) -> None:
    """Per-part span: how much the period moves when only that part moves."""
    print(f"\n{'part':<6} {'period at band edges':>28} {'span':>10} {'S_x':>8}")
    print("-" * 56)

    for part in TIMING_PARTS:
        rows = [
            r
            for r in results
            if r.label.startswith(part + " ") and math.isfinite(r.period)
        ]
        if len(rows) < 2:
            continue
        rows.sort(key=_value_of(part))
        low, high = rows[0], rows[-1]
        span = (high.period - low.period) / TARGET_PERIOD
        # Normalised sensitivity: a 1% change in the part moves T by S_x %.
        change = high.values[part] / low.values[part] - 1.0
        sensitivity = (
            (high.period / low.period - 1.0) / change if change else float("nan")
        )
        print(
            f"{part:<6} {low.period:>12.6f} .. {high.period:<12.6f} "
            f"{span:>9.3%} {sensitivity:>+8.3f}"
        )
    print("\n  S_x is (dT/T)/(dx/x): a 1% change in that part moves T by S_x %.")
    print(
        f"  {TIMING_CAPACITOR} sits at 1.0 -- the period is exactly proportional to it."
    )
    if len(TIMING_RESISTOR) == 1:
        print(f"  {TIMING_RESISTOR[0]} also sits at 1.0, for the same reason.")
    else:
        print(
            "  " + " + ".join(TIMING_RESISTOR) + " share that same 1.0 between them, "
            "each in\n  proportion to its share of the total timing resistance -- a "
            "small resistor\n  in the string carries correspondingly less of the "
            "error."
        )
    print("  R3/R4 land near +/-0.6: they set the hysteresis window, and only")
    print("  their ratio matters, so moving one alone still moves the period.")
    print("  R1/R2 are near zero: mismatch there shifts the duty cycle, and only")
    print("  reaches the period second order.")


def write_csv(results: Sequence[Result], path: Path) -> None:
    """Every run, for the writeup."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "build",
                *(p.lower() for p in TIMING_PARTS),
                "period_s",
                "duty_cycle",
                "percent_from_target",
                "error",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.label,
                    *(f"{row.values[p]:.10g}" for p in TIMING_PARTS),
                    f"{row.period:.9f}" if math.isfinite(row.period) else "",
                    f"{row.duty_cycle:.6f}" if math.isfinite(row.duty_cycle) else "",
                    f"{row.error_percent:.6f}" if math.isfinite(row.period) else "",
                    row.error or "",
                ]
            )
    print(f"\nwrote {len(results):,} runs to {path}")


def describe_run(
    mode: str, engine: str, count: int, nominal: dict[str, float]
) -> tuple[str, str]:
    """
    Title and provenance line for a plot.

    A plot that ends up in a report has to say what produced it, or six weeks
    later nobody can tell which run it came from.
    """
    tool = "LTspice" if engine == "spice" else "closed-form model"
    headline = {
        "montecarlo": f"Monte Carlo tolerance simulation - {count:,} {tool} runs",
        "sweep": (
            f"One-part-at-a-time tolerance sweep - {count:,} {tool} runs "
            f"({LEVELS} levels x {len(TIMING_PARTS)} parts)"
        ),
        "corners": (
            f"Worst-case corner analysis - all {count} tolerance extremes " f"({tool})"
        ),
        "factorial": (
            f"Full factorial tolerance grid - {count:,} {tool} runs "
            f"({LEVELS} levels ^ {len(TIMING_PARTS)} parts)"
        ),
    }.get(mode, f"{mode} - {count:,} {tool} runs")

    shape = {
        "truncated": "truncated Gaussian",
        "gauss": "Gaussian",
        "uniform": "uniform",
    }.get(DISTRIBUTION, DISTRIBUTION)
    spread = (
        f"R +/-{RESISTOR_TOLERANCE:.0%}, C +/-{CAPACITOR_TOLERANCE:.0%} "
        f"read as {SIGMAS:g} sigma ({shape})"
        if mode != "corners"
        else f"R +/-{RESISTOR_TOLERANCE:.0%}, C +/-{CAPACITOR_TOLERANCE:.0%} "
        "pushed to both band edges"
    )
    series = [name for name in TIMING_RESISTOR if name in nominal]
    total = format_component(timing_resistance(nominal), False)
    spelled = (
        total
        if len(series) == 1
        else " + ".join(format_component(nominal[n], False) for n in series)
        + f" = {total}"
    )
    values = (
        f"R5={spelled}  "
        f"{TIMING_CAPACITOR}="
        f"{format_component(nominal[TIMING_CAPACITOR], True)}"
    )
    provenance = f"{spread}   |   nominal {values}   |   {NETLIST.name}" + (
        f"   |   seed {SEED}" if mode == "montecarlo" else ""
    )
    return headline, provenance


def format_component(value: float, is_capacitor: bool) -> str:
    """Component value the way a parts list would print it."""
    units = (("uF", 1e-6), ("nF", 1e-9)) if is_capacitor else (("M", 1e6), ("K", 1e3))
    for suffix, scale in units:
        if abs(value) >= scale:
            return f"{value / scale:.4g}{suffix}"
    return f"{value:.4g}"


def plot_results(  # pylint: disable=too-many-locals
    results: Sequence[Result],
    mode: str,
    path: Path,
    engine: str,
    nominal: dict[str, float],
) -> None:
    """Histogram for the sampling modes, per-part curves for a sweep."""
    # pylint: disable=import-outside-toplevel  # only needed when plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    periods = np.array([r.period for r in results if math.isfinite(r.period)])
    headline, provenance = describe_run(mode, engine, len(results), nominal)
    figure, axes = plt.subplots(figsize=(9.5, 6.0))

    if mode == "sweep":
        for part in TIMING_PARTS:
            rows = [
                r
                for r in results
                if r.label.startswith(part + " ") and math.isfinite(r.period)
            ]
            if not rows:
                continue
            centre = np.median([r.values[part] for r in rows])
            deviation = np.array([100 * (r.values[part] / centre - 1) for r in rows])
            order = np.argsort(deviation)
            axes.plot(
                deviation[order],
                np.array([r.period for r in rows])[order],
                marker=".",
                markersize=3,
                linewidth=1,
                label=part,
            )
        axes.axhline(TARGET_PERIOD, color="black", linewidth=0.8)
        axes.axhspan(
            TARGET_PERIOD * (1 - TOLERANCE_BAND),
            TARGET_PERIOD * (1 + TOLERANCE_BAND),
            color="tab:green",
            alpha=0.10,
            label=f"spec, +/-{TOLERANCE_BAND:.0%}",
        )
        axes.set(
            xlabel="deviation of that one part from its nominal value (%)",
            ylabel="oscillator period (s)",
        )
        axes.legend(fontsize=8, title="part varied", title_fontsize=8)
        note = (
            "each curve varies ONE part across its tolerance band\n"
            f"with the other {len(TIMING_PARTS) - 1} held at nominal"
        )
        axes.text(
            0.015,
            0.975,
            note,
            transform=axes.transAxes,
            fontsize=8,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
    else:
        axes.hist(periods, bins=60, color="tab:blue", alpha=0.85)
        for edge in (1 - TOLERANCE_BAND, 1 + TOLERANCE_BAND):
            axes.axvline(
                TARGET_PERIOD * edge, color="tab:red", linestyle="--", linewidth=1.3
            )
        axes.axvline(TARGET_PERIOD, color="black", linewidth=1.0)

        inside = np.abs(periods - TARGET_PERIOD) <= TOLERANCE_BAND * TARGET_PERIOD
        slow = periods > TARGET_PERIOD * (1 + TOLERANCE_BAND)
        fast = periods < TARGET_PERIOD * (1 - TOLERANCE_BAND)
        axes.set(xlabel="oscillator period (s)", ylabel="number of builds")
        axes.text(
            0.015,
            0.975,
            "\n".join(
                (
                    f"mean   {periods.mean():.4f} s  "
                    f"({periods.mean() / TARGET_PERIOD - 1:+.2%})",
                    f"sigma  {periods.std(ddof=1):.4f} s  "
                    f"({periods.std(ddof=1) / periods.mean():.2%})",
                    f"inside  {inside.sum():,} / {periods.size:,}  "
                    f"({inside.mean():.2%})",
                    f"too slow  {slow.sum():,}   too fast  {fast.sum():,}",
                )
            ),
            transform=axes.transAxes,
            fontsize=8.5,
            va="top",
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        axes.annotate(
            f"spec limits, {TARGET_PERIOD:g} s +/-{TOLERANCE_BAND:.0%}",
            xy=(TARGET_PERIOD * (1 + TOLERANCE_BAND), axes.get_ylim()[1] * 0.92),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="tab:red",
        )

    axes.grid(alpha=0.3)
    axes.set_title(headline, fontsize=12, pad=16)
    # Provenance under the title: what was varied, from which netlist, what seed.
    axes.text(
        0.5,
        1.012,
        provenance,
        transform=axes.transAxes,
        fontsize=8,
        color="0.35",
        ha="center",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    print(f"plot written to {path}")


# --------------------------------------------------------------------------- #
# The hook get_good_numbers.py calls
# --------------------------------------------------------------------------- #


def run_monte_carlo(
    values: dict[str, float],
    tolerances: dict[str, float],
    count: int = 20000,
    seed: int = 0,
    sigmas: float = 3.0,
    distribution: str = "truncated",
) -> np.ndarray:
    """
    Sampled periods for one design. Analytic, because get_good_numbers.py calls
    this once per candidate and there are thousands of candidates.
    """
    rng = np.random.default_rng(seed)
    samples = draw(values, tolerances, count, rng, sigmas, distribution)
    return np.array([period for period, _ in evaluate_analytic(samples)])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def read_nominal(
    netlist_path: Path, resistor_tolerance: float, capacitor_tolerance: float
) -> tuple[dict[str, float], dict[str, float]]:
    """Nominal values from the netlist, tolerances from the command line."""
    # pylint: disable=import-outside-toplevel  # keeps the module importable
    import ltspice

    netlist = ltspice.Netlist.from_file(netlist_path)
    missing = [p for p in TIMING_PARTS if p not in netlist.components]
    if missing:
        raise SystemExit(f"{netlist_path} has no {', '.join(missing)}")

    nominal = {part: netlist.components[part] for part in TIMING_PARTS}
    tolerances = {
        part: capacitor_tolerance if part.startswith("C") else resistor_tolerance
        for part in TIMING_PARTS
    }
    return nominal, tolerances


def main() -> int:
    """Main function. Everything it reads is in the SETTINGS block up top."""
    if not NETLIST.exists():
        raise SystemExit(f"{NETLIST} does not exist (set NETLIST in the settings)")

    nominal, tolerances = read_nominal(NETLIST, RESISTOR_TOLERANCE, CAPACITOR_TOLERANCE)
    samples, labels = build_plan(
        MODE, nominal, tolerances, LEVELS, RUNS, SEED, SIGMAS, DISTRIBUTION, MAX_RUNS
    )

    print("nominal: " + "  ".join(f"{p}={nominal[p]:.6g}" for p in TIMING_PARTS))
    print(
        f"tolerance: resistors +/-{RESISTOR_TOLERANCE:.0%}, capacitor "
        f"+/-{CAPACITOR_TOLERANCE:.0%}, read as {SIGMAS:g} sigma ({DISTRIBUTION})"
    )
    print(f"{MODE}: {len(samples):,} runs on the {ENGINE} engine")

    started = time.time()
    measured: list[tuple[float, float, Optional[str]]]
    if ENGINE == "analytic":
        measured = [(p, d, None) for p, d in evaluate_analytic(samples)]
    elif ENGINE == "spice":
        if len(samples) > MAX_RUNS:
            raise SystemExit(
                f"{len(samples):,} SPICE runs is over MAX_RUNS ({MAX_RUNS:,}). "
                'Raise MAX_RUNS, or set ENGINE = "analytic".'
            )
        measured = evaluate_spice(samples, NETLIST, WORKROOT, tran=TRAN, chunk=CHUNK)
    else:
        raise SystemExit(f'ENGINE must be "spice" or "analytic", not {ENGINE!r}')
    elapsed = time.time() - started

    results = [
        Result(
            values=sample, period=period, duty_cycle=duty, label=label, error=failure
        )
        for sample, label, (period, duty, failure) in zip(samples, labels, measured)
    ]
    print(
        f"\n{len(samples):,} runs in {elapsed:.1f} s "
        f"({len(samples) / max(elapsed, 1e-9):.1f} runs/s)"
    )

    print_table(results, SHOW_ROWS, sort=MODE != "sweep")
    if MODE == "sweep":
        print_sweep_summary(results)
    print_summary(summarise(results), MODE, ENGINE)

    write_csv(results, CSV_PATH)
    if PLOT_PATH is not None:
        plot_results(results, MODE, PLOT_PATH, ENGINE, nominal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
