"""
Measure oscillator timing from a waveform.

Works on an LTspice .raw, a tab-separated LTspice text export, or arrays passed
straight in. Nothing here is specific to MP1: give it any squarish signal and it
returns the period, duty cycle and rails.

Method, and why it is not peak-finding: edges are located by interpolating the
crossings of the waveform's own mid-level, not by looking for peaks. A peak
finder on a flat-topped square wave keys on whatever numerical wobble sits on
the top, so its answer depends on the timestep. Interpolated mid-crossings do
not. Crossings are qualified by a Schmitt band, so a glitch that grazes the mid
level without completing an edge is not counted as one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

# =========================================================================== #
# SETTINGS -- edit these, then run:  python analyze_vout.py
# =========================================================================== #

# The waveform to measure: an LTspice .raw, or a tab-separated text export.
WAVEFORM: Path = Path("../mp1_sim.raw")

# Which trace in it to measure.
TRACE: str = "V(vout)"

# The period to compare against, in seconds.
TARGET: float = 1.0

# Cycles to discard at the start. The supplies ramp during `startup`, so the
# first cycles are not representative. In cycles rather than seconds, so the
# same number works on a 10 ms oscillator and a 10 s one.
SKIP_CYCLES: float = 2.0

# Save an annotated plot here. Set to None to skip plotting.
PLOT_PATH: Optional[Path] = None

# =========================================================================== #

# Defaults for callers that import this module rather than running it.
DEFAULT_TRACE: str = TRACE
DEFAULT_TARGET: float = TARGET
DEFAULT_SKIP_CYCLES: float = SKIP_CYCLES

# Fraction of the swing above and below mid that an edge must reach to count.
HYSTERESIS: float = 0.25


class Timing(NamedTuple):
    """What one waveform says about the oscillator."""

    period: float
    frequency: float
    duty_cycle: float
    high_time: float
    low_time: float
    v_low: float
    v_high: float
    v_mid: float
    cycles: int
    period_spread: float  # std dev across individual cycles
    first_edge: float
    periods: np.ndarray  # every measured cycle, for histograms

    def error(self, target: float = DEFAULT_TARGET) -> float:
        """Fractional period error against a target."""
        return self.period / target - 1.0


class NotOscillating(ValueError):
    """The waveform has too few complete cycles to measure."""


def rails(signal: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate the low rail, high rail and mid level.

    Medians of the two halves rather than min/max, so overshoot on one edge
    does not set the rail for the whole record.
    """
    guess = 0.5 * (float(signal.min()) + float(signal.max()))
    low_side, high_side = signal[signal < guess], signal[signal > guess]
    if low_side.size == 0 or high_side.size == 0:
        raise NotOscillating("signal never crosses its own midpoint")

    v_low, v_high = float(np.median(low_side)), float(np.median(high_side))
    return v_low, v_high, 0.5 * (v_low + v_high)


def crossings(  # pylint: disable=too-many-locals
    time: np.ndarray, signal: np.ndarray, level: float, hysteresis: float = HYSTERESIS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolated times at which `signal` crosses `level`, and their directions.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Crossing times, and +1 for rising or -1 for falling.
    """
    v_low, v_high, _ = rails(signal)
    swing = v_high - v_low
    if swing <= 0:
        raise NotOscillating("signal has no swing")

    # Schmitt state: +1 once the signal clears the upper band, -1 once it drops
    # below the lower band, and whatever it was before in between. Forward-fill
    # the band hits so every sample carries a definite state.
    upper, lower = level + hysteresis * swing, level - hysteresis * swing
    event = np.where(signal > upper, 1, np.where(signal < lower, -1, 0)).astype(np.int8)
    hits = np.flatnonzero(event)
    if hits.size == 0:
        raise NotOscillating("signal never leaves its hysteresis band")
    state = event[
        hits[
            np.clip(np.searchsorted(hits, np.arange(signal.size), "right") - 1, 0, None)
        ]
    ]
    state[: hits[0]] = event[hits[0]]

    flips = np.flatnonzero(np.diff(state) != 0)
    if flips.size == 0:
        raise NotOscillating("signal never completes an edge")

    # Every mid-level crossing, then keep the last one at or before each flip:
    # that is the crossing belonging to the edge the flip confirmed.
    above = signal > level
    mids = np.flatnonzero(np.diff(above.astype(np.int8)) != 0)
    if mids.size == 0:
        raise NotOscillating("signal never crosses the mid level")

    slot = np.searchsorted(mids, flips, side="right") - 1
    keep = mids[slot[slot >= 0]]
    keep, order = np.unique(keep, return_index=True)
    direction = state[flips[slot >= 0]][order]

    # Linear interpolation between the straddling samples. Without this the
    # resolution is the timestep and the measured period is quantised garbage.
    before, after = signal[keep], signal[keep + 1]
    span = np.where(after == before, 1.0, after - before)
    fraction = (level - before) / span
    times = time[keep] + fraction * (time[keep + 1] - time[keep])

    return times, direction.astype(np.int8)


def measure(  # pylint: disable=too-many-locals
    time: np.ndarray,
    signal: np.ndarray,
    skip_cycles: float = DEFAULT_SKIP_CYCLES,
    hysteresis: float = HYSTERESIS,
) -> Timing:
    """
    Measure period, duty cycle and rails.

    `skip_cycles` discards the start of the record, in units of the period it
    measures first -- expressed that way rather than in seconds so the same
    call works on a 10 ms oscillator and a 10 s one.
    """
    time, signal = np.asarray(time, float), np.asarray(signal, float)
    if time.size != signal.size:
        raise ValueError("time and signal have different lengths")

    v_low, v_high, v_mid = rails(signal)
    times, direction = crossings(time, signal, v_mid, hysteresis)

    rising = times[direction > 0]
    if rising.size >= 2 and skip_cycles > 0:
        # Supplies ramp during `startup`, so the first cycles are not
        # representative. Estimate the period, then re-cut on it.
        rough = float(np.median(np.diff(rising)))
        cut = time[0] + skip_cycles * rough
        keep = times >= cut
        if np.count_nonzero(keep & (direction > 0)) >= 2:
            times, direction = times[keep], direction[keep]

    rising, falling = times[direction > 0], times[direction < 0]
    if rising.size < 2:
        raise NotOscillating(
            f"only {rising.size} rising edge(s) after skipping "
            f"{skip_cycles:g} cycles; simulate longer"
        )

    periods = np.diff(rising)
    # Endpoint-to-endpoint over all whole cycles: interpolation error at the
    # two ends is divided by the cycle count instead of averaged in.
    period = float((rising[-1] - rising[0]) / (rising.size - 1))

    # Pair each rising edge with the next falling edge to get the high time.
    slot = np.searchsorted(falling, rising[:-1], side="right")
    valid = slot < falling.size
    high_time = (
        float(np.mean(falling[slot[valid]] - rising[:-1][valid]))
        if np.any(valid)
        else float("nan")
    )

    return Timing(
        period=period,
        frequency=1.0 / period,
        duty_cycle=high_time / period,
        high_time=high_time,
        low_time=period - high_time,
        v_low=v_low,
        v_high=v_high,
        v_mid=v_mid,
        cycles=int(periods.size),
        period_spread=float(np.std(periods)),
        first_edge=float(rising[0]),
        periods=periods,
    )


def load(path: Path, trace: str = DEFAULT_TRACE) -> tuple[np.ndarray, np.ndarray]:
    """Read (time, signal) from an LTspice .raw or a tab-separated export."""
    if path.suffix.lower() == ".raw":
        # pylint: disable=import-outside-toplevel  # text path needs no wine
        import ltspice

        wave = ltspice.read_raw(path)
        return wave.time, wave.trace(trace)

    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    names = [name.strip() for name in header.split("\t")]
    table = np.loadtxt(path, skiprows=1)

    wanted = trace.lower()
    for index, name in enumerate(names):
        if name.lower() in (wanted, f"v({wanted})"):
            return table[:, 0], table[:, index]
    raise KeyError(f"no column {trace!r} in {path.name}; it has {names}")


def report(timing: Timing, target: float = DEFAULT_TARGET) -> str:
    """One-screen summary."""
    return "\n".join(
        (
            f"  period       {timing.period:.6f} s   ({timing.error(target):+.3%} "
            f"vs {target:g} s)",
            f"  frequency    {timing.frequency:.6f} Hz",
            f"  duty cycle   {timing.duty_cycle:.4%}   "
            f"(high {timing.high_time:.6f} s, low {timing.low_time:.6f} s)",
            f"  rails        {timing.v_low:.4f} V .. {timing.v_high:.4f} V   "
            f"(mid {timing.v_mid:.4f} V)",
            f"  measured on  {timing.cycles} cycles from t = {timing.first_edge:.4f} s",
            f"  cycle spread {timing.period_spread * 1e6:.1f} us "
            f"({timing.period_spread / timing.period:.4%} of the period)",
        )
    )


def plot(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    time: np.ndarray,
    signal: np.ndarray,
    timing: Timing,
    target: float,
    path: Path,
    source: str = "",
    trace: str = "",
) -> None:
    """Waveform with the detected edges marked, saved next to the input."""
    # pylint: disable=import-outside-toplevel  # only needed when plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, direction = crossings(time, signal, timing.v_mid)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=(2, 1))

    axes[0].plot(time, signal, linewidth=0.9, label="Vout")
    axes[0].axhline(timing.v_mid, color="grey", linestyle=":", linewidth=0.8)
    axes[0].plot(
        times[direction > 0],
        np.full(np.count_nonzero(direction > 0), timing.v_mid),
        "o",
        color="tab:green",
        markersize=4,
        label="rising",
    )
    axes[0].plot(
        times[direction < 0],
        np.full(np.count_nonzero(direction < 0), timing.v_mid),
        "o",
        color="tab:red",
        markersize=4,
        label="falling",
    )
    # Shade the startup cycles, so it is clear the marked edges before this
    # point were found but not counted.
    if timing.first_edge > time[0]:
        axes[0].axvspan(
            time[0],
            timing.first_edge,
            color="0.5",
            alpha=0.15,
            label="discarded (startup)",
        )
    axes[0].set(xlabel="time (s)", ylabel="volts")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].text(
        0.012,
        0.96,
        "\n".join(
            (
                f"period  {timing.period:.6f} s  "
                f"({timing.error(target):+.3%} vs {target:g} s)",
                f"duty    {timing.duty_cycle:.2%}",
                f"rails   {timing.v_low:.3f} .. {timing.v_high:.3f} V",
            )
        ),
        transform=axes[0].transAxes,
        fontsize=8.5,
        va="top",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    axes[1].plot(timing.periods, "o-", markersize=3, linewidth=0.8)
    axes[1].axhline(target, color="tab:red", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="cycle number (startup cycles already discarded)",
        ylabel="period (s)",
    )
    axes[1].grid(alpha=0.3)

    # Say what this is and where it came from, so the figure stands alone.
    figure.suptitle(
        "Oscillator timing measured from an LTspice transient", fontsize=12, y=0.995
    )
    axes[0].set_title(
        f"{source}   |   {trace}   |   "
        f"{timing.cycles} cycles measured from t = {timing.first_edge:.3f} s   |   "
        "edges are interpolated mid-level crossings",
        fontsize=8,
        color="0.35",
        pad=8,
    )

    figure.tight_layout()
    figure.savefig(path, dpi=130)
    print(f"  plot written to {path}")


def main() -> int:
    """Main function. Everything it reads is in the SETTINGS block up top."""
    if not WAVEFORM.exists():
        print(
            f"{WAVEFORM} does not exist (set WAVEFORM in the settings)",
            file=sys.stderr,
        )
        return 1

    time, signal = load(WAVEFORM, TRACE)
    try:
        timing = measure(time, signal, skip_cycles=SKIP_CYCLES)
    except NotOscillating as error:
        print(f"{WAVEFORM.name}: {error}", file=sys.stderr)
        return 1

    print(f"\n{WAVEFORM.name}  [{TRACE}]")
    print(report(timing, TARGET))
    if PLOT_PATH is not None:
        plot(time, signal, timing, TARGET, PLOT_PATH, WAVEFORM.name, TRACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
