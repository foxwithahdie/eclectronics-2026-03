"""Template a netlist, run LTspice under wine, read the waveform back."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional, Sequence

import numpy as np

LTSPICE_ENV: str = "LTSPICE_EXE"
LTSPICE_DEFAULTS: tuple[str, ...] = (
    "~/.wine/drive_c/Program Files/ADI/LTspice/LTspice.exe",
    "~/.wine/drive_c/Program Files/LTC/LTspiceXVII/XVIIx64.exe",
)

# MEG before M: in SPICE a bare M means milli, not mega.
SPICE_SUFFIXES: tuple[tuple[str, float], ...] = (
    ("MEG", 1e6),
    ("T", 1e12),
    ("G", 1e9),
    ("K", 1e3),
    ("M", 1e-3),
    ("U", 1e-6),
    ("N", 1e-9),
    ("P", 1e-12),
    ("F", 1e-15),
)

# Both micro signs, folded to "u" before upper(). U+00B5 uppercases to Greek
# capital Mu, so unfolded "1µF" parses as 1 farad instead of 1 microfarad.
MICRO: str = "µμ"

# designator, two nodes, value, then anything else ("tol=1 pwr=1/10").
INSTANCE: re.Pattern[str] = re.compile(
    r"^(?P<name>[RCL]\w*)(?P<gap>\s+\S+\s+\S+\s+)(?P<value>\S+)(?P<rest>.*)$",
    re.IGNORECASE,
)
TRAN: re.Pattern[str] = re.compile(r"^\.tran\s+", re.IGNORECASE)
STEP_RAN: re.Pattern[str] = re.compile(r"^\.step\s+idx=(\d+)", re.IGNORECASE)

# LTspice chokes on very long directive lines, so tables are wrapped with SPICE
# "+" continuations at this many entries per line.
TABLE_WRAP: int = 12


class RunFailed(RuntimeError):
    """LTspice did not produce a usable .raw."""


class Waveform(NamedTuple):
    """One run's output."""

    time: np.ndarray
    traces: dict[str, np.ndarray]

    def trace(self, name: str) -> np.ndarray:
        """Look up a trace, ignoring case and V()/I() spelling."""
        wanted = name.lower()
        for key, values in self.traces.items():
            if key.lower() in (wanted, f"v({wanted})", f"i({wanted})"):
                return values
        raise KeyError(f"no trace {name!r}; file has {sorted(self.traces)}")


def parse_spice_value(text: str) -> float:
    """Read a SPICE value token ("475K", "1uF", "1e-6") as a float."""
    match = re.match(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)(.*)$", text.strip())
    if match is None:
        raise ValueError(f"cannot read {text!r} as a SPICE value")

    suffix = match.group(2).strip()
    for sign in MICRO:
        suffix = suffix.replace(sign, "u")
    number, suffix = float(match.group(1)), suffix.upper()
    if "E" in match.group(1).upper():  # exponent already set the magnitude
        return number
    for letters, scale in SPICE_SUFFIXES:
        if suffix.startswith(letters):
            return number * scale
    return number


def format_spice_value(value: float) -> str:
    """Write a float back out. Scientific notation can't be misread as milli."""
    return f"{value:.10e}"


class Netlist:
    """A .net file whose R/C/L values can be substituted."""

    def __init__(self, lines: Sequence[str]) -> None:
        self.lines: list[str] = list(lines)
        self.components: dict[str, float] = {}
        self.rows: dict[str, int] = {}

        for row, line in enumerate(self.lines):
            match = INSTANCE.match(line)
            if match is None:
                continue
            try:
                value = parse_spice_value(match.group("value"))
            except ValueError:
                continue  # an expression like {R5*2}, not a literal
            self.components[match.group("name").upper()] = value
            self.rows[match.group("name").upper()] = row

    @classmethod
    def from_file(cls, path: Path) -> "Netlist":
        """Read a netlist off disk."""
        return cls(path.read_text(encoding="utf-8").splitlines())

    def render(
        self,
        values: Optional[dict[str, float]] = None,
        tran: Optional[str] = None,
        save: Optional[Iterable[str]] = None,
        uncompressed: bool = True,
    ) -> str:
        """
        Netlist text for one run.

        `save` narrows the .raw from ~144 bytes per point to 12, which matters
        across thousands of runs. `uncompressed` disables LTspice's waveform
        compression, which otherwise mangles the edge times being measured.
        """
        lines = list(self.lines)

        for name, value in (values or {}).items():
            row = self.rows.get(name.upper())
            if row is None:
                raise KeyError(
                    f"{name} is not a passive here; netlist has "
                    f"{sorted(self.components)}"
                )
            match = INSTANCE.match(lines[row])
            assert match is not None  # matched once already in __init__
            lines[row] = (
                match.group("name")
                + match.group("gap")
                + format_spice_value(value)
                + match.group("rest")
            )

        if tran is not None:
            lines = [f".tran {tran}" if TRAN.match(line) else line for line in lines]

        extra: list[str] = []
        if uncompressed:
            extra.append(".options plotwinsize=0")
        if save is not None:
            extra.append(".save " + " ".join(save))

        return "\n".join(_before_end(lines, extra)) + "\n"


def _before_end(lines: Sequence[str], extra: Sequence[str]) -> list[str]:
    """Splice directives in ahead of .end."""
    if not extra:
        return list(lines)
    for row in range(len(lines) - 1, -1, -1):
        if lines[row].strip().lower() == ".end":
            return [*lines[:row], *extra, *lines[row:]]
    return [*lines, *extra]


def read_raw(  # pylint: disable=too-many-locals
    path: Path, only: Optional[Iterable[str]] = None
) -> Waveform:
    """
    Read an LTspice binary .raw.

    Header is UTF-16LE; a transient stores the axis as float64 and every other
    trace as float32. LTspice flags some timepoints by setting the axis sign
    bit, so the magnitude is the time and the sign is not data.
    """
    blob = path.read_bytes()
    marker = "Binary:\n".encode("utf-16-le")
    split = blob.find(marker)
    if split < 0:
        raise RunFailed(f"{path.name} has no binary section; run did not finish")

    header = blob[:split].decode("utf-16-le", errors="replace")
    body = blob[split + len(marker) :]

    fields: dict[str, str] = {}
    names: list[str] = []
    in_variables = False
    for line in header.splitlines():
        if line.startswith("Variables:"):
            in_variables = True
        elif in_variables:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0] == "":
                names.append(parts[2])
        else:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    if "real" not in fields.get("Flags", ""):
        raise RunFailed(f"{path.name} is not a real-valued analysis")

    count, width = int(fields["No. Points"]), int(fields["No. Variables"])
    if width != len(names):
        raise RunFailed(f"{path.name} declares {width} variables, lists {len(names)}")

    stride = 8 + 4 * (width - 1)
    if len(body) < stride * count:
        raise RunFailed(f"{path.name} is truncated")

    flat = np.frombuffer(body, dtype=np.uint8, count=stride * count)
    table = flat.reshape(count, stride)
    axis = np.abs(table[:, :8].copy().view(np.float64).reshape(count))

    wanted = {item.lower() for item in only} if only is not None else None
    traces: dict[str, np.ndarray] = {}
    for index, name in enumerate(names[1:]):
        if wanted is not None and name.lower() not in wanted:
            continue
        start = 8 + 4 * index
        traces[name] = (
            table[:, start : start + 4].copy().view(np.float32).reshape(count)
        ).astype(np.float64)

    return Waveform(time=axis, traces=traces)


def find_ltspice() -> Path:
    """Locate LTspice.exe, honouring $LTSPICE_EXE."""
    override = os.environ.get(LTSPICE_ENV)
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return path
        raise RunFailed(f"{LTSPICE_ENV} points at {path}, which does not exist")

    for candidate in LTSPICE_DEFAULTS:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    raise RunFailed(f"cannot find LTspice.exe; set {LTSPICE_ENV} to its path")


def to_windows_path(path: Path) -> str:
    """Translate for wine. A POSIX path makes LTspice do nothing and exit 0."""
    try:
        done = subprocess.run(
            ["winepath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunFailed(f"winepath could not translate {path}: {error}") from error
    return done.stdout.strip()


def run(
    netlist: str,
    workdir: Path,
    name: str = "run",
    only: Optional[Iterable[str]] = None,
    timeout: float = 600.0,
) -> Waveform:
    """Write one netlist out, simulate it, read the waveform back."""
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / f"{name}.net"
    source.write_text(netlist, encoding="utf-8")

    try:
        done = subprocess.run(
            ["wine", str(find_ltspice()), "-b", "-Run", to_windows_path(source)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
    except subprocess.TimeoutExpired as error:
        raise RunFailed(f"{name} did not finish within {timeout} s") from error

    # A non-zero code means wine killed it rather than LTspice declining to
    # run -- worth saying so, because it is a retry-worthy wedge and not a
    # problem with the netlist.
    if done.returncode:
        raise RunFailed(
            f"{name}: wine exited {done.returncode} "
            f"(transient wedge, not a netlist error)"
        )

    # Batch mode returns 0 whether or not it simulated anything, so a readable
    # .raw is the only reliable signal.
    output = workdir / f"{name}.raw"
    if not output.exists():
        raise RunFailed(f"{name} produced no .raw\n{_log_tail(workdir/f'{name}.log')}")
    try:
        return read_raw(output, only=only)
    except RunFailed as error:
        raise RunFailed(f"{error}\n{_log_tail(workdir / f'{name}.log')}") from error


def _log_tail(path: Path, lines: int = 12) -> str:
    """Tail of an LTspice log, for putting inside an exception."""
    if not path.exists():
        return "  (no log file)"
    tail = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(f"  {line}" for line in tail[-lines:])


def run_batch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    netlists: Sequence[str],
    workroot: Optional[Path] = None,
    workers: int = 0,
    only: Optional[Iterable[str]] = None,
    timeout: float = 600.0,
    keep: bool = False,
) -> Iterator[tuple[int, Optional[Waveform], Optional[str]]]:
    """
    Run many netlists, several at a time, yielding (index, waveform, error).

    Each run gets its own directory because LTspice names outputs after the
    netlist. A failed run yields its error instead of raising, so one
    non-converging corner doesn't lose the rest of the sweep.
    """
    if workers <= 0:
        workers = max(1, min(16, os.cpu_count() or 4))

    scratch: Optional[str] = None
    if workroot is None:
        scratch = tempfile.mkdtemp(prefix="ltspice-")
        workroot = Path(scratch)
    workroot.mkdir(parents=True, exist_ok=True)

    def one(index: int) -> tuple[int, Optional[Waveform], Optional[str]]:
        try:
            return (
                index,
                run(
                    netlists[index],
                    workroot / f"{index:06d}",
                    only=only,
                    timeout=timeout,
                ),
                None,
            )
        except RunFailed as error:
            return index, None, str(error)

    try:
        # Threads, not processes: every worker is blocked on an external
        # process, so there is no GIL to contend for.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(one, range(len(netlists)))
    finally:
        if scratch is not None and not keep:
            shutil.rmtree(scratch, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Stepped sweeps
# --------------------------------------------------------------------------- #


def render_sweep(
    netlist: Netlist,
    samples: Sequence[dict[str, float]],
    tran: Optional[str] = None,
    save: Optional[Iterable[str]] = None,
    uncompressed: bool = True,
) -> str:
    """
    One netlist that runs every sample, via .step over a table of values.

    Worth the complexity: a wine process start costs ~4 s and the solve itself
    costs ~0.1 s, so running samples one invocation each is forty times slower
    than stepping them inside a single invocation.
    """
    if not samples:
        raise ValueError("no samples to sweep")

    varying = sorted({name.upper() for sample in samples for name in sample})
    for name in varying:
        if name not in netlist.components:
            raise KeyError(f"{name} is not a passive in this netlist")

    lines = list(netlist.lines)
    for name in varying:
        row = netlist.rows[name]
        match = INSTANCE.match(lines[row])
        assert match is not None
        lines[row] = (
            match.group("name")
            + match.group("gap")
            + "{%s_v}" % name.lower()
            + match.group("rest")
        )

    if tran is not None:
        lines = [f".tran {tran}" if TRAN.match(line) else line for line in lines]

    extra = [".param idx=1", f".step param idx 1 {len(samples)} 1"]
    for name in varying:
        # Fall back to the netlist value so a sample may omit a component.
        default = netlist.components[name]
        entries = [
            f"{i + 1},{sample.get(name, sample.get(name.lower(), default)):.10e}"
            for i, sample in enumerate(samples)
        ]
        extra.extend(_wrap_table(f"{name.lower()}_v", entries))

    if uncompressed:
        extra.append(".options plotwinsize=0")
    if save is not None:
        extra.append(".save " + " ".join(save))

    return "\n".join(_before_end(lines, extra)) + "\n"


def _wrap_table(param: str, entries: Sequence[str]) -> list[str]:
    """Emit ".param p=table(idx,...)" wrapped over SPICE continuation lines."""
    head = f".param {param}=table(idx"
    out: list[str] = []
    for start in range(0, len(entries), TABLE_WRAP):
        chunk = ",".join(entries[start : start + TABLE_WRAP])
        out.append(f"{head},{chunk}" if start == 0 else f"+,{chunk}")
    out[-1] += ")"
    return out


def split_steps(wave: Waveform) -> list[Waveform]:
    """Split a stepped .raw into one Waveform per step, on time resets."""
    breaks = np.flatnonzero(np.diff(wave.time) < 0) + 1
    bounds = np.concatenate(([0], breaks, [len(wave.time)]))
    return [
        Waveform(
            time=wave.time[a:b],
            traces={name: values[a:b] for name, values in wave.traces.items()},
        )
        for a, b in zip(bounds[:-1], bounds[1:])
    ]


def reset_wine(settle: float = 5.0, warm_up: Optional[str] = None) -> None:
    """
    Restart the wineserver, wait for it, and run `warm_up` once to prove it.

    LTspice under wine occasionally wedges: it starts, logs as far as
    "method = trap", then exits 0 having written a zero-byte .raw. Restarting
    the wineserver clears it. The wait afterwards is not optional -- a run
    launched immediately after the kill fails the same way, and killing again
    to fix that just repeats the race.
    """
    quiet = {**os.environ, "WINEDEBUG": "-all"}
    subprocess.run(["wineserver", "-k"], capture_output=True, check=False, env=quiet)
    subprocess.run(["wineserver", "-w"], capture_output=True, check=False, env=quiet)
    time.sleep(settle)

    if warm_up is None:
        return

    # The first LTspice launch after a restart fails the same way the wedge
    # does, so burn one throwaway run here. Without this the retry that follows
    # a reset is always the doomed one, and no number of retries ever recovers.
    scratch = Path(tempfile.mkdtemp(prefix="ltspice-warmup-"))
    try:
        for _ in range(4):
            try:
                run(warm_up, scratch, name="warmup", timeout=120.0)
                return
            except RunFailed:
                continue
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def steps_completed(path: Path) -> int:
    """How many steps LTspice logged, so a short .raw can be spotted."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if STEP_RAN.match(line.strip()))


def run_sweep(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    netlist: Netlist,
    samples: Sequence[dict[str, float]],
    workroot: Path,
    tran: Optional[str] = None,
    save: Optional[Iterable[str]] = None,
    chunk: int = 200,
    attempts: int = 6,
    backoff: float = 20.0,
    timeout: float = 1800.0,
    keep: bool = False,
    progress: bool = False,
) -> Iterator[tuple[int, Optional[Waveform], Optional[str]]]:
    """
    Simulate every sample, yielding (index, waveform, error) in order.

    Samples are stepped in chunks inside one LTspice invocation each. A chunk
    that comes back short or empty is retried after a wineserver reset; if it
    still fails, its samples are yielded as errors rather than losing the sweep.
    """
    workroot.mkdir(parents=True, exist_ok=True)
    only = list(save) if save is not None else None

    for start in range(0, len(samples), chunk):
        batch = list(samples[start : start + chunk])
        workdir = workroot / f"chunk{start:06d}"
        waves: list[Waveform] = []
        error: Optional[str] = None

        for attempt in range(attempts):
            # Reset before every retry. This is only safe because reset_wine
            # warms itself up: the first launch after a wineserver kill fails
            # the same way the wedge does, so without the warm-up the retry
            # would always be the doomed one and nothing would ever recover.
            #
            # The backoff matters as much as the reset. The wedge does not
            # clear the instant the wineserver is restarted -- it takes tens of
            # seconds to come good on its own, so retrying immediately just
            # burns attempts. Waiting longer each time is what recovers.
            if attempt:
                time.sleep(backoff * attempt)
                reset_wine(warm_up=render_sweep(netlist, batch[:1], tran=tran))
            try:
                wave = run(
                    render_sweep(netlist, batch, tran=tran, save=save),
                    workdir,
                    name="sweep",
                    only=only,
                    timeout=timeout,
                )
                waves = split_steps(wave)
                if len(waves) == len(batch):
                    error = None
                    break
                error = (
                    f"got {len(waves)} steps for {len(batch)} samples "
                    f"({steps_completed(workdir / 'sweep.log')} logged)"
                )
            except RunFailed as failure:
                error = str(failure)

        if progress:
            done = min(start + len(batch), len(samples))
            print(
                f"  {done}/{len(samples)} simulated"
                + ("" if error is None else f"  [{error.splitlines()[0]}]"),
                flush=True,
            )

        for offset in range(len(batch)):
            if error is None:
                yield start + offset, waves[offset], None
            else:
                yield start + offset, None, error

        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
