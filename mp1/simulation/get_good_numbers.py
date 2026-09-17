"""
Picks real, buyable component values for the MP1 relaxation oscillator.

R1 = R2 and R3 = R4 are fixed by the design, so the period collapses to

    T = 2 * R5 * C1 * ln(3)

and the only free quantity left is the RC product. That makes this a one
dimensional search: walk every R5 and C1 the parts list can supply, and find
whichever product lands closest to the target.

The catch is that the parts list is a sparse ladder rather than a full
E-series. Every stocked capacitor is an exact power of ten, so the capacitor
only chooses the decade -- the achievable mantissas are exactly the resistor
mantissas, and around the value we want the ladder jumps 402K -> 475K -> 499K.
The value we actually want falls in that hole.

Each candidate is scored three ways, and all three go in the table:

    nominal     where the period sits with every part at its marked value.
    worst case  every part pushed to the end of its band, all 2**6 corners.
                A guarantee rather than a sample.
    yield       the fraction of builds expected inside the spec, estimated in
                closed form from the sensitivities. This is the number that
                tolerance_simulation.py should reproduce with a real Monte
                Carlo; if the two disagree, one of them is wrong.

The ranked table is written to candidates.csv so tolerance_simulation.py can
read the values back instead of having them typed in twice.

The hand derivation this file is built on is kept verbatim in ORIGINAL_NOTES
below, and the same work written out properly is in README sections 3.1 to 3.8.
"""

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from itertools import combinations
from math import erf, log, sqrt
from pathlib import Path
from types import ModuleType
from typing import Optional, NamedTuple

# tolerance_simulation.py is where the Monte Carlo lives. It is optional here,
# since everything below is closed form and runs with or without it. If it
# grows a run_monte_carlo(), the detail block picks up a measured yield to set
# against the estimate.
MONTE_CARLO: Optional[ModuleType]
try:
    import tolerance_simulation

    MONTE_CARLO = tolerance_simulation
except ImportError:
    MONTE_CARLO = None


ORIGINAL_NOTES: str = """
The hand derivation, kept as first written, for the report.

Two things were wrong on the page and are right in the code below. Both are
operator precedence, not physics:

  * `r4 / r3 + r4` parses as `(r4/r3) + r4`, which is 10001.0, not 0.5. It
    needs to be `r4 / (r3 + r4)`. Same for `r3 / r3 + r4`.
  * `ln(a / b + c)` needs the denominator bracketed as `ln(a / (b + c))`.

The structure underneath was already correct: the leading minus times the two
logarithms gives exactly the two half-periods, and it agrees with the closed
form to machine precision once the brackets are in.

----------------------------------------------------------------------------

window width = r3 / r3 + r4

tau = r5 * c1
minimum_point = half_vdd * (r4 / r3 + r4)
height_point = half_vdd * (r4 / r3 + r4) + (r3 / r3 + r4) * vdd
v(t) = 3v3 + (minimum_point - 3v3) * exp(-t/tau) # high
v(t) = (v_start) * exp(-t/tau) # low

v(time_charge) = height_point
height_point = 3v3 + (minimum_point - 3v3) * exp(-time_charge/tau)
(height_point - 3v3) / (minimum_point - 3v3) = exp(-time_charge/tau)
ln(height_point - 3v3 / minimum_point - 3v3) = -t / tau
time_charge = -tau * ln(height_point - 3v3 / minimum_point - 3v3)

v(t) = (v_start) * exp(-t/tau) # low
v(time_discharge) = minimum_point
minimum_point = height_point * exp(-t/tau)
ln(minimum_point/height_point) = -t/tau
-tau * ln(minimum_point/height_point) = time_discharge

T = time_discharge + time_charge
k = r2 / (r1 + r2)
half_vdd = k * vdd

T = -tau * ln(minimum_point / height_point)
    + -tau * ln(height_point - 3v3 / minimum_point - 3v3)

T = -(r5 * c1) *
    (ln(half_vdd * (r4 / r3 + r4) / half_vdd * (r4 / r3 + r4) + (r3 / r3 + r4) * vdd) +
     ln(half_vdd * (r4 / r3 + r4) + (r3 / r3 + r4) * vdd - vdd / half_vdd * (r4 / r3 + r4) - vdd))

T = -(r5 * c1) * (
    ln(
        ((r2 / (r1 + r2)) *
        (r3 / (r3 + r4)))
        /
        (
            (r2 / (r1 + r2)) * (r4 / (r3 + r4))
            +
            (r3 / (r3 + r4))
        )
    )
    + ln(
        ((r2 / (r1 + r2)) * (r4 / (r3 + r4))
        +
        ((r3 / (r3 + r4)) - 1)) /
        (((r2 / (r1 + r2)) * (r4 / r3 + r4)) - 1)
    )
)

assuming r2/r1+r2 = 0.5 and r4/r3+r4 = 0.5, ans assuming T is 1,
ln product is 2.19722457734
1 / ln product is 0.45511961331
1 / ln_product * C assuming C is 1uF = 455119.613313
the closest resistor we have is 475000
the tolerance for this resistor is 1%
therefore (ideal resistor_value) / 475000 * 1.01 = 0.94865995479
1 - 0.94865995479 = 0.0513400452 * 100 = 5.13400452039%
5% is worst case

"""


# =========================================================================== #
# SETTINGS -- edit these, then run:  python get_good_numbers.py
# =========================================================================== #

# How many candidates to list.
SHOW_COUNT: int = 15

# Allow R5 to be two stocked resistors in series. The parts list is a sparse
# ladder and the value that is wanted falls in a hole between 402K and 475K, so
# a single resistor cannot get close. Two in series can, at the cost of a part.
ALLOW_SERIES: bool = True

# Extra sections, each off by default because they are long.
SHOW_DETAIL: bool = False  # error budget and all 64 tolerance corners
SHOW_LADDER: bool = False  # where the stocked resistor ladder has holes
SHOW_CAVEATS: bool = False  # what this model does not cover

# Where the full scored table is written. tolerance_simulation.py can read the
# six values and six tolerances straight out of it.
CSV_PATH: str = "candidates.csv"

# =========================================================================== #

# The design spec.
TARGET_PERIOD: float = 1.0
TOLERANCE_BAND: float = 0.10
VDD: float = 3.3

# Fixed by the design and not up for negotiation. R1 = R2 holds the duty cycle
# at exactly 50%, and R3 = R4 makes the hysteresis window symmetric about
# half_vdd. Only R5 and C1 are searched over.
FIXED: dict[str, float] = {"R1": 10e3, "R2": 10e3, "R3": 10e3, "R4": 10e3}

# Bounds on the timing resistor. These are real design decisions, not arbitrary
# limits, so the reasoning for each is written down rather than left implicit.
R5_MIN: float = 50e3  # Below this the op-amp sources real current into the RC.
R5_MAX: float = 2e6  # Above this, bias current and board leakage start to show.

# A tolerance is a guaranteed bound, not a standard deviation, so turning one
# into the other needs a convention stated out loud. Three sigma is the
# industry-standard reading, with 99.7% of parts inside the marked band.
# Setting this to 2.0 is the conservative alternative, and it is worth running
# the whole thing both ways.
SIGMAS_PER_TOLERANCE: float = 3.0

# Every resistor on the parts list is 1%.
RESISTOR_TOLERANCE: float = 0.01

# Capacitor tolerance is not printed in the Digi-Key description, so it is
# decoded from the EIA tolerance letter in the manufacturer part number:
# F = +/-1%, J = +/-5%, K = +/-10%, M = +/-20%.
CAPACITOR_TOLERANCES: dict[str, float] = {
    "GRT188R61C106KE13D": 0.10,  # 10uF X5R, ...106[K]E13D
    "C0603C105K3RACTU": 0.10,  # 1uF X7R, ...105[K]3RAC
    "CC0603JRX7R9BB104": 0.05,  # 0.1uF X7R, CC0603[J]RX7R
    "C0603C103J5RACTU": 0.05,  # 0.01uF X7R, ...103[J]5RAC
    "GRM1885C1H102FA01J": 0.01,  # 1000pF C0G, ...102[F]A01J
    "CL10C101FB8NNNC": 0.01,  # 100pF C0G, ...101[F]B8
    "GRM1885C1H100FA01J": 0.01,  # 10pF C0G, ...100[F]A01J
}

# The six parts in the timing loop. R6 and D1 are the LED branch and cannot
# move the period, so they are excluded from every tolerance calculation.
TIMING_PARTS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "C1")

PARTS_LIST: str = "Eclectronics MP1 Parts List.xlsx"

SPREADSHEET_NAMESPACE: str = (
    "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
)
UNIT_MULTIPLIERS: dict[str, float] = {
    "": 1.0,
    "K": 1e3,
    "M": 1e6,
    "PF": 1e-12,
    "NF": 1e-9,
    "UF": 1e-6,
}


class Part(NamedTuple):
    """A single purchasable component from the parts list."""

    value: float
    tolerance: float
    digikey: str
    manufacturer: str
    description: str


class Candidate(NamedTuple):
    """One buyable way of building R5 and C1, already scored."""

    resistors: tuple[Part, ...]  # One part, or two in series.
    capacitor: Part
    resistance: float
    resistor_sigma: float  # Fractional, already combined across the series.
    period: float
    error: float
    sigma: float  # Fractional spread of the period.
    estimated_yield: float
    corner_low: float
    corner_high: float
    margin: float

    @property
    def label(self) -> str:
        """
        Builds a short name for the candidate, for tables and for the CSV.

        Returns
        -------
        str
            The resistor values making up R5, joined with a plus sign.
        """
        return " + ".join(format_value(part.value) for part in self.resistors)


# --------------------------------------------------------------------------- #
# Timing model (README sections 3.1 to 3.7)
# --------------------------------------------------------------------------- #


def period_halves(
    r1: float, r2: float, r3: float, r4: float, r5: float, c1: float
) -> tuple[float, float]:
    """
    Calculates the two half-cycles of the oscillator, in seconds.

    This is the two-half pair from README section 3.6, deliberately not the
    collapsed one-liner from 3.7, even though R1 == R2 makes them agree here.
    Keeping the general form means the tolerance corners can push R1 and R2
    apart, which is the only way R1/R2 error shows up at all.

    VDD does not appear anywhere because it cancels out of both logarithms.

    Parameters
    ----------
    r1, r2 : float
        The mid-rail divider, in ohms.
    r3, r4 : float
        The hysteresis divider, in ohms.
    r5 : float
        The timing resistor, in ohms.
    c1 : float
        The timing capacitor, in farads.

    Returns
    -------
    tuple[float, float]
        The charging half-cycle (Vout high) and the discharging half-cycle
        (Vout low), in seconds.
    """
    # Weight of half_vdd and of Vout at the non-inverting pin. These sum to 1.
    weight_reference: float = r4 / (r3 + r4)
    weight_output: float = r3 / (r3 + r4)
    # Divider ratio, so that half_vdd = divider_ratio * VDD. This is 0.5 only
    # when R1 == R2, which is the whole reason R1/R2 mismatch matters.
    divider_ratio: float = r2 / (r1 + r2)
    tau: float = r5 * c1

    charge_time: float = tau * log(
        (1.0 - weight_reference * divider_ratio)
        / (weight_reference * (1.0 - divider_ratio))
    )
    discharge_time: float = tau * log(
        (weight_reference * divider_ratio + weight_output)
        / (weight_reference * divider_ratio)
    )

    return charge_time, discharge_time


def period(r5: float, c1: float, **overrides: float) -> float:
    """
    Calculates the oscillator period for a given RC, in seconds.

    Parameters
    ----------
    r5 : float
        The timing resistor, in ohms.
    c1 : float
        The timing capacitor, in farads.
    **overrides : float
        Any of R1 through R4 to use instead of their fixed values, which is
        how the tolerance corners pull the matched pairs apart.

    Returns
    -------
    float
        The period, in seconds.
    """
    design: dict[str, float] = {**FIXED, **overrides, "R5": r5, "C1": c1}
    charge_time, discharge_time = period_halves(
        *(design[part] for part in TIMING_PARTS)
    )

    return charge_time + discharge_time


def duty_cycle(r5: float, c1: float, **overrides: float) -> float:
    """
    Calculates the fraction of the period spent with Vout high, so with the
    LED lit. This is exactly 0.5 if and only if R1 == R2.

    Parameters
    ----------
    r5 : float
        The timing resistor, in ohms.
    c1 : float
        The timing capacitor, in farads.
    **overrides : float
        Any of R1 through R4 to use instead of their fixed values.

    Returns
    -------
    float
        The duty cycle, as a fraction.
    """
    design: dict[str, float] = {**FIXED, **overrides, "R5": r5, "C1": c1}
    charge_time, discharge_time = period_halves(
        *(design[part] for part in TIMING_PARTS)
    )

    return charge_time / (charge_time + discharge_time)


def thresholds(**overrides: float) -> tuple[float, float]:
    """
    Calculates the two comparator thresholds, for checking that they stay
    inside the op-amp's input common-mode range.

    Parameters
    ----------
    **overrides : float
        Any of R1 through R4 to use instead of their fixed values.

    Returns
    -------
    tuple[float, float]
        The lower and upper thresholds, in volts.
    """
    design: dict[str, float] = {**FIXED, **overrides}
    weight_reference: float = design["R4"] / (design["R3"] + design["R4"])
    weight_output: float = design["R3"] / (design["R3"] + design["R4"])
    divider_ratio: float = design["R2"] / (design["R1"] + design["R2"])

    lower: float = weight_reference * divider_ratio * VDD
    upper: float = (weight_reference * divider_ratio + weight_output) * VDD

    return lower, upper


def required_product() -> float:
    """
    Solves for the R5 * C1 product that lands the period exactly on target.

    The period is exactly proportional to the RC product, so evaluating it once
    at R5 * C1 = 1 gives the constant of proportionality and the answer follows
    by division. Doing it this way instead of hardcoding 1 / (2 * ln 3) keeps
    it correct if R3 and R4 are ever changed.

    Returns
    -------
    float
        The RC product needed, in seconds.
    """
    return TARGET_PERIOD / period(1.0, 1.0)


def sensitivities(r5: float, c1: float, nudge: float = 0.01) -> dict[str, float]:
    """
    Calculates the normalized sensitivity (dT/T) / (dx/x) of each timing part.

    A central difference is used so that the second-order R1/R2 terms come out
    near zero, instead of picking up a spurious first-order slope from a
    one-sided step.

    Parameters
    ----------
    r5 : float
        The timing resistor, in ohms.
    c1 : float
        The timing capacitor, in farads.
    nudge : float
        The fractional step to perturb each part by.

    Returns
    -------
    dict[str, float]
        The sensitivity of the period to each timing part.
    """
    nominal: dict[str, float] = {**FIXED, "R5": r5, "C1": c1}
    base_period: float = period(r5, c1)

    results: dict[str, float] = {}
    for part in TIMING_PARTS:
        swung: list[float] = []
        for direction in (1 + nudge, 1 - nudge):
            pushed: dict[str, float] = {**nominal, part: nominal[part] * direction}
            swung.append(period(pushed.pop("R5"), pushed.pop("C1"), **pushed))
        results[part] = (swung[0] - swung[1]) / base_period / (2 * nudge)

    return results


def worst_case_period(
    r5: float, c1: float, tolerances: dict[str, float]
) -> tuple[float, float]:
    """
    Finds the hardest bound on the period by walking every tolerance corner.

    All 2**6 corners are evaluated by brute force rather than pushing each part
    to the sign of its sensitivity, because the period is not monotonic in
    R1/R2 -- mismatch lengthens the period in either direction. Sixty-four
    evaluations is free, and unlike a Monte Carlo this is a guarantee rather
    than a sample.

    Parameters
    ----------
    r5 : float
        The timing resistor, in ohms.
    c1 : float
        The timing capacitor, in farads.
    tolerances : dict[str, float]
        The fractional tolerance of each timing part.

    Returns
    -------
    tuple[float, float]
        The shortest and longest achievable periods, in seconds.
    """
    nominal: dict[str, float] = {**FIXED, "R5": r5, "C1": c1}

    periods: list[float] = []
    # Each part independently sits at one end of its band or the other, so the
    # corners are the 64 rows of a six-bit counter.
    for corner in range(2 ** len(TIMING_PARTS)):
        pushed: dict[str, float] = {}
        for bit, part in enumerate(TIMING_PARTS):
            sign: int = 1 if corner >> bit & 1 else -1
            pushed[part] = nominal[part] * (1 + sign * tolerances[part])
        periods.append(period(pushed.pop("R5"), pushed.pop("C1"), **pushed))

    return min(periods), max(periods)


def part_sigmas(capacitor_tolerance: float, resistor_sigma: float) -> dict[str, float]:
    """
    Turns the marked tolerances into standard deviations, one per part.

    R5 is handled separately because a series pair is statistically stiffer
    than either resistor alone, so its sigma has already been combined by the
    caller rather than being read off a single tolerance.

    Parameters
    ----------
    capacitor_tolerance : float
        The fractional tolerance of the capacitor.
    resistor_sigma : float
        The fractional standard deviation of R5.

    Returns
    -------
    dict[str, float]
        The fractional standard deviation of each timing part.
    """
    spreads: dict[str, float] = {
        part: RESISTOR_TOLERANCE / SIGMAS_PER_TOLERANCE for part in TIMING_PARTS
    }
    spreads["R5"] = resistor_sigma
    spreads["C1"] = capacitor_tolerance / SIGMAS_PER_TOLERANCE

    return spreads


def predicted_sigma(r5: float, c1: float, spreads: dict[str, float]) -> float:
    """
    Predicts the fractional spread of the period from the sensitivities.

    For independent errors the relative variances add in quadrature, weighted
    by the square of each sensitivity. This is the closed-form prediction that
    tolerance_simulation.py's histogram should land on; a disagreement between
    the two means one of them is wrong, which is the whole reason for computing
    it both ways.

    Parameters
    ----------
    r5 : float
        The timing resistor, in ohms.
    c1 : float
        The timing capacitor, in farads.
    spreads : dict[str, float]
        The fractional standard deviation of each timing part.

    Returns
    -------
    float
        The fractional standard deviation of the period.
    """
    weights: dict[str, float] = sensitivities(r5, c1)

    return sqrt(sum((weights[part] * spreads[part]) ** 2 for part in TIMING_PARTS))


def normal_cdf(z: float) -> float:
    """
    Evaluates the standard normal CDF, since there is no scipy in the venv.

    Parameters
    ----------
    z : float
        The standard score.

    Returns
    -------
    float
        The probability of drawing below z.
    """
    return 0.5 * (1 + erf(z / sqrt(2)))


def estimated_yield(nominal_period: float, sigma: float) -> float:
    """
    Estimates the fraction of builds landing inside the spec window.

    Because S_C1 is exactly 1 and the capacitor dominates the error budget, the
    period is very nearly a linear function of one normal variable, so a normal
    model for the period is fair. It ignores the hard truncation at the marked
    tolerance, which makes it slightly pessimistic in the tails -- a real Monte
    Carlo with truncated sampling should come out a little better than this.

    Parameters
    ----------
    nominal_period : float
        The period with every part at its marked value, in seconds.
    sigma : float
        The fractional standard deviation of the period.

    Returns
    -------
    float
        The estimated yield, as a fraction.
    """
    spread: float = sigma * nominal_period
    upper: float = normal_cdf(
        ((1 + TOLERANCE_BAND) * TARGET_PERIOD - nominal_period) / spread
    )
    lower: float = normal_cdf(
        ((1 - TOLERANCE_BAND) * TARGET_PERIOD - nominal_period) / spread
    )

    return upper - lower


# --------------------------------------------------------------------------- #
# Parts list
# --------------------------------------------------------------------------- #


def spreadsheet_rows(path: Path) -> list[list[str]]:
    """
    Reads the parts list spreadsheet into rows of strings.

    An .xlsx file is a zip of XML, so this uses zipfile and ElementTree instead
    of pulling openpyxl into the venv for one file.

    Parameters
    ----------
    path : Path
        The filepath to the parts list spreadsheet.

    Returns
    -------
    list[list[str]]
        Columns A through D of every row in the sheet.
    """
    rows: list[list[str]] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = [
            "".join(node.text or "" for node in item.iter(f"{SPREADSHEET_NAMESPACE}t"))
            for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).iter(
                f"{SPREADSHEET_NAMESPACE}si"
            )
        ]
        sheet: ET.Element = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

        for row in sheet.iter(f"{SPREADSHEET_NAMESPACE}row"):
            cells: dict[str, str] = {}
            for cell in row.iter(f"{SPREADSHEET_NAMESPACE}c"):
                column: str = re.sub(r"\d", "", cell.get("r", "A"))
                value: Optional[ET.Element] = cell.find(f"{SPREADSHEET_NAMESPACE}v")
                if value is None or value.text is None:
                    continue
                # Most text cells are indices into the shared string table.
                cells[column] = (
                    shared_strings[int(value.text)]
                    if cell.get("t") == "s"
                    else value.text
                )
            rows.append([cells.get(column, "") for column in ("A", "B", "C", "D")])

    return rows


def read_parts_list(path: Path) -> tuple[list[Part], list[Part]]:
    """
    Reads the resistors and capacitors out of the parts list spreadsheet.

    Parameters
    ----------
    path : Path
        The filepath to the parts list spreadsheet.

    Returns
    -------
    tuple[list[Part], list[Part]]
        The resistors and the capacitors, each sorted by value.
    """
    resistors: list[Part] = []
    capacitors: list[Part] = []
    # Column A only names the section on its first row, so it has to be carried
    # down to the rows underneath it.
    section: str = ""

    for kind, description, digikey, manufacturer in spreadsheet_rows(path):
        section = kind or section

        if section == "Resistors":
            resistor: Optional[Part] = parse_resistor(
                description, digikey, manufacturer
            )
            if resistor is not None:
                resistors.append(resistor)

        elif section == "Capacitors":
            capacitor: Optional[Part] = parse_capacitor(
                description, digikey, manufacturer
            )
            if capacitor is not None:
                capacitors.append(capacitor)

    return sorted(resistors), sorted(capacitors)


def parse_resistor(description: str, digikey: str, manufacturer: str) -> Optional[Part]:
    """
    Pulls the value and tolerance out of one resistor's description.

    Parameters
    ----------
    description : str
        The Digi-Key description, such as "RES SMD 475K OHM 1% 1/10W 0603".
    digikey : str
        The Digi-Key catalog number.
    manufacturer : str
        The manufacturer part number.

    Returns
    -------
    Part | None
        The parsed part, or None if the row is not a resistor.
    """
    match: Optional[re.Match[str]] = re.search(
        r"RES SMD ([\d.]+)\s*([KM]?)\s*OHM\s+([\d.]+)%", description
    )
    if match is None:
        return None

    return Part(
        value=float(match.group(1)) * UNIT_MULTIPLIERS[match.group(2)],
        tolerance=float(match.group(3)) / 100.0,
        digikey=digikey,
        manufacturer=manufacturer,
        description=description,
    )


def parse_capacitor(
    description: str, digikey: str, manufacturer: str
) -> Optional[Part]:
    """
    Pulls the value out of one capacitor's description, and its tolerance out
    of the EIA letter in the manufacturer part number.

    Parameters
    ----------
    description : str
        The Digi-Key description, such as "CAP CER 1UF 25V X7R 0603".
    digikey : str
        The Digi-Key catalog number.
    manufacturer : str
        The manufacturer part number.

    Returns
    -------
    Part | None
        The parsed part, or None if the row is not a capacitor.

    Raises
    ------
    SystemExit
        If the part number is not in CAPACITOR_TOLERANCES, since guessing a
        capacitor's tolerance would quietly invalidate every yield below.
    """
    match: Optional[re.Match[str]] = re.search(
        r"CAP CER ([\d.]+)\s*(PF|NF|UF)\s+\d+V\s+(\S+)", description
    )
    if match is None:
        return None

    if manufacturer not in CAPACITOR_TOLERANCES:
        raise SystemExit(
            f"no tolerance decoded for capacitor {manufacturer}; "
            "add its EIA tolerance letter to CAPACITOR_TOLERANCES"
        )

    return Part(
        value=float(match.group(1)) * UNIT_MULTIPLIERS[match.group(2)],
        tolerance=CAPACITOR_TOLERANCES[manufacturer],
        digikey=digikey,
        manufacturer=manufacturer,
        description=description,
    )


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #


def score_candidate(resistors: tuple[Part, ...], capacitor: Part) -> Candidate:
    """
    Scores one way of building R5, against one capacitor.

    Parameters
    ----------
    resistors : tuple[Part, ...]
        The resistor, or the two resistors in series, making up R5.
    capacitor : Part
        The timing capacitor.

    Returns
    -------
    Candidate
        The scored candidate.
    """
    resistance: float = sum(part.value for part in resistors)

    # Two independent resistors in series are statistically stiffer than one,
    # because their errors partly cancel. At the worst-case corner they are
    # not: pushing both to the same end pushes the total to the same end.
    resistor_sigma: float = (
        sqrt(
            sum(
                (part.value * part.tolerance / SIGMAS_PER_TOLERANCE) ** 2
                for part in resistors
            )
        )
        / resistance
    )
    worst_resistor_tolerance: float = (
        sum(part.value * part.tolerance for part in resistors) / resistance
    )

    tolerances: dict[str, float] = {part: RESISTOR_TOLERANCE for part in TIMING_PARTS}
    tolerances["R5"] = worst_resistor_tolerance
    tolerances["C1"] = capacitor.tolerance

    this_period: float = period(resistance, capacitor.value)
    sigma: float = predicted_sigma(
        resistance,
        capacitor.value,
        part_sigmas(capacitor.tolerance, resistor_sigma),
    )
    corner_low, corner_high = worst_case_period(resistance, capacitor.value, tolerances)

    # How much of the +/-10% window is still unused at the worse corner. A
    # positive margin means every build in the tolerance band passes.
    worst_error: float = max(
        abs(corner_low - TARGET_PERIOD), abs(corner_high - TARGET_PERIOD)
    )

    return Candidate(
        resistors=resistors,
        capacitor=capacitor,
        resistance=resistance,
        resistor_sigma=resistor_sigma,
        period=this_period,
        error=this_period / TARGET_PERIOD - 1,
        sigma=sigma,
        estimated_yield=estimated_yield(this_period, sigma),
        corner_low=corner_low,
        corner_high=corner_high,
        margin=TOLERANCE_BAND - worst_error / TARGET_PERIOD,
    )


def search(resistors: list[Part], capacitors: list[Part]) -> list[Candidate]:
    """
    Scores every single-resistor R5 and C1 pairing the parts list can build.

    Parameters
    ----------
    resistors : list[Part]
        Every resistor on the parts list.
    capacitors : list[Part]
        Every capacitor on the parts list.

    Returns
    -------
    list[Candidate]
        Every viable pairing, best estimated yield first.
    """
    candidates: list[Candidate] = []

    for capacitor in capacitors:
        for resistor in resistors:
            if not R5_MIN <= resistor.value <= R5_MAX:
                continue
            candidates.append(score_candidate((resistor,), capacitor))

    return rank(candidates)


def search_series(
    resistors: list[Part], capacitors: list[Part], smallest_share: float = 0.02
) -> list[Candidate]:
    """
    Scores every pairing where R5 is split across two stocked resistors.

    With R3/R4 locked at 1:1 the RC product is the only knob left, and a single
    stocked resistor cannot get closer to the target than the ladder allows.
    Two in series can, at the cost of one extra part.

    Parameters
    ----------
    resistors : list[Part]
        Every resistor on the parts list.
    capacitors : list[Part]
        Every capacitor on the parts list.
    smallest_share : float
        Ignore pairs where one resistor is a smaller fraction of the total than
        this, since it buys no accuracy and adds another error term.

    Returns
    -------
    list[Candidate]
        Every viable pairing, best estimated yield first.
    """
    candidates: list[Candidate] = []

    for capacitor in capacitors:
        for first, second in combinations(resistors, 2):
            total: float = first.value + second.value
            if not R5_MIN <= total <= R5_MAX:
                continue
            if min(first.value, second.value) < smallest_share * total:
                continue
            candidates.append(score_candidate((first, second), capacitor))

    return rank(candidates)


def rank(candidates: list[Candidate]) -> list[Candidate]:
    """
    Orders candidates by estimated yield, then by nominal error.

    Yield leads because it is the number the spec actually cares about. Nominal
    error breaks ties, since two builds can share a yield to four decimal places
    and still differ in where they sit.

    Parameters
    ----------
    candidates : list[Candidate]
        The candidates to order.

    Returns
    -------
    list[Candidate]
        The same candidates, best first.
    """
    return sorted(candidates, key=lambda item: (-item.estimated_yield, abs(item.error)))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_value(value: float, is_capacitor: bool = False) -> str:
    """
    Formats a component value the way a parts list would print it.

    Parameters
    ----------
    value : float
        The component value, in ohms or farads.
    is_capacitor : bool
        Whether the value is a capacitance, which decides the units.

    Returns
    -------
    str
        The formatted value.
    """
    if is_capacitor:
        for suffix, scale in (("uF", 1e-6), ("nF", 1e-9), ("pF", 1e-12)):
            if value >= scale:
                return f"{value / scale:.4g}{suffix}"
        return f"{value:.4g}F"

    for suffix, scale in (("M", 1e6), ("K", 1e3), ("", 1.0)):
        if value >= scale:
            return f"{value / scale:.4g}{suffix}"

    return f"{value:.4g}"


def bar_header(width: int = 45) -> str:
    """
    Draws the axis labels for the text plot, lined up with error_bar().

    Parameters
    ----------
    width : int
        How many characters wide the bar is.

    Returns
    -------
    str
        The rendered header row.
    """
    span: float = TOLERANCE_BAND * 1.5
    centre: int = width // 2
    row: list[str] = [" "] * width

    for label, error in ((f"-{span:.0%}", -span), ("0", 0.0), (f"+{span:.0%}", span)):
        start: int = min(
            max(int(round(centre * (1 + error / span))) - len(label) // 2, 0),
            width - len(label),
        )
        row[start : start + len(label)] = label

    return "".join(row)


def error_bar(error: float, width: int = 45) -> str:
    """
    Draws one row of a text plot showing how far off target a period is.

    Parameters
    ----------
    error : float
        The fractional period error.
    width : int
        How many characters wide the bar is.

    Returns
    -------
    str
        The rendered row.
    """
    span: float = TOLERANCE_BAND * 1.5
    centre: int = width // 2
    row: list[str] = [" "] * width

    row[centre] = "|"
    row[max(0, int(round(centre * (1 - TOLERANCE_BAND / span))))] = "["
    row[min(width - 1, int(round(centre * (1 + TOLERANCE_BAND / span))))] = "]"

    position: int = int(round(centre * (1 + error / span)))
    row[min(max(position, 0), width - 1)] = "#" if abs(error) <= TOLERANCE_BAND else "x"

    return "".join(row)


def report_ladder(resistors: list[Part], capacitor: Part) -> None:
    """
    Walks the rungs of the resistor ladder around the value that is wanted.

    This is where the sparse parts list shows itself: the exact value falls in
    a hole, and stepping along the rungs either side shows how far the nearest
    buyable one leaves you.

    Parameters
    ----------
    resistors : list[Part]
        Every resistor on the parts list.
    capacitor : Part
        The timing capacitor being paired with.
    """
    wanted: float = required_product() / capacitor.value

    print(
        f"\nwith C1 = {format_value(capacitor.value, True)} "
        f"(+/-{capacitor.tolerance:.0%}, {capacitor.digikey})"
    )
    print(
        f"  R5 * C1 has to be {required_product():.6f} s, so R5 wants to be "
        f"{wanted:,.1f} ohm"
    )

    # Show the handful of rungs either side of the value that is wanted.
    ladder: list[Part] = [
        part for part in resistors if 0.25 * wanted <= part.value <= 4 * wanted
    ]
    if not ladder:
        print("  nothing on the ladder is anywhere near that")
        return

    print(f"\n  {'R5':>10} {'period':>9} {'error':>9}   {bar_header()}")
    print("  " + "-" * 92)

    printed_gap: bool = False
    for part in ladder:
        # Drop the wanted value into the table where it belongs, so the hole in
        # the ladder is visible rather than implied.
        if not printed_gap and part.value > wanted:
            print(
                f"  {wanted:>10,.0f} {TARGET_PERIOD:>9.4f} {0.0:>+8.2%}   "
                f"{error_bar(0.0)}  <- wanted, NOT STOCKED"
            )
            printed_gap = True

        this_period: float = period(part.value, capacitor.value)
        error: float = this_period / TARGET_PERIOD - 1
        print(
            f"  {format_value(part.value):>10} {this_period:>9.4f} {error:>+8.2%}   "
            f"{error_bar(error)}  {part.digikey}"
        )

    nearest: Part = min(ladder, key=lambda part: abs(part.value - wanted))
    print(
        f"\n  nearest buyable is {format_value(nearest.value)}, leaving "
        f"{period(nearest.value, capacitor.value) / TARGET_PERIOD - 1:+.2%} "
        "of nominal error"
    )


def print_table(candidates: list[Candidate], title: str) -> None:
    """
    Prints a ranked table of candidates.

    Parameters
    ----------
    candidates : list[Candidate]
        The candidates to print.
    title : str
        A heading for the table.
    """
    print(f"\n{title}")
    print(
        f"  {'R5':>17} {'C1':>7} {'Ctol':>5} {'R5*C1':>8} {'nominal':>9} {'err':>8} "
        f"{'sigma':>7} {'yield':>9} {'worst case':>17} {'margin':>8}  WC"
    )
    print("  " + "-" * 122)

    for item in candidates:
        verdict: str = "pass" if item.margin >= 0 else "FAIL"
        print(
            f"  {item.label:>17} "
            f"{format_value(item.capacitor.value, True):>7} "
            f"{item.capacitor.tolerance:>4.0%} "
            f"{item.resistance * item.capacitor.value:>8.5f} "
            f"{item.period:>9.4f} {item.error:>+8.2%} "
            f"{item.sigma:>6.2%} {item.estimated_yield:>9.4%} "
            f"{item.corner_low:>7.4f}..{item.corner_high:<7.4f} "
            f"{item.margin:>+7.2%}  {verdict}"
        )


def candidate_row(position: int, item: Candidate) -> dict[str, object]:
    """
    Flattens one candidate into the columns written to the CSV.

    Parameters
    ----------
    position : int
        The candidate's rank, counting from one.
    item : Candidate
        The candidate to flatten.

    Returns
    -------
    dict[str, object]
        One row, keyed by column name.
    """
    return {
        "rank": position,
        "label": f"{item.label} x {format_value(item.capacitor.value, True)}",
        "r5_parts": "+".join(str(part.value) for part in item.resistors),
        "r5_digikey": "+".join(part.digikey for part in item.resistors),
        "c1_digikey": item.capacitor.digikey,
        "r1": FIXED["R1"],
        "r2": FIXED["R2"],
        "r3": FIXED["R3"],
        "r4": FIXED["R4"],
        "r5": item.resistance,
        "c1": item.capacitor.value,
        "r1_tolerance": RESISTOR_TOLERANCE,
        "r2_tolerance": RESISTOR_TOLERANCE,
        "r3_tolerance": RESISTOR_TOLERANCE,
        "r4_tolerance": RESISTOR_TOLERANCE,
        "r5_tolerance": RESISTOR_TOLERANCE,
        "c1_tolerance": item.capacitor.tolerance,
        "r5_sigma": item.resistor_sigma,
        "nominal_period": item.period,
        "nominal_error": item.error,
        "duty_cycle": duty_cycle(item.resistance, item.capacitor.value),
        "predicted_sigma": item.sigma,
        "estimated_yield": item.estimated_yield,
        "corner_low": item.corner_low,
        "corner_high": item.corner_high,
        "worst_case_margin": item.margin,
    }


def write_table(candidates: list[Candidate], path: Path) -> None:
    """
    Writes the scored candidates out for tolerance_simulation.py to read.

    Every column the Monte Carlo needs is here, so the values never have to be
    typed in twice: the six nominal values, the six tolerances, and this file's
    own closed-form predictions to check the sampled ones against.

    Parameters
    ----------
    candidates : list[Candidate]
        The candidates to write, already ranked.
    path : Path
        Where to write the CSV.
    """
    columns: list[str] = list(candidate_row(1, candidates[0]).keys())

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for position, item in enumerate(candidates, start=1):
            writer.writerow(candidate_row(position, item))


def report_error_budget(candidate: Candidate) -> None:
    """
    Prints where the period's spread actually comes from, part by part.

    The shares are what make the Monte Carlo interpretable instead of a black
    box: they say in advance which component the histogram's width is really
    made of.

    Parameters
    ----------
    candidate : Candidate
        The candidate to break down.
    """
    weights: dict[str, float] = sensitivities(
        candidate.resistance, candidate.capacitor.value
    )
    spreads: dict[str, float] = part_sigmas(
        candidate.capacitor.tolerance, candidate.resistor_sigma
    )

    print(
        f"\n  error budget, reading each tolerance as "
        f"{SIGMAS_PER_TOLERANCE:.0f} sigma:"
    )
    print(f"    {'part':<5} {'S_x':>9} {'sigma_x':>9} {'S*sigma':>9} {'share':>8}")
    for part in TIMING_PARTS:
        contribution: float = abs(weights[part] * spreads[part])
        share: float = (contribution / candidate.sigma) ** 2
        print(
            f"    {part:<5} {weights[part]:>+9.4f} {spreads[part]:>8.3%} "
            f"{contribution:>8.3%} {share:>8.1%}"
        )
    print(f"    predicted sigma of the period: {candidate.sigma:.3%}")


def report_candidate(candidate: Candidate, title: str) -> None:
    """
    Prints everything worth knowing about one candidate.

    Parameters
    ----------
    candidate : Candidate
        The candidate to describe.
    title : str
        A heading for the block.
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    for part in ("R1", "R2", "R3", "R4"):
        print(
            f"  {part:<3} = {format_value(FIXED[part]):>8}   "
            f"+/-{RESISTOR_TOLERANCE:.0%}"
        )
    for index, resistor in enumerate(candidate.resistors):
        name: str = "R5" if len(candidate.resistors) == 1 else f"R5{'ab'[index]}"
        print(
            f"  {name:<3} = {format_value(resistor.value):>8}   "
            f"+/-{resistor.tolerance:.0%}   {resistor.digikey}"
        )
    print(
        f"  C1  = {format_value(candidate.capacitor.value, True):>8}   "
        f"+/-{candidate.capacitor.tolerance:.0%}   {candidate.capacitor.digikey}, "
        f"{candidate.capacitor.description}"
    )

    threshold_low, threshold_high = thresholds()
    print(
        f"\n  nominal period    {candidate.period:.6f} s  "
        f"({candidate.period - TARGET_PERIOD:+.6f} s, {candidate.error:+.3%})"
    )
    print(
        f"  duty cycle        "
        f"{duty_cycle(candidate.resistance, candidate.capacitor.value):.4%}"
    )
    print(
        f"  thresholds        {threshold_low:.3f} V .. {threshold_high:.3f} V   "
        f"(rails 0 .. {VDD} V)"
    )

    report_error_budget(candidate)

    print("\n  worst case over all 64 tolerance corners:")
    print(
        f"    {candidate.corner_low:.6f} s .. {candidate.corner_high:.6f} s   "
        f"({candidate.corner_low - TARGET_PERIOD:+.2%} .. "
        f"{candidate.corner_high - TARGET_PERIOD:+.2%})"
    )
    if candidate.margin >= 0:
        print(
            f"    PASS with {candidate.margin:.2%} of the window to spare. This is a "
            "guarantee,\n    not a sample, so it needs no confidence interval."
        )
    else:
        print(
            f"    FAIL, over the +/-{TOLERANCE_BAND:.0%} window by "
            f"{-candidate.margin:.2%} at the bad corner."
        )

    print(
        f"\n  estimated yield   {candidate.estimated_yield:.4%}  "
        f"(about {1 - candidate.estimated_yield:.4%} of builds out of spec)"
    )
    print("    Closed form and untruncated, so a little pessimistic in the tails.")
    print("    tolerance_simulation.py should land near this, not exactly on it.")

    report_measured_yield(candidate)


def report_measured_yield(candidate: Candidate) -> None:
    """
    Prints the sampled yield, if tolerance_simulation.py can supply one.

    Parameters
    ----------
    candidate : Candidate
        The candidate to measure.
    """
    if MONTE_CARLO is None or not hasattr(MONTE_CARLO, "run_monte_carlo"):
        return

    values: dict[str, float] = {
        **FIXED,
        "R5": candidate.resistance,
        "C1": candidate.capacitor.value,
    }
    tolerances: dict[str, float] = {part: RESISTOR_TOLERANCE for part in TIMING_PARTS}
    tolerances["C1"] = candidate.capacitor.tolerance

    # Guarded by the hasattr above; pylint cannot see into a module that is
    # still empty, so the member check has to be silenced here.
    periods = MONTE_CARLO.run_monte_carlo(  # pylint: disable=no-member
        values, tolerances
    )
    inside: int = sum(
        1
        for sampled in periods
        if abs(sampled - TARGET_PERIOD) <= TOLERANCE_BAND * TARGET_PERIOD
    )
    print(f"    measured by tolerance_simulation: {inside / len(periods):.4%}")


def print_shortlist(candidates: list[Candidate], count: int) -> None:
    """The answer: the best buyable R5/C1 pairs, best first."""
    print(
        f"{'#':>3}  {'R5':>17}  {'C1':>7}  {'period':>9}  {'err':>8}  "
        f"{'yield':>8}  {'worst case':>16}"
    )
    print("-" * 78)

    for position, item in enumerate(candidates[:count], start=1):
        print(
            f"{position:>3}  {item.label:>17}  "
            f"{format_value(item.capacitor.value, True):>7}  "
            f"{item.period:>9.4f}  {item.error:>+8.2%}  "
            f"{item.estimated_yield:>8.3%}  "
            f"{item.corner_low:>7.4f}..{item.corner_high:<7.4f}"
            + ("" if item.margin >= 0 else "  (corner misses)")
        )


def main() -> int:
    """Main function. Everything it reads is in the SETTINGS block up top."""
    here: Path = Path(__file__).resolve().parent
    resistors, capacitors = read_parts_list(here.parent / PARTS_LIST)

    singles: list[Candidate] = search(resistors, capacitors)
    series: list[Candidate] = search_series(resistors, capacitors)
    shortlist: list[Candidate] = rank(singles + series) if ALLOW_SERIES else singles

    print(
        f"R1 = R2 = {format_value(FIXED['R1'])}, "
        f"R3 = R4 = {format_value(FIXED['R3'])}, so R5 * C1 must be "
        f"{required_product():.6f} s for T = {TARGET_PERIOD:g} s.\n"
    )
    print_shortlist(shortlist, SHOW_COUNT)

    if SHOW_LADDER:
        print(f"\n{'=' * 78}\nWHERE THE LADDER HAS HOLES\n{'=' * 78}")
        for capacitor in capacitors:
            wanted: float = required_product() / capacitor.value
            if R5_MIN / 4 <= wanted <= R5_MAX * 4:
                report_ladder(resistors, capacitor)

    if SHOW_DETAIL:
        report_candidate(singles[0], "BEST WITH ONE RESISTOR")
        report_candidate(series[0], "BEST WITH TWO IN SERIES")

    if SHOW_CAVEATS:
        print(f"\n{'=' * 78}\nWHAT THIS DOES NOT MODEL\n{'=' * 78}")
        print("  * C1 dominates: S_C1 is exactly 1.0 and it is the loosest part.")
        print("    X7R also loses capacitance to DC bias and tempco, neither of")
        print("    which is in the marked tolerance.")
        print("  * R1/R2 and R3/R4 are treated as independent. Same-reel parts")
        print("    are correlated, so the real ratios are stiffer than this says.")
        print("  * Supply tolerance is left out because VDD cancels out.")
        print("  * The yield column assumes a normal, untruncated period.")
        print("    tolerance_simulation.py checks that against real samples.")

    combined: list[Candidate] = rank(singles + series)
    write_table(combined, Path(CSV_PATH))
    print(f"\n{len(combined):,} scored builds written to {CSV_PATH} (best first).")
    if not (SHOW_DETAIL or SHOW_LADDER or SHOW_CAVEATS):
        print(
            "Set SHOW_DETAIL, SHOW_LADDER or SHOW_CAVEATS to True at the top of\n"
            "this file for the error budget, the value ladder, or the modelling\n"
            "limits. ALLOW_SERIES allows two resistors in series for R5."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
