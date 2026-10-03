"""
Timing harness for :mod:`benchmarks.cases`.

Two things here matter more than the timings themselves.

**Scaling, not absolutes.** Each case runs at two sizes and the ratio is
reported. Per-item cost that stays flat as the workload grows is the signature
of a linear algorithm; per-item cost that climbs is a superlinear one. An
absolute millisecond threshold cannot tell those apart, and cannot tell a
regression apart from a slow CI runner.

**Operation counting, where it is available.** Timing is noisy enough that a
1.3x regression is arguable. Where a case can report how much work it did
rather than how long it took, that number is exact and is what the gate
actually asserts on.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .cases import BENCH_CASES, CALIBRATION_CASE, BenchCase

__all__ = [
    "MAX_SCALING_RATIO",
    "MAX_SLOWDOWN",
    "Result",
    "check_scaling",
    "compare",
    "main",
    "measure",
    "run_cases",
]

#: A case may not be slower than this multiple of its recorded baseline. Loose
#: on purpose: this catches an algorithmic regression (order of magnitude), not
#: ordinary machine noise. A tighter bound produces flaky CI.
MAX_SLOWDOWN = 1.5

#: Ratio between per-item cost at the two scales above which a case is reported
#: as growing. Linear work stays near 1.0; the threshold is generous because the
#: ratio is a difference of two noisy small-sample means, not a direct
#: measurement. Verified: repeated runs of the same code report flat.
MAX_SCALING_RATIO = 1.25

#: Repeats per measurement. The minimum is taken, since noise only ever adds
#: time; the median is reported too so a bimodal result is visible.
REPEATS = 5


@dataclass
class Result:
    """Timing and shape for one case at one scale."""

    name: str
    scale: int
    seconds: float
    #: Work items processed, when the case can report one. Enables a
    #: per-item cost that is independent of machine speed.
    items: int | None = None
    #: Whether `items` tracks the input, which is what makes a per-item scaling
    #: ratio meaningful. False when the case returns a fixed count, or one
    #: bounded by a cap (a context limited to a character budget reports fewer
    #: and fewer items per site as the batch grows, so its per-item cost would
    #: *fall* for linear work).
    scaling_meaningful: bool = True
    samples: list[float] = field(default_factory=list)

    @property
    def per_item_us(self) -> float | None:
        """Microseconds per work item, or None when the case reported no count."""
        if not self.items:
            return None
        return self.seconds / self.items * 1e6

    @property
    def seconds_formatted(self) -> str:
        """Best wall-clock time, scaled so seconds and milliseconds both read."""
        if self.seconds >= 1:
            return f"{self.seconds:.2f}s"
        if self.seconds >= 0.001:
            return f"{self.seconds * 1000:.1f}ms"
        return f"{self.seconds * 1e6:.0f}us"

    @property
    def normalised_us(self) -> float | None:
        """
        Microseconds per item, divided by the machine's own speed.

        Comparing a raw timing against a baseline recorded elsewhere is
        meaningless: the first CI run of this suite reported every case 1.6-1.7x
        slower than a baseline measured on the developer's machine, with no
        change to the code at all. Dividing by a calibration workload timed in
        the same run cancels the machine out, so the figure reflects the
        algorithm rather than the host.
        """
        per_item = self.per_item_us
        if per_item is None or not self.machine_us:
            return None
        return per_item / self.machine_us

    #: Microseconds the calibration case spends per unit of work, for this run.
    machine_us: float = 0.0


def _time_once(fn: Callable[[], Any]) -> tuple[float, Any]:
    """Run *fn*, returning (seconds, result)."""
    gc.collect()
    gc.disable()
    try:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
    finally:
        gc.enable()
    return elapsed, result


def measure(
    case: BenchCase, scale: int, *, repeats: int = REPEATS, machine_us: float = 0.0
) -> Result:
    """
    Time *case* at *scale*, returning the best of *repeats* runs.

    *machine_us* is the calibration cost per unit for this run; it is carried on
    the result so normalised figures can be reported without re-measuring.
    """
    samples: list[float] = []
    items: int | None = None
    for _ in range(repeats):
        elapsed, result = _time_once(lambda: case.run(scale))
        samples.append(elapsed)
        # The result is a cheap scalar (a character count, a file count) when
        # one is available; that is what makes the per-item figure meaningful.
        if isinstance(result, int) and not isinstance(result, bool):
            items = result
    return Result(
        name=case.name,
        scale=scale,
        seconds=min(samples),
        items=items,
        scaling_meaningful=case.scaling_meaningful,
        machine_us=machine_us,
        samples=samples,
    )


def run_cases(
    cases: tuple[BenchCase, ...] = BENCH_CASES, *, repeats: int = REPEATS
) -> list[Result]:
    """
    Measure every case at every scale.

    The calibration case is measured first, and its cost per unit becomes the
    divisor for every other figure in the run. A case that is not in *cases* is
    still calibrated against, so a filtered run stays comparable.
    """
    calibration = next((c for c in cases if c.name == CALIBRATION_CASE.name), None)
    if calibration is None:
        calibration = CALIBRATION_CASE

    machine_us = 0.0
    cal_results = [measure(calibration, scale, repeats=repeats) for scale in calibration.scales]
    if cal_results:
        # Average both scales: the calibration is linear, so this is a stable
        # estimate of how fast this machine is relative to the reference.
        per_item = [r.per_item_us for r in cal_results if r.per_item_us]
        if per_item:
            machine_us = sum(per_item) / len(per_item)

    results: list[Result] = []
    for case in cases:
        if case.name == CALIBRATION_CASE.name:
            results.extend(
                measure(case, scale, repeats=repeats, machine_us=machine_us)
                for scale in case.scales
            )
            continue
        for scale in case.scales:
            results.append(measure(case, scale, repeats=repeats, machine_us=machine_us))
    return results


def _fmt_us(value: float | None) -> str:
    """Format a microsecond figure, or a dash when the case reported no count."""
    if value is None:
        return "       -"
    if value >= 1000:
        return f"{value / 1000:8.2f}ms"
    return f"{value:8.1f}us"


def report(results: list[Result]) -> None:
    """Print a table, one row per case, showing cost per item at each scale."""
    print(f"{'case':<20} {'scale':>8} {'best':>10} {'per item':>11} {'normalised':>12}")
    print("-" * 64)
    by_case: dict[str, list[Result]] = {}
    for r in results:
        by_case.setdefault(r.name, []).append(r)

    for group in by_case.values():
        group.sort(key=lambda r: r.scale)
        for r in group:
            print(
                f"{r.name:<20} {r.scale:>8} {r.seconds_formatted:>10} {_fmt_us(r.per_item_us):>11}"
            )
        print()

    print("scaling: per-item cost at the large scale / at the small scale")
    print("-" * 52)
    for group in by_case.values():
        if len(group) != 2:
            continue
        small, large = group
        if not small.items or not large.items:
            continue
        if not small.scaling_meaningful or not large.scaling_meaningful:
            continue
        if not small.per_item_us or not large.per_item_us:
            continue
        ratio = large.per_item_us / small.per_item_us
        verdict = "flat (linear)" if ratio < MAX_SCALING_RATIO else f"GROWING {ratio:.2f}x"
        print(f"{small.name:<20} {small.scale:>7} -> {large.scale:<7} {verdict}")
    print()


def check_scaling(results: list[Result]) -> int:
    """
    Fail when a case's per-item cost grows with its input. Returns an exit code.

    This is the gate CI enforces, rather than an absolute-timing comparison.
    Two measured facts drove that choice:

    * Absolute timings are not comparable across machines. The first CI run of
      this suite reported every case 1.6-1.7x slower than a baseline recorded
      locally, with no code change.
    * Even on one machine, absolute timings are not stable. Three consecutive
      runs of identical code produced 69.1 ms, 69.4 ms and 92.7 ms for
      `parse_html` — a 1.34x spread, which a 1.5x threshold cannot survive
      without flaking.

    The scaling ratio is a comparison *within* a single run, so both problems
    cancel. It is also the property that actually matters: a superlinear cost is
    what makes a 500-page crawl take hours instead of minutes, and it is exactly
    what reintroducing a per-page full-document scan looks like. Every case
    reported flat across repeated runs.
    """
    by_case: dict[str, list[Result]] = {}
    for r in results:
        if r.items and r.scaling_meaningful:
            by_case.setdefault(r.name, []).append(r)
    known = {c.name: c for c in BENCH_CASES}

    failures = 0
    print("scaling gate: per-item cost at the large scale / at the small scale")
    print("-" * 64)
    for name, group in by_case.items():
        if len(group) != 2:
            continue
        small, large = sorted(group, key=lambda r: r.scale)
        if not small.per_item_us or not large.per_item_us:
            continue
        ratio = large.per_item_us / small.per_item_us

        if ratio > MAX_SCALING_RATIO:
            # A busy host inflates one scale and not the other, which looks
            # exactly like superlinear growth. Measured: is_url_blocked reported
            # 2.34x while the machine sat at load average 10.9, and flat in every
            # run once it was idle. A real regression reproduces, so re-measure
            # before failing rather than trusting one sample of a ratio.
            case = known.get(name)
            if case is not None:
                again = sorted(
                    (measure(case, scale, repeats=REPEATS) for scale in case.scales),
                    key=lambda r: r.scale,
                )
                costs = [r.per_item_us for r in again]
                if len(again) == 2 and costs[0] and costs[1]:
                    ratio = costs[1] / costs[0]
                    print(
                        f"{name:<20} {small.scale:>7} -> {large.scale:<7} re-measured {ratio:.2f}x"
                    )

        if ratio > MAX_SCALING_RATIO:
            print(f"{name:<20} {small.scale:>7} -> {large.scale:<7} SUPERLINEAR {ratio:.2f}x")
            failures += 1
        else:
            print(f"{name:<20} {small.scale:>7} -> {large.scale:<7} flat ({ratio:.2f}x)")

    print()
    if failures:
        print(f"{failures} case(s) became superlinear")
    else:
        print("no case became superlinear")
    return 1 if failures else 0


def compare(before: list[dict], after: list[dict]) -> int:
    """
    Compare two recorded runs. Returns a process exit code.

    Non-zero when a case got slower by more than :data:`MAX_SLOWDOWN`, or when
    a case disappeared. Informational rather than enforced by CI, for the
    reasons in :func:`check_scaling`.
    """
    before_by_key = {(d["name"], d["scale"]): d for d in before}
    after_by_key = {(d["name"], d["scale"]): d for d in after}

    print("normalised cost per item (machine-independent)")
    print(f"{'case':<20} {'scale':>8} {'before':>11} {'after':>11} {'change':>10}")
    print("-" * 64)

    failures = 0
    for key in sorted(before_by_key, key=lambda k: (k[0], k[1])):
        name, scale = key
        old = before_by_key[key]
        if key not in after_by_key:
            was = old.get("normalised") or old["seconds"]
            print(f"{name:<20} {scale:>8} {was:>11.4f} {'MISSING':>10} {'--':>10}")
            failures += 1
            continue
        new = after_by_key[key]
        # Prefer the machine-normalised figure. Raw seconds are only a fallback
        # for a baseline recorded before normalisation existed.
        old_n = old.get("normalised")
        new_n = new.get("normalised")
        if old_n and new_n:
            old_s, new_s = old_n, new_n
        else:
            old_s, new_s = old["seconds"], new["seconds"]
        change = (new_s / old_s) if old_s else float("inf")
        marker = ""
        if change > MAX_SLOWDOWN:
            marker = "  <-- SLOWER"
            failures += 1
        elif change < 1 / MAX_SLOWDOWN:
            marker = "  <-- faster"
        print(f"{name:<20} {scale:>8} {old_s:>9.3f}s {new_s:>9.3f}s {change:>9.2f}x{marker}")

    print()
    if failures:
        print(f"{failures} case(s) regressed beyond {MAX_SLOWDOWN}x")
    else:
        print(f"no regression beyond {MAX_SLOWDOWN}x")
    return 1 if failures else 0


def _to_json(results: list[Result]) -> list[dict[str, Any]]:
    return [
        {
            "name": r.name,
            "scale": r.scale,
            "seconds": r.seconds,
            "items": r.items,
            "scaling_meaningful": r.scaling_meaningful,
            "normalised": r.normalised_us,
            "median": statistics.median(r.samples),
        }
        for r in results
    ]


def _load(path: Path) -> list[dict[str, Any]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = data["results"] if isinstance(data, dict) else data
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Measure protor's hot paths and report per-item cost.",
    )
    parser.add_argument("--json", type=Path, help="write results to this file")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--case", action="append", help="only run these cases")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument(
        "--check",
        type=Path,
        help="baseline JSON to compare against (informational; not machine-stable)",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="enforce the scaling gate; exits non-zero if a case became superlinear",
    )
    args = parser.parse_args(argv)

    if args.compare:
        before, after = _load(args.compare[0]), _load(args.compare[1])
        return compare(before, after)

    cases = BENCH_CASES
    if args.case:
        wanted = set(args.case)
        cases = tuple(c for c in cases if c.name in wanted)
        if not cases:
            print(f"no case matched {sorted(wanted)}", file=sys.stderr)
            return 2

    results = run_cases(cases, repeats=args.repeats)
    report(results)

    payload = _to_json(results)
    if args.json:
        args.json.write_text(json.dumps({"results": payload}, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    status = 0
    if args.gate:
        status |= check_scaling(results)
    if args.check:
        status |= compare(_load(args.check), payload)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
