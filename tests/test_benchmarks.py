"""
Tests for the benchmark harness itself.

A benchmark that silently stops measuring is worse than no benchmark: it reads
like a passing check. These pin the properties the gate depends on — that
fixtures grow with the scale knob, that scaling ratios are computed from
per-item costs, and that a real regression is actually reported as one.
"""

from __future__ import annotations

import json

import pytest
from benchmarks.cases import (
    BENCH_CASES,
    CALIBRATION_CASE,
    BenchCase,
    artificial_page,
    site_batch,
)
from benchmarks.runner import (
    MAX_SLOWDOWN,
    REPEATS,
    Result,
    check_scaling,
    compare,
    measure,
    run_cases,
)


class TestFixtures:
    @pytest.mark.parametrize("size", [10, 100, 400])
    def test_page_grows_with_the_scale_knob(self, size):
        """A fixture that ignored *size* would report flat scaling for any code."""
        page = artificial_page(size)
        assert page.count("<section") == size
        assert len(page) > 1000

    def test_page_is_deterministic_for_a_given_size(self):
        """Two runs must process identical input, or timings are not comparable."""
        assert artificial_page(50) == artificial_page(50)

    def test_prose_varies_between_sections(self):
        """Prose is drawn from a pool, so section bodies differ within a page.

        Only the section paragraphs are compared: the blockquote text is fixed
        by design, so it repeats by construction.
        """
        import re

        page = artificial_page(30)
        bodies = re.findall(r"<section[^>]*><h2>Section \d+</h2><p>(.*?)</p>", page)
        assert len(bodies) == 30
        assert len(set(bodies)) == 30, "two sections rendered identically"

    def test_page_carries_the_noise_the_filter_must_strip(self):
        page = artificial_page(20, noise=True)
        assert "<nav" in page and 'class="widget' in page and "<script" in page

    def test_noise_can_be_switched_off(self):
        assert "<nav" not in artificial_page(20, noise=False)

    def test_nested_mode_adds_depth(self):
        shallow = artificial_page(30, nested=False)
        deep = artificial_page(30, nested=True)
        assert deep.count("<div>") > shallow.count("<div>")

    @pytest.mark.parametrize("count", [1, 25, 200])
    def test_site_batch_size_and_shape(self, count):
        batch = site_batch(count)
        assert len(batch) == count
        assert all("text_content" in s and "domain" in s for s in batch)


class TestCasesAreWellFormed:
    def test_every_case_has_two_scales_in_ascending_order(self):
        for case in BENCH_CASES:
            assert len(case.scales) == 2, case.name
            assert case.scales[0] < case.scales[1], f"{case.name} scales are not ascending"

    def test_case_names_are_unique(self):
        names = [c.name for c in BENCH_CASES]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("case", BENCH_CASES, ids=lambda c: c.name)
    def test_each_case_runs_and_reports_items(self, case):
        """Every case must do real work; a case returning nothing measures overhead."""
        result = measure(case, case.scales[0], repeats=1)
        assert result.items, f"{case.name} reported no work items"
        assert result.seconds > 0, f"{case.name} took no measurable time"


class TestResultMaths:
    def test_per_item_cost_is_seconds_over_items(self):
        r = Result(name="x", scale=10, seconds=2.0, items=1000)
        assert r.per_item_us == pytest.approx(2000.0)

    def test_per_item_cost_is_none_without_items(self):
        assert Result(name="x", scale=1, seconds=1.0).per_item_us is None

    def test_seconds_formatting_scales_with_magnitude(self):
        assert Result(name="x", scale=1, seconds=2.5).seconds_formatted.endswith("s")
        assert Result(name="x", scale=1, seconds=0.05).seconds_formatted.endswith("ms")
        assert Result(name="x", scale=1, seconds=0.0002).seconds_formatted.endswith("us")

    def test_measure_takes_the_best_of_repeats(self, tmp_path):
        """Noise can only add time, so the minimum is the honest figure."""
        calls: list[int] = []

        def work(scale: int) -> int:
            calls.append(scale)
            return scale

        case = BenchCase("noop", work, (5, 10))
        result = measure(case, 5, repeats=4)
        assert len(calls) == 4
        assert result.seconds == min(result.samples)


class TestRegressionDetection:
    @staticmethod
    def _record(name: str, scale: int, seconds: float) -> dict:
        return {
            "name": name,
            "scale": scale,
            "seconds": seconds,
            "items": 100,
            "scaling_meaningful": True,
            "normalised": seconds,
            "median": seconds,
        }

    def test_identical_runs_report_no_regression(self, capsys):
        rec = self._record("parse_html", 200, 0.1)
        assert compare([rec], [dict(rec)]) == 0
        assert "no regression" in capsys.readouterr().out

    def test_large_slowdown_is_reported_and_fails(self, capsys):
        before = [self._record("parse_html", 200, 0.1)]
        after = [self._record("parse_html", 200, 0.1 * (MAX_SLOWDOWN + 0.5))]
        assert compare(before, after) == 1
        out = capsys.readouterr().out
        assert "SLOWER" in out
        assert "regressed" in out

    def test_small_variation_is_tolerated(self):
        """A busy runner adds noise; the gate must not trip on it."""
        before = [self._record("parse_html", 200, 0.1)]
        after = [self._record("parse_html", 200, 0.1 * 1.2)]
        assert compare(before, after) == 0

    def test_large_speedup_is_not_a_failure(self, capsys):
        before = [self._record("parse_html", 200, 1.0)]
        after = [self._record("parse_html", 200, 0.1)]
        assert compare(before, after) == 0
        assert "faster" in capsys.readouterr().out

    def test_a_disappeared_case_fails(self, capsys):
        before = [self._record("parse_html", 200, 0.1)]
        assert compare(before, []) == 1
        assert "MISSING" in capsys.readouterr().out

    def test_gate_uses_normalised_figures_when_present(self, capsys):
        """
        Raw seconds from a different machine are meaningless. A baseline
        recorded locally made every case look 1.6-1.7x slower on CI's runner
        with no code change, so the comparison must divide out machine speed.
        """
        rec = self._record("parse_html", 200, 0.1)
        before = [{**rec, "normalised": 2.0}]
        after = [{**rec, "normalised": 2.1}]
        assert compare(before, after) == 0, "5x slower wall clock must not fail"

    def test_normalised_regression_still_fails(self, capsys):
        rec = self._record("parse_html", 200, 0.1)
        before = [{**rec, "normalised": 2.0}]
        after = [{**rec, "normalised": 4.0}]
        assert compare(before, after) == 1
        assert "SLOWER" in capsys.readouterr().out

    def test_names_are_matched_on_both_name_and_scale(self):
        """Two scales of one case are separate measurements, not interchangeable."""
        before = [self._record("parse_html", 200, 0.1)]
        after = [self._record("parse_html", 800, 0.4)]
        assert compare(before, after) == 1  # the 200 row is missing


#: Cases whose work item count grows with the input page.
_PAGE_CASES = ("parse_html", "clean_soup", "extract_text")


class TestScalesAreBelowSaturation:
    """
    The scales must stay under the renderer's character budgets.

    This is not a nicety. With both scales saturated, a budgeted function stops
    walking at the cap and does the same fixed work regardless of page size, so
    its per-item ratio is flat no matter what the code does. Measured directly:
    a full-document rescan injected into `clean_soup` passed the gate at
    (200, 800) and was caught at the same scales only after the page cases were
    resized -- the injected bug in `_process_element` went undetected for the
    same reason.
    """

    @pytest.mark.parametrize(
        "name,scale",
        [("parse_html", 20), ("parse_html", 80), ("extract_text", 10), ("extract_text", 30)],
    )
    def test_page_scales_are_not_truncated(self, name, scale):
        from benchmarks.cases import BENCH_CASES, artificial_page
        from bs4 import BeautifulSoup

        from protor.parser import _extract_text, parse_html

        case = next(c for c in BENCH_CASES if c.name == name)
        assert scale in case.scales, f"{name} no longer measures {scale}"

        if name == "parse_html":
            _, page = parse_html(artificial_page(scale), "https://example.com/")
            assert not page.markdown_content.endswith("[truncated]"), (
                f"{scale} blocks saturate the markdown budget"
            )
        else:
            soup = BeautifulSoup(artificial_page(scale), "lxml")
            assert not _extract_text(soup, max_chars=10_000).endswith("[truncated]"), (
                f"{scale} blocks saturate the text budget"
            )

    def test_the_larger_scale_does_real_extra_work(self):
        """Guards against a scale change that quietly makes both sides equal."""
        from benchmarks.cases import artificial_page

        from protor.parser import parse_html

        _, small = parse_html(artificial_page(20), "https://example.com/")
        _, large = parse_html(artificial_page(80), "https://example.com/")
        assert len(large.markdown_content) > len(small.markdown_content) * 2


class TestScalingGate:
    """
    `--gate` is what CI enforces, so its behaviour is pinned here rather than
    only in the workflow file.
    """

    @staticmethod
    def _r(name: str, scale: int, seconds: float) -> Result:
        """
        A result for *scale* items that took *seconds* total.

        ``items`` has to equal ``scale`` for a per-item ratio to mean what it
        says: the ratio asks whether the cost of handling one item changes when
        there are more of them.
        """
        return Result(name=name, scale=scale, seconds=seconds, items=scale)

    def test_flat_scaling_passes(self, capsys):
        # 4x the items in 4.1x the time: per-item cost unchanged.
        results = [self._r("parse_html", 200, 1.0), self._r("parse_html", 800, 4.1)]
        assert check_scaling(results) == 0
        assert "flat" in capsys.readouterr().out

    def test_superlinear_scaling_fails(self, capsys):
        """
        4x the input for 16x the time is quadratic — what a re-scan looks like.

        Uses a name that is not a real case, so the gate has nothing to
        re-measure and must decide on the evidence given.
        """
        results = [self._r("hypothetical_case", 200, 1.0), self._r("hypothetical_case", 800, 16.0)]
        assert check_scaling(results) == 1
        assert "SUPERLINEAR" in capsys.readouterr().out

    def test_a_known_case_is_re_measured_before_failing(self, capsys):
        """
        A single bad sample on a real case is treated as contention, not proof.

        Measured: `is_url_blocked` reported 2.34x while the host was at load
        average 10.9, and was flat in every run once idle. Re-measuring costs a
        few seconds only when there is a signal, and is the difference between a
        gate people trust and one they learn to ignore.
        """
        results = [self._r("clean_soup", 200, 1.0), self._r("clean_soup", 800, 6.0)]
        assert check_scaling(results) == 0
        assert "re-measured" in capsys.readouterr().out

    def test_a_confirmed_regression_still_fails(self, monkeypatch, capsys):
        """
        The re-measure must not launder a regression that reproduces.

        Verified end to end by hand as well: injecting a full-document rescan
        into `clean_soup` made the gate report SUPERLINEAR 5.3x and exit 1,
        while the same command on the real code reported flat and exited 0.
        Here the re-measure is stubbed to confirm, so the decision path is
        exercised without a slow real measurement.
        """
        import benchmarks.runner as runner

        def confirming(case, scale, *, repeats=1, machine_us=0.0):
            # Quadratic on re-measure too: 4x items costs 16x the time.
            seconds = (scale / 200) ** 2
            return Result(name=case.name, scale=scale, seconds=seconds, items=scale)

        monkeypatch.setattr(runner, "measure", confirming)
        results = [self._r("clean_soup", 200, 1.0), self._r("clean_soup", 800, 6.0)]
        assert runner.check_scaling(results) == 1
        assert "SUPERLINEAR" in capsys.readouterr().out

    def test_sublinear_is_not_a_failure(self, capsys):
        """Per-item cost falling with scale is fine; batching can cause it."""
        results = [self._r("http_cache_put", 100, 1.0), self._r("http_cache_put", 400, 2.0)]
        assert check_scaling(results) == 0

    def test_cases_without_meaningful_items_are_skipped(self, capsys):
        results = [Result(name="x", scale=1, seconds=1.0), Result(name="x", scale=2, seconds=9.0)]
        assert check_scaling(results) == 0

    @pytest.mark.slow
    def test_real_suite_is_flat(self, capsys):
        """
        The committed suite must itself be linear, or the gate is noise.

        Marked slow and given the default repeat count: it is a real measurement
        of every case, and at two repeats the ratio is a difference of two noisy
        samples, which fails intermittently. Excluded from the default run by
        the ``slow`` marker below; CI's benchmark job exercises the same code
        path directly via ``--gate``.
        """
        assert check_scaling(run_cases(repeats=REPEATS)) == 0


class TestMachineNormalisation:
    """The calibration case exists to make timings comparable across hosts."""

    def test_calibration_case_is_present_and_runs(self):
        assert CALIBRATION_CASE in BENCH_CASES
        result = measure(CALIBRATION_CASE, CALIBRATION_CASE.scales[0], repeats=1)
        assert result.items == CALIBRATION_CASE.scales[0]

    def test_every_result_carries_the_machine_reference(self):
        results = run_cases(repeats=1)
        assert all(r.machine_us > 0 for r in results), "no machine reference was recorded"

    def test_normalised_figure_divides_out_machine_speed(self):
        slow = Result(name="x", scale=1, seconds=2.0, items=1000, machine_us=2.0)
        fast = Result(name="x", scale=1, seconds=1.0, items=1000, machine_us=1.0)
        # Twice the wall clock on twice the reference speed is the same work.
        assert slow.normalised_us == pytest.approx(fast.normalised_us)

    def test_normalised_is_none_without_a_reference(self):
        assert Result(name="x", scale=1, seconds=1.0, items=10).normalised_us is None


class TestScalingIsMeaningful:
    def test_cases_without_input_scaled_items_are_excluded(self):
        """A fixed item count makes a per-item ratio meaningless, not flat."""
        excluded = {c.name for c in BENCH_CASES if not c.scaling_meaningful}
        assert "http_cache_put" in excluded
        assert "prepare_context" in excluded

    def test_page_cases_report_input_scaled_items(self):
        """These process a growing page, so their per-item ratio is meaningful."""
        page_cases = {c.name for c in BENCH_CASES if c.name in _PAGE_CASES}
        assert page_cases == {"parse_html", "clean_soup", "extract_text"}
        assert all(c.scaling_meaningful for c in BENCH_CASES if c.name in _PAGE_CASES)


class TestBaselineFile:
    def test_recorded_baseline_is_usable_and_complete(self):
        """CI gates on this file; a stale or truncated one breaks the gate."""
        from pathlib import Path

        baseline = Path(__file__).resolve().parent.parent / "benchmarks" / "baseline.json"
        data = json.loads(baseline.read_text(encoding="utf-8"))
        records = data["results"]

        for r in records:
            assert r.get("normalised") is not None, (
                f"{r['name']} has no normalised figure; re-record the baseline"
            )

        recorded = {(r["name"], r["scale"]) for r in records}
        expected = {(c.name, s) for c in BENCH_CASES for s in c.scales}
        assert recorded == expected, "baseline.json does not match the current cases"

        for r in records:
            assert r["seconds"] > 0, f"{r['name']} has a non-positive baseline"
            assert r["items"], f"{r['name']} has no work-item count"
