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
from benchmarks.cases import BENCH_CASES, BenchCase, artificial_page, site_batch
from benchmarks.runner import MAX_SLOWDOWN, Result, compare, measure


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

    def test_names_are_matched_on_both_name_and_scale(self):
        """Two scales of one case are separate measurements, not interchangeable."""
        before = [self._record("parse_html", 200, 0.1)]
        after = [self._record("parse_html", 800, 0.4)]
        assert compare(before, after) == 1  # the 200 row is missing


#: Cases whose work item count grows with the input page.
_PAGE_CASES = ("parse_html", "clean_soup", "extract_text")


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

        recorded = {(r["name"], r["scale"]) for r in records}
        expected = {(c.name, s) for c in BENCH_CASES for s in c.scales}
        assert recorded == expected, "baseline.json does not match the current cases"

        for r in records:
            assert r["seconds"] > 0, f"{r['name']} has a non-positive baseline"
            assert r["items"], f"{r['name']} has no work-item count"
