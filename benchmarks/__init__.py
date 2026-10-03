"""
Benchmarks for protor's hot paths.

These are not the test suite. Tests assert behaviour; these measure cost, so a
regression shows up as a number that moved rather than a failure someone has to
notice. Run them directly::

    python -m benchmarks

Or compare two revisions::

    git stash
    python -m benchmarks --json > before.json
    git stash pop
    python -m benchmarks --json > after.json
    python -m benchmarks --compare before.json after.json

The runner also enforces :data:`MAX_SLOWDOWN`, so ``--check`` makes this usable
as a gate: it exits non-zero when a case is measurably slower than its recorded
baseline, which is what CI runs.
"""
