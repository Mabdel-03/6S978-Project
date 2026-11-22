"""The closed-form cluster bootstrap must equal the explicit resampling one, draw for draw."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dnaharness.analysis import metrics as M


def _synthetic(seed=0, n_att=3):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(5):
        for p in range(rng.integers(2, 6)):
            for r in range(rng.integers(1, 4)):
                lab = f"L{rng.integers(0, 3)}"
                for a in range(n_att):
                    rows.append({"row_id": f"g{g}p{p}r{r}", "text_group_id": f"g{g}", "pair_id": f"g{g}p{p}",
                                 "label": lab, "attempt": a, "correct": bool(rng.random() < 0.5),
                                 "is_minority_label": lab == "L2"})
    return pd.DataFrame(rows)


def _both(d, fn, n=60, paired_with=None):
    fast = M.paired_bootstrap(d, paired_with, fn, n=n) if paired_with is not None else M.bootstrap(d, fn, n=n)
    saved = M._FAST
    try:
        M._FAST = {}
        slow = M.paired_bootstrap(d, paired_with, fn, n=n) if paired_with is not None else M.bootstrap(d, fn, n=n)
    finally:
        M._FAST = saved
    return fast, slow


@pytest.mark.parametrize("fn", [M.accuracy, M.group_accuracy, M.minority_accuracy, M.pair_accuracy])
def test_fast_equals_slow_synthetic(fn):
    d = _synthetic()
    fast, slow = _both(d, fn)
    assert np.allclose(fast, slow, equal_nan=True), (fast, slow)
    fast, slow = _both(d, fn, paired_with=d.assign(correct=~d["correct"]))
    assert np.allclose(fast, slow, equal_nan=True), (fast, slow)


def test_incomplete_seed_grid_falls_back_for_pair_accuracy():
    d = _synthetic()
    d = d.drop(index=d.index[::7])                     # some rows miss some seeds
    fast, slow = _both(d, M.pair_accuracy, n=30)
    assert np.allclose(fast, slow, equal_nan=True)


REPRO = Path(__file__).resolve().parents[1] / "results" / "repro_trials.parquet"


@pytest.mark.skipif(not REPRO.exists(), reason="needs results/repro_trials.parquet (cluster run)")
@pytest.mark.parametrize("fn", [M.accuracy, M.pair_accuracy])
def test_fast_equals_slow_on_real_cohort(fn):
    d = pd.read_parquet(REPRO)
    wt = d[d.condition == "wt"]
    fast, slow = _both(wt, fn, n=25)
    assert np.allclose(fast, slow, equal_nan=True), (fast, slow)
