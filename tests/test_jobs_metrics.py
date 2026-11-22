import numpy as np
import pandas as pd
import pytest
import yaml

from dnaharness import config, jobs
from dnaharness.analysis import metrics as M


def _ensure_dataset_dirs(spec: dict) -> None:
    """Create empty dataset dirs so chunking / paths resolve (same rule as jobs.render)."""
    for ds in spec["datasets"]:
        p = (
            (config.path("datasets") / ds)
            if "/" in ds
            else config.path("datasets") / spec.get("dataset_root", "kegg-dx") / ds
        )
        p.mkdir(parents=True, exist_ok=True)


def _write_job_spec(tmp_path, *, name: str, purpose: str, datasets: list, dataset_root: str | None = None):
    """Minimal oracle job for access-check / render tests. Uses a temp jobs_dir via caller."""
    body = {
        "name": name,
        "purpose": purpose,
        "serve": {"models": ["qwen3-1.7b"], "policy": "sampled"},
        "datasets": datasets,
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "cells": [{"arm": "oracle"}],
    }
    if dataset_root is not None:
        body["dataset_root"] = dataset_root
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def _assert_harbor_job(out):
    pytest.importorskip("harbor")
    from harbor.models.job.config import JobConfig

    cfg = yaml.safe_load((out / "harbor_config.yaml").read_text())
    jc = JobConfig.model_validate(cfg)
    assert jc.environment.type.value == "singularity"
    for a in jc.agents:
        assert a.name or a.import_path
        if a.import_path and a.import_path.endswith(":MiniSweGatewayAgent"):
            # Harbor built-in mini-swe: endpoint lives in its native config (resolved at run time)
            assert a.kwargs["config"]["model"]["model_kwargs"]["api_base"].startswith("http://127.0.0.1:")
            assert a.kwargs["skip_install"] is True and not getattr(a, "base_url", None)
        elif a.import_path:
            assert a.kwargs["gateway_url"].startswith("http://127.0.0.1:")
    assert (out / "serve.json").exists()


@pytest.mark.parametrize("spec", sorted(p.name for p in (config.CONFIG_DIR / "jobs").glob("*.yaml")))
def test_job_specs_render_and_validate_with_harbor(spec, tmp_path):
    # create empty dataset dirs so chunking / paths resolve
    s = yaml.safe_load((config.CONFIG_DIR / "jobs" / spec).read_text())
    # audit.yaml (and other baseline specs) omit purpose so the access check is a no-op
    if spec == "audit.yaml":
        assert "purpose" not in s or s.get("purpose") is None
    _ensure_dataset_dirs(s)
    out = jobs.render(str(config.CONFIG_DIR / "jobs" / spec), jobs_dir=str(tmp_path))
    _assert_harbor_job(out)


def test_purpose_train_rejects_eval_dataset_before_job_dir(tmp_path):
    """purpose: train + kegg-dx/raw-wt must SystemExit before any job directory is written."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    spec_path = _write_job_spec(
        tmp_path,
        name="access_train_rejects_kegg",
        purpose="train",
        datasets=["kegg-dx/raw-wt"],
    )
    with pytest.raises(SystemExit) as ei:
        jobs.render(str(spec_path), jobs_dir=str(jobs_dir))
    msg = str(ei.value)
    assert "train" in msg and "kegg-dx" in msg
    assert "cannot name eval dataset" in msg or "not allowed" in msg
    assert not (jobs_dir / "access_train_rejects_kegg").exists()
    assert list(jobs_dir.iterdir()) == []


def test_purpose_eval_harness_pilot_renders(tmp_path):
    jobs_dir = tmp_path / "jobs"
    spec_path = _write_job_spec(
        tmp_path,
        name="access_eval_pilot",
        purpose="eval",
        datasets=["harness-pilot/pilot"],
    )
    s = yaml.safe_load(spec_path.read_text())
    _ensure_dataset_dirs(s)
    out = jobs.render(str(spec_path), jobs_dir=str(jobs_dir))
    assert out == jobs_dir / "access_eval_pilot"
    _assert_harbor_job(out)


def test_purpose_train_harness_train_renders(tmp_path):
    jobs_dir = tmp_path / "jobs"
    spec_path = _write_job_spec(
        tmp_path,
        name="access_train_harness",
        purpose="train",
        datasets=["harness-train/train"],
    )
    s = yaml.safe_load(spec_path.read_text())
    _ensure_dataset_dirs(s)
    out = jobs.render(str(spec_path), jobs_dir=str(jobs_dir))
    assert out == jobs_dir / "access_train_harness"
    _assert_harbor_job(out)


def test_rlm_and_coding_differ_only_in_arm():
    arms = jobs._arms()
    a = jobs.agent_entry({"arm": "rlm", "model": "m"}, arms, "http://127.0.0.1:8000")
    b = jobs.agent_entry({"arm": "coding_repl", "model": "m"}, arms, "http://127.0.0.1:8000")
    assert a["import_path"] == b["import_path"]
    diff = {k for k in set(a["kwargs"]) | set(b["kwargs"]) if a["kwargs"].get(k) != b["kwargs"].get(k)}
    assert diff == {"arm"}


def _toy():
    rows = []
    # one text group, 4 rows, 2 labels, 4 distinct pairs, 2 seeds
    spec = [("r1", "A", "p1"), ("r2", "A", "p2"), ("r3", "B", "p3"), ("r4", "B", "p4")]
    correct = {("r1", 0): 1, ("r1", 1): 1, ("r2", 0): 1, ("r2", 1): 0, ("r3", 0): 1, ("r3", 1): 1, ("r4", 0): 0, ("r4", 1): 0}
    for rid, lab, pid in spec:
        for s in (0, 1):
            rows.append({"row_id": rid, "label": lab, "pair_id": pid, "text_group_id": "g", "attempt": s,
                         "correct": bool(correct[(rid, s)]), "is_minority_label": lab == "B",
                         "predicted_label": lab if correct[(rid, s)] else "other"})
    return pd.DataFrame(rows)


def test_pair_accuracy_definition():
    d = _toy()
    # cross-label pairs: (r1,r3) (r1,r4) (r2,r3) (r2,r4); seed-avg both-correct: 1, 0, .5, 0
    assert M.pair_accuracy(d) == pytest.approx(np.mean([1, 0, .5, 0]))
    assert M.accuracy(d) == pytest.approx(np.mean([1, .5, 1, 0]))
    assert M.minority_accuracy(d) == pytest.approx(0.5)


def test_bootstrap_runs_and_brackets_point():
    d = pd.concat([_toy().assign(text_group_id=f"g{i}", row_id=lambda x, i=i: x.row_id + f"_{i}",
                                 pair_id=lambda x, i=i: x.pair_id + f"_{i}") for i in range(5)])
    p, lo, hi = M.bootstrap(d, M.accuracy, n=200)
    assert lo <= p <= hi
    p, lo, hi = M.paired_bootstrap(d, d, M.accuracy, n=100)
    assert p == 0 and lo == 0 and hi == 0
