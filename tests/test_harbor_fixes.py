"""Regression tests for the two Harbor 0.27 workarounds found while running the sweeps."""
import json
import subprocess
import time

from dnaharness.harbor_ext import resume_repair, sandbox_reaper as R


def test_reaper_kills_whole_tree_and_respects_start_time():
    p = subprocess.Popen(["bash", "-c", "sleep 300 & sleep 300 & wait"])
    try:
        for _ in range(50):
            t = R._table()
            if len(R._descendants(p.pid, t)) >= 2:
                break
            time.sleep(0.05)
        kids = R._descendants(p.pid, t)
        assert len(kids) >= 2
        st = t[p.pid][1]
        assert R._kill_tree(p.pid, st + 1, t) == 0          # wrong start time: a recycled PID is never touched
        assert all(k in R._table() for k in kids)
        assert R._kill_tree(p.pid, st, t) == len(kids) + 1  # the root and its descendants
        p.wait(timeout=5)
        time.sleep(0.2)
        alive = R._table()
        assert not any(k in alive and alive[k][2] == "sleep" for k in kids)
    finally:
        if p.poll() is None:
            p.kill()


def test_reaper_reaps_after_server_exit(monkeypatch):
    """poll_once records a server's 'starter' descendants while it lives and kills them once it is gone."""
    fake = subprocess.Popen(["bash", "-c", "exec -a starter sleep 300"])     # stands in for the starter
    try:
        t = R._table()
        monkeypatch.setattr(R, "_servers", {999999999: {"st": None, "starters": {fake.pid: t[fake.pid][1]}}})
        R.poll_once()                                       # server pid 999999999 does not exist -> reap
        fake.wait(timeout=5)
        assert fake.returncode is not None and not R._servers
    finally:
        if fake.poll() is None:
            fake.kill()


def test_resume_repair_restores_only_the_scrubbed_dummy_key(tmp_path):
    ip = "dnaharness.harbor_ext.agents:MiniSweGatewayAgent"
    mk = {"api_base": "http://127.0.0.1:8000/v1", "api_key": "sandbox"}
    agent = {"import_path": ip, "name": None, "kwargs": {"config": {"model": {"model_kwargs": mk}}}}
    (tmp_path / "config.json").write_text(json.dumps({"agents": [agent, {"import_path": "x:DirectAgent", "kwargs": {}}]}))
    red = json.loads(json.dumps(agent))
    red["kwargs"]["config"]["model"]["model_kwargs"]["api_key"] = "[REDACTED]"
    for name, a in [("t1", red), ("t2", {"import_path": "x:DirectAgent", "kwargs": {"note": "[REDACTED]"}})]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps({"agent": a, "trial_name": name}))
    assert resume_repair.repair(tmp_path) == 1
    t1 = json.loads((tmp_path / "t1" / "config.json").read_text())
    assert t1["agent"]["kwargs"]["config"]["model"]["model_kwargs"]["api_key"] == "sandbox"
    assert json.loads((tmp_path / "t2" / "config.json").read_text())["agent"]["kwargs"]["note"] == "[REDACTED]"
    assert resume_repair.repair(tmp_path) == 0                # idempotent


def test_resume_repair_restores_scrubbed_env_values(tmp_path):
    ip = "dnaharness.harbor_ext.agents:MiniSweGatewayAgent"
    agent = {"import_path": ip, "name": None, "env": {"OPENAI_API_KEY": "****"}, "kwargs": {"config": {"model": {"model_kwargs": {}}}}}
    (tmp_path / "config.json").write_text(json.dumps({"agents": [agent]}))
    t = json.loads(json.dumps(agent)); t["env"]["OPENAI_API_KEY"] = "[REDACTED]"
    (tmp_path / "t1").mkdir(); (tmp_path / "t1" / "config.json").write_text(json.dumps({"agent": t}))
    assert resume_repair.repair(tmp_path) == 1
    assert json.loads((tmp_path / "t1" / "config.json").read_text())["agent"]["env"]["OPENAI_API_KEY"] == "****"
