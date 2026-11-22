"""Sandbox isolation audit as a Harbor task (hard gate before any sweep).

The oracle agent copies solution/solve.sh into the sandbox and runs it DURING THE AGENT PHASE, so
the checks see exactly what a model's code would see. The verifier turns the result into a reward:
correct=1 only if every REQUIRED check passes.

Required:  no /orcd (lab storage, HF cache, task dirs with gold labels), no host $HOME, no host /tmp, no /tests
           during the agent phase, proxied HTTP blocked, vLLM rejects key-less requests, gateway admin
           API rejects key-less requests, gateway reachable.
Reported (informational): raw-socket internet egress. Singularity shares the host network, so code
           that deliberately bypasses the proxy can still reach the internet; leak_scan.py flags it.

    python -m dnaharness.adapters.audit      # -> <datasets>/audit/audit/
"""
from __future__ import annotations

import json
from pathlib import Path

from dnaharness import config
from dnaharness.adapters.common import reset_dir, sif_path, write_task_toml, write_verifier

CHECKS_PY = r'''
import json, os, socket, urllib.request, urllib.error
R = {}
def http(url, proxy=True, headers=None):
    opener = urllib.request.build_opener() if proxy else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with opener.open(req, timeout=6) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return "ERR:" + type(e).__name__
R["no_orcd"] = not os.path.exists("/orcd")
R["no_nfs"] = not os.path.exists("/nfs")
R["no_host_home"] = not any(os.path.exists(p) for p in ["/home/__USER__", "/orcd/home"])
R["no_hf_cache"] = not os.path.exists("__HF_HOME__")
R["no_tests_during_agent"] = not os.path.exists("/tests")
# the launcher stages Harbor task copies (incl. tests/) under host /tmp/dnah-<jobid>; the sandbox /tmp
# must be its own tmpfs, never the host's
R["no_host_tmp"] = not any(n.startswith(("dnah-", "claude-", "apptainer-")) for n in os.listdir("/tmp"))
R["proxied_http_blocked"] = str(http("https://huggingface.co")).startswith("ERR")
# every vLLM port that is listening must reject key-less requests, and at least one must be listening
_v = {pt: http(f"http://127.0.0.1:{pt}/v1/models") for pt in (__VLLM_PORTS__)}
_up = {pt: r for pt, r in _v.items() if not str(r).startswith("ERR")}
R["vllm_requires_key"] = bool(_up) and all(r in (401, 403) for r in _up.values())
R["info_vllm_ports"] = {str(k): v for k, v in _v.items()}
R["gateway_admin_requires_key"] = http("http://127.0.0.1:__GW_PORT__/admin/trials/x") == 401
R["gateway_reachable"] = http("http://127.0.0.1:__GW_PORT__/health") == 200
R["info_raw_internet_egress"] = http("https://huggingface.co", proxy=False)
R["info_writable_tmp_mb"] = os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize // 2**20
os.makedirs("/logs/agent", exist_ok=True)
json.dump(R, open("/logs/agent/audit.json", "w"), indent=2)
print(json.dumps(R, indent=2))
'''

VERIFY_PY = r'''
import json, os
R = json.load(open("/logs/agent/audit.json")) if os.path.exists("/logs/agent/audit.json") else {}
req = [k for k in R if not k.startswith("info_")]
ok = bool(R) and all(R[k] is True for k in req)
os.makedirs("/logs/verifier", exist_ok=True)
json.dump({"correct": int(ok), "valid_output": int(bool(R))}, open("/logs/verifier/reward.json", "w"))
json.dump({"checks": R, "failed": [k for k in req if R.get(k) is not True]}, open("/logs/verifier/prediction.json", "w"))
'''


def _all_model_ports(p: dict) -> list:
    """Every port a vLLM server of this allocation can listen on: the standard ports AND the ports of
    locally registered checkpoints (Stage-3 SFT models), all with this allocation's port offset."""
    ports = set(p["vllm_ports"].values())
    for name, cfg in config.models().get("models", {}).items():
        if cfg.get("port"):
            ports.add(config.model_port(name))
    return sorted(ports)


def generate(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or config.path("datasets") / "audit"
    tdir = out_dir / "audit" / "audit-sandbox"
    reset_dir(tdir, overwrite=True)
    p = config.paths()
    import getpass
    checks = (CHECKS_PY.replace("__USER__", getpass.getuser()).replace("__HF_HOME__", str(p["hf_home"]))
              .replace("__VLLM_PORTS__", ", ".join(str(v) for v in _all_model_ports(p)) + ",").replace("__GW_PORT__", str(p["gateway_port"])))
    (tdir / "environment").mkdir()
    (tdir / "environment" / "inputs.json").write_text(json.dumps({"question": "audit"}))
    (tdir / "instruction.md").write_text("Sandbox isolation audit (run with the oracle agent).\n")
    write_task_toml(tdir, name="audit-sandbox", description="sandbox isolation audit",
                    metadata={"benchmark": "audit"}, sif=sif_path(), agent_timeout_sec=300.0)
    write_verifier(tdir, VERIFY_PY, {"note": "audit"})
    sol = tdir / "solution"
    sol.mkdir()
    (sol / "solve.sh").write_text("#!/bin/bash\n/usr/bin/python3 - <<'PYEOF'\n" + checks + "\nPYEOF\n")
    (sol / "solve.sh").chmod(0o755)
    return tdir


if __name__ == "__main__":
    print(generate())
