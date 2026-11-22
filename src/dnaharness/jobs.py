"""Experiment specs (configs/jobs/*.yaml) -> Harbor job config + serving spec.

An experiment spec is compact and declarative:

    name: exp_a_shared
    serve: {models: [qwen3-1.7b, bioreason-1.7b], policy: sampled}
    dataset_root: kegg-dx                 # under <datasets>/
    datasets: [raw-wt, raw-no_dna, raw-scramble]
    n_attempts: 3
    n_concurrent_trials: 24
    cells:                                # arm x model (arms from configs/harness/arms.yaml)
      - {arm: direct, model: qwen3-1.7b}
      - {arm: rlm, model: bioreason-1.7b, kwargs: {max_iterations: 12}}

`render` writes <jobs>/<name>/harbor_config.yaml (feed to `harbor run -c`) and serve.json (read by
slurm/serve_and_run.sbatch to start vLLM + the gateway).

If a spec sets `purpose`, every dataset must belong to that class in configs/data_access.yaml.
`purpose: train` and `purpose: dev` cannot name an eval dataset. The check runs after the spec is
loaded and before any job directory is written. Specs that omit `purpose` are rendered exactly as
before.

    python -m dnaharness.jobs render configs/jobs/exp_a_shared.yaml [--chunk i/n]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from dnaharness import config


def _arms() -> dict:
    return yaml.safe_load((config.CONFIG_DIR / "harness" / "arms.yaml").read_text())


def agent_entry(cell: dict, arms: dict, gateway_url: str) -> Dict[str, Any]:
    arm = arms["arms"][cell["arm"]]
    d = arms["defaults"]
    if arm.get("builtin_plumbing"):                      # oracle / nop: no model, no gateway
        return {"name": arm["name"], "kwargs": {}}
    model = cell["model"]
    kwargs = copy.deepcopy(arm.get("kwargs", {}))
    if cell["arm"] == "mini_swe":
        cfg = yaml.safe_load((config.REPO_ROOT / kwargs.pop("config_file")).read_text())
        cfg["agent"]["step_limit"] = d["max_iterations"]
        # The endpoint goes in the native config: with model_class litellm_textbased Harbor refuses an
        # agent-level base_url ("doesn't support --base-url overrides") and reads model_kwargs.api_base
        # instead. Harbor still adds model_kwargs.extra_headers.X-Session-ID (gateway budget key).
        mk = cfg.setdefault("model", {}).setdefault("model_kwargs", {})
        mk["api_base"] = f"{gateway_url}/v1"
        # No api_key here: Harbor scrubs the agent's env values (OPENAI_API_KEY=sandbox) from every saved
        # trial file, so a copy of that value in the config is persisted as "[REDACTED]" and the stored
        # trial configs never match the job plan again (`harbor jobs resume` fails). LiteLLM reads the
        # key from OPENAI_API_KEY in the agent env instead.
        kwargs["config"] = cfg
        entry = {"import_path": arm["import_path"], "model_name": f"openai/{model}",
                 "env": {"OPENAI_API_KEY": "sandbox", "MSWEA_API_KEY": "sandbox"},
                 "override_timeout_sec": d["agent_timeout_sec"]}
    else:
        for k in ("max_generated_tokens", "max_child_calls"):
            kwargs.setdefault(k, d[k])
        if arm["import_path"].endswith(":RlmAgent"):
            kwargs.setdefault("max_iterations", d["max_iterations"])
            kwargs.setdefault("agent_timeout_sec", d["agent_timeout_sec"])
        kwargs["gateway_url"] = gateway_url
        entry = {"import_path": arm["import_path"], "model_name": f"gateway/{model}"}
    kwargs.update(cell.get("kwargs", {}))
    root = str(config.paths()["root"])
    entry["kwargs"] = {k: (v.replace("${root}", root) if isinstance(v, str) else v) for k, v in kwargs.items()}
    return entry


def _dataset_family(spec: dict, ds: str) -> str:
    """Dataset name one spec entry resolves to, using the same rule as `render`.

    "name" -> <dataset_root>/name (dataset_root defaults to kegg-dx).
    "group/name" -> <datasets>/group/name. The path is normalized so "dev/../kegg-dx/..."
    is the dataset the job would actually open, not the first path component.
    """
    if not isinstance(ds, str) or not ds:
        raise SystemExit(f"dataset entry must be a non-empty string, got {ds!r}")
    datasets_root = config.path("datasets")
    ds_root = datasets_root / spec.get("dataset_root", "kegg-dx")
    # "name" -> <dataset_root>/name ; "group/name" -> <datasets>/group/name
    dpath = (datasets_root / ds) if "/" in ds else (ds_root / ds)
    try:
        rel = dpath.resolve().relative_to(datasets_root.resolve())
    except ValueError:
        raise SystemExit(f"dataset {ds!r} resolves outside the datasets root ({dpath})") from None
    if not rel.parts or rel.parts[0] in (".", ".."):
        raise SystemExit(f"dataset {ds!r} does not name a dataset under the datasets root")
    return rel.parts[0]


def _check_data_access(spec: dict) -> None:
    """If `purpose` is set, every dataset must be in that class. No-op otherwise.

    Runs before a job directory is written. `purpose: train` and `purpose: dev` cannot name an
    eval dataset. Missing `purpose` returns without reading the permission list.
    """
    if not isinstance(spec, dict) or "purpose" not in spec or spec.get("purpose") is None:
        return
    purpose = spec.get("purpose")
    access = yaml.safe_load((config.CONFIG_DIR / "data_access.yaml").read_text())
    if not isinstance(access, dict):
        raise SystemExit("configs/data_access.yaml must map each purpose to a list of dataset names")
    classes: Dict[str, List[str]] = {}
    for key, value in access.items():
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise SystemExit(f"configs/data_access.yaml class {key!r} must be a list of dataset names")
        classes[str(key)] = list(value)
    if not isinstance(purpose, str) or purpose not in classes:
        known = ", ".join(sorted(classes))
        raise SystemExit(f"unknown purpose {purpose!r}; expected one of: {known}")
    datasets = spec.get("datasets")
    if not isinstance(datasets, list):
        raise SystemExit(f"purpose {purpose!r} requires a datasets list")
    allowed = set(classes[purpose])
    eval_names = set(classes.get("eval", []))
    for ds in datasets:
        family = _dataset_family(spec, ds)
        if purpose in ("train", "dev") and family in eval_names:
            raise SystemExit(f"purpose {purpose!r} cannot name eval dataset {family!r} (from {ds!r})")
        if family not in allowed:
            raise SystemExit(
                f"dataset {family!r} (from {ds!r}) is not allowed for purpose {purpose!r}; "
                f"allowed: {', '.join(classes[purpose])}"
            )


def render(spec_path: str, chunk: str | None = None, jobs_dir: str | None = None) -> Path:
    spec = yaml.safe_load(Path(spec_path).read_text())
    # Opt-in, and before out.mkdir. Missing purpose returns immediately so the writes below
    # stay byte-for-byte the same (audit.yaml and the other baseline specs have no purpose).
    _check_data_access(spec)
    arms = _arms()
    gateway_url = f"http://127.0.0.1:{config.paths()['gateway_port']}"
    name = spec["name"] + (f"-c{chunk.replace('/', 'of')}" if chunk else "")
    root = Path(jobs_dir) if jobs_dir else config.path("jobs")
    out = root / name
    out.mkdir(parents=True, exist_ok=True)

    ds_root = config.path("datasets") / spec.get("dataset_root", "kegg-dx")
    datasets: List[dict] = []
    meta_dirs = set(spec["serve"].get("extra_meta_dirs", []))
    for ds in spec["datasets"]:
        # "name" -> <dataset_root>/name ; "group/name" -> <datasets>/group/name
        dpath = (config.path("datasets") / ds) if "/" in ds else (ds_root / ds)
        if (dpath.parent / "_meta").is_dir():
            meta_dirs.add(str(dpath.parent / "_meta"))
        entry: Dict[str, Any] = {"path": str(dpath)}
        if chunk:
            i, n = (int(x) for x in chunk.split("/"))
            names = sorted(p.name for p in dpath.iterdir() if (p / "task.toml").exists())
            entry["task_names"] = names[i::n]
        datasets.append(entry)

    harbor_cfg = {
        "job_name": name,
        "jobs_dir": str(root),
        "n_attempts": int(spec.get("n_attempts", 1)),
        "n_concurrent_trials": int(spec.get("n_concurrent_trials", 16)),
        "timeout_multiplier": float(spec.get("timeout_multiplier", 1.0)),
        "retry": {"max_retries": int(spec.get("max_retries", 1))},
        "environment": {
            "type": "singularity",
            "delete": True,
            "kwargs": {"singularity_image_cache_dir": str(config.path("sif_dir"))},
        },
        "agents": [agent_entry(c, arms, gateway_url) for c in spec["cells"]],
        "datasets": datasets,
    }
    serve = {"name": name, "models": spec["serve"]["models"], "policy": spec["serve"].get("policy", "sampled"),
             "meta_dirs": sorted(meta_dirs),
             "log_messages": bool(spec["serve"].get("log_messages", False)),
             "gpus": spec["serve"].get("gpus", 1), "placement": spec["serve"].get("placement", {})}
    (out / "harbor_config.yaml").write_text(yaml.safe_dump(harbor_cfg, sort_keys=False))
    (out / "serve.json").write_text(json.dumps(serve, indent=2))
    (out / "spec.yaml").write_text(Path(spec_path).read_text())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("spec")
    r.add_argument("--chunk", default=None, help="i/n: only every n-th task starting at i (resumable chunks)")
    r.add_argument("--jobs-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "render":
        print(render(a.spec, a.chunk, a.jobs_dir))


if __name__ == "__main__":
    main()
