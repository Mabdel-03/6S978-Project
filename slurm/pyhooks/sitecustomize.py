"""Loaded by Python at startup when this directory is on PYTHONPATH (serve_and_run.sbatch sets it for
the `harbor` CLI only). Installs the sandbox reaper: Harbor 0.27's Singularity environment leaks the
whole sandbox process tree after every trial -- see dnaharness/harbor_ext/sandbox_reaper.py."""
try:
    from dnaharness.harbor_ext import sandbox_reaper as _r
    _r.install()
except Exception as _e:  # never break interpreter startup
    import sys
    print(f"[dnah] sandbox reaper not installed: {_e}", file=sys.stderr)
