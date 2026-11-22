"""Reap Harbor Singularity sandboxes whose processes outlive their trial.

Harbor 0.27's SingularityEnvironment stops a sandbox by SIGTERM/SIGKILL-ing the `singularity exec`
CLI process and then `pkill -P <pid>` -- which runs after the CLI is dead, when its children have
been re-parented, so it matches nothing. With --fakeroot and a private PID namespace the container
init ignores the SIGTERM, so every finished trial left its whole sandbox running:

    starter ─┬─ appinit ─┬─ faked
             │           └─ fakeroot ── python3 /staging/_hbexec.py (+ any REPL code still running)
             ├─ fuse-overlayfs
             └─ squashfuse_ll

On a node shared by several allocations this accumulated thousands of sandboxes within hours (load
~8,600, free memory ~2 GB) and new sandboxes then timed out at start.

The fix, installed into the `harbor` process by slurm/pyhooks/sitecustomize.py:
  * every `singularity exec` server Harbor spawns is registered (the module's asyncio is shimmed);
  * a daemon thread polls every POLL_S seconds: while a server is alive it records its `starter`
    descendants (pid + start time); once the server is gone -- whichever Harbor path killed it
    (stop, failed start / port retry, start timeout, memory watchdog) -- it SIGKILLs each recorded
    starter's whole process tree;
  * at interpreter exit, everything still registered is reaped.
Start times are checked before every kill, so a recycled PID is never touched, and only processes
owned by this user are considered. `reap_orphans()` cleans up sandboxes left by earlier runs: it
kills only starter trees that contain Harbor's exec server (`_hbexec.py`).
"""
from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from typing import Dict, Iterable, Set, Tuple

POLL_S = 2.0
_UID = os.getuid()
_lock = threading.Lock()
_servers: Dict[int, dict] = {}          # cli pid -> {"st": starttime, "starters": {pid: starttime}}
_installed = False
stats = {"registered": 0, "reaped_sandboxes": 0, "killed_processes": 0}


def _table() -> Dict[int, Tuple[int, int, str]]:
    """{pid: (ppid, starttime, comm)} for this user's processes."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != _UID:
                continue
            with open(f"/proc/{d}/stat") as fh:
                s = fh.read()
        except OSError:
            continue
        r = s.rfind(")")
        comm = s[s.find("(") + 1:r]
        f = s[r + 2:].split()
        out[int(d)] = (int(f[1]), int(f[19]), comm)       # ppid (field 4), starttime (field 22)
    return out


def _descendants(root: int, table) -> Set[int]:
    kids: Dict[int, list] = {}
    for p, (pp, _, _) in table.items():
        kids.setdefault(pp, []).append(p)
    seen, stack = set(), [root]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _kill_tree(root: int, root_st: int, table) -> int:
    if root not in table or table[root][1] != root_st:
        return 0                                           # gone, or the PID was recycled
    victims = {root} | _descendants(root, table)
    n = 0
    for p in victims:
        try:
            os.kill(p, signal.SIGKILL)
            n += 1
        except (ProcessLookupError, PermissionError):
            pass
    return n


def register(pid: int) -> None:
    t = _table()
    with _lock:
        _servers[pid] = {"st": t[pid][1] if pid in t else None, "starters": {}}
        stats["registered"] += 1


def poll_once() -> None:
    t = _table()
    with _lock:
        items = list(_servers.items())
    for pid, info in items:
        alive = pid in t and (info["st"] is None or t[pid][1] == info["st"])
        if alive:
            for d in _descendants(pid, t):
                if t[d][2] == "starter":
                    info["starters"].setdefault(d, t[d][1])
            continue
        n = sum(_kill_tree(sp, st, t) for sp, st in info["starters"].items())
        with _lock:
            _servers.pop(pid, None)
            if n:
                stats["reaped_sandboxes"] += 1
                stats["killed_processes"] += n


def _loop() -> None:
    while True:
        try:
            poll_once()
        except Exception:                                  # never take the harbor process down
            pass
        time.sleep(POLL_S)


def _reap_all_at_exit() -> None:
    t = _table()
    for info in list(_servers.values()):
        for sp, st in info["starters"].items():
            _kill_tree(sp, st, t)


class _AsyncioShim:
    """Stands in for `asyncio` inside harbor's singularity module; registers sandbox servers."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def create_subprocess_exec(self, *args, **kwargs):
        proc = await self._real.create_subprocess_exec(*args, **kwargs)
        if len(args) > 1 and os.path.basename(str(args[0])) in ("singularity", "apptainer") and args[1] == "exec":
            register(proc.pid)
        return proc


def install() -> bool:
    """Idempotent. Returns True if the harbor singularity module was patched."""
    global _installed
    if _installed:
        return True
    try:
        import asyncio
        from harbor.environments.singularity import singularity as sing
    except Exception:
        return False
    sing.asyncio = _AsyncioShim(asyncio)
    threading.Thread(target=_loop, name="dnah-sandbox-reaper", daemon=True).start()
    atexit.register(_reap_all_at_exit)
    _installed = True
    return True


def reap_orphans(dry_run: bool = False) -> int:
    """Kill leftover Harbor sandboxes from earlier runs: starter trees, re-parented away from any live
    `singularity` CLI, that contain Harbor's exec server. Returns the number of sandboxes reaped."""
    t = _table()
    n = 0
    for p, (pp, st, comm) in t.items():
        if comm != "starter" or (pp in t and t[pp][2] in ("singularity", "apptainer")):
            continue
        tree = _descendants(p, t)
        try:
            ours = any(b"_hbexec.py" in open(f"/proc/{d}/cmdline", "rb").read() for d in tree)
        except OSError:
            ours = False
        if ours:
            n += 1
            if not dry_run:
                _kill_tree(p, st, t)
    return n


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    print(f"{'would reap' if dry else 'reaped'} {reap_orphans(dry_run=dry)} orphaned Harbor sandboxes on {os.uname().nodename}")
