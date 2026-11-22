"""Input-use metrics with a pre-registered hierarchical cluster bootstrap.

Definitions (fixed before any sweep is run):
  accuracy            mean over episodes of the seed-averaged correctness
  group_accuracy      mean over text groups of within-group accuracy (each of the 14 groups weighs 1)
  pair_accuracy       within each text group, ALL unordered pairs of rows with different labels AND
                      different pair_id; a pair scores 1 for a seed if both rows are correct for that
                      seed; averaged over seeds, then within group, then across groups (group-weighted).
                      `pair_accuracy_pw` weights every pair equally.
  minority_accuracy   accuracy on rows whose label is not their text group's majority label
  text_only_delta     accuracy(wt) - accuracy(no_dna)        (same arm/model/interface)
  flip_rate[c]        P(predicted label under c != under wt), matched on row and seed
  follows_donor/orig  source preference under swap_pair / text_conflict (diagnostic, not accuracy)
Health: valid_rate, unmapped_rate, error_rate, budget_exhausted_rate, attempted_subcall_rate,
truncated_think_rate.  Cost: mean prompt/completion/reasoning tokens and DNA positions per trial.

Uncertainty: resample text groups with replacement, then pair_ids within each sampled group; all
seeds of a sampled pair stay together (seeds measure decoding variability, not independent biology).
Paired config differences use the SAME resample for both configs. 14 clusters -> wide intervals.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CONFIG_KEYS = ["experiment", "arm", "model", "interface"]


def with_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """`experiment` = job name without the chunk suffix (-c3of8), so a sweep's chunks pool into one config."""
    if "experiment" in df or "job" not in df:
        return df
    return df.assign(experiment=df["job"].astype(str).str.replace(r"-c\d+of\d+$", "", regex=True))


def _seed_avg(d: pd.DataFrame, col: str = "correct") -> pd.Series:
    return d.groupby("row_id")[col].mean()


def accuracy(d: pd.DataFrame) -> float:
    return float(_seed_avg(d).mean()) if len(d) else np.nan


def group_accuracy(d: pd.DataFrame) -> float:
    if not len(d):
        return np.nan
    per_row = d.groupby(["text_group_id", "row_id"])["correct"].mean().reset_index()
    return float(per_row.groupby("text_group_id")["correct"].mean().mean())


def minority_accuracy(d: pd.DataFrame) -> float:
    m = d[d["is_minority_label"] == True]  # noqa: E712
    return accuracy(m)


def _pairs(d: pd.DataFrame) -> List[Tuple[str, str, str]]:
    rows = d.drop_duplicates("row_id")[["row_id", "text_group_id", "label", "pair_id"]]
    out = []
    for g, grp in rows.groupby("text_group_id"):
        recs = grp.to_dict("records")
        for a, b in combinations(recs, 2):
            if a["label"] != b["label"] and a["pair_id"] != b["pair_id"]:
                out.append((g, a["row_id"], b["row_id"]))
    return out


def pair_accuracy(d: pd.DataFrame, pair_weighted: bool = False) -> float:
    if not len(d):
        return np.nan
    by = d.set_index(["row_id", "attempt"])["correct"].to_dict()
    attempts = sorted(d["attempt"].unique())
    scores: Dict[str, List[float]] = {}
    for g, a, b in _pairs(d):
        vals = [float(by[(a, s)] and by[(b, s)]) for s in attempts if (a, s) in by and (b, s) in by]
        if vals:
            scores.setdefault(g, []).append(float(np.mean(vals)))
    if not scores:
        return np.nan
    if pair_weighted:
        return float(np.mean([v for vs in scores.values() for v in vs]))
    return float(np.mean([np.mean(vs) for vs in scores.values()]))


def flip_rate(d_cond: pd.DataFrame, d_wt: pd.DataFrame) -> float:
    m = d_cond.merge(d_wt, on=["row_id", "attempt"], suffixes=("", "_wt"))
    if not len(m):
        return np.nan
    a = m["predicted_label"].fillna("<none>")
    b = m["predicted_label_wt"].fillna("<none>")
    return float((a != b).mean())


def health(d: pd.DataFrame) -> Dict[str, float]:
    if not len(d):
        return {}
    f = lambda c: float(d[c].fillna(False).astype(float).mean()) if c in d else np.nan  # noqa: E731
    out = {
        "n_trials": int(len(d)),
        "valid_rate": f("valid_output"),
        "unmapped_rate": float((d["valid_output"].fillna(False) & ~d.get("mapped", pd.Series(True, index=d.index)).fillna(False)).mean())
        if "mapped" in d else np.nan,
        "error_rate": float(((d["status"] == "error") | d["exception"].notna()).mean()),
        "budget_exhausted_rate": f("budget_exhausted"),
        "attempted_subcall_rate": float((d["blocked_child_calls"].fillna(0) > 0).mean()) if "blocked_child_calls" in d else np.nan,
        "truncated_think_rate": float((d["truncated_in_think"].fillna(0) > 0).mean()) if "truncated_in_think" in d else np.nan,
    }
    for c in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "dna_positions", "child_calls", "n_iterations"):
        if c in d:
            out[f"mean_{c}"] = float(pd.to_numeric(d[c], errors="coerce").mean())
    return out


# ------------------------------------------------------------------------------- bootstrap

def _cluster_resample(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups = d["text_group_id"].unique()
    parts = []
    for gi, g in enumerate(rng.choice(groups, size=len(groups), replace=True)):
        dg = d[d["text_group_id"] == g]
        pairs = dg["pair_id"].unique()
        for pj, p in enumerate(rng.choice(pairs, size=len(pairs), replace=True)):
            part = dg[dg["pair_id"] == p].copy()
            # keep resampled duplicates distinct (pair metrics would otherwise collapse them)
            part["text_group_id"] = f"{g}#{gi}"
            part["row_id"] = part["row_id"] + f"#{gi}.{pj}"
            part["pair_id"] = f"{p}#{gi}.{pj}"
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _cluster_layout(d: pd.DataFrame):
    """Groups and per-group pair_ids in exactly the order _cluster_resample draws them."""
    groups = d["text_group_id"].unique()
    return groups, {g: d.loc[d["text_group_id"] == g, "pair_id"].unique() for g in groups}


def _draw_counts(groups, pairs_by_group, rng) -> List[Tuple[object, np.ndarray]]:
    """Replays _cluster_resample's RNG calls; returns (group, draw count per pair cluster) per group copy."""
    out = []
    for g in rng.choice(groups, size=len(groups), replace=True):
        pairs = pairs_by_group[g]
        drawn = rng.choice(pairs, size=len(pairs), replace=True)
        pos = {p: i for i, p in enumerate(pairs)}
        out.append((g, np.bincount([pos[p] for p in drawn], minlength=len(pairs)).astype(float)))
    return out


class _FastStats:
    """Exact closed-form statistics of a cluster resample, from per-(group, pair) summaries.

    Equivalent to fn(_cluster_resample(d, rng)) for accuracy / group_accuracy / minority_accuracy /
    pair_accuracy (a resample is a multiset of pair clusters per group copy; duplicates are distinct
    rows, and two rows from the SAME copy of a cluster share a pair_id and never form a pair)."""

    def __init__(self, d: pd.DataFrame, groups, pairs_by_group):
        self.ok = len(d) > 0
        att = sorted(d["attempt"].unique()) if len(d) else []
        per_row = d.groupby("row_id")["attempt"].nunique() if len(d) else pd.Series(dtype=int)
        self.pair_ok = self.ok and bool((per_row == len(att)).all())   # pair formula needs full seed grid
        self.g = {}
        for g in groups:
            dg = d[d["text_group_id"] == g]
            pairs = pairs_by_group[g]
            k = len(pairs)
            rec = {"rs": np.zeros(k), "rn": np.zeros(k), "ms": np.zeros(k), "mn": np.zeros(k),
                   "Xn": np.zeros((k, k)), "Xd": np.zeros((k, k))}
            if len(dg):
                rows = dg.groupby("row_id").agg(avg=("correct", "mean"), pair_id=("pair_id", "first"),
                                                label=("label", "first"),
                                                minority=("is_minority_label", "first") if "is_minority_label" in dg else ("label", "size"))
                pos = {p: i for i, p in enumerate(pairs)}
                u = rows["pair_id"].map(pos).to_numpy()
                M1 = np.zeros((len(rows), k))
                M1[np.arange(len(rows)), u] = 1.0
                avg = rows["avg"].astype(float).to_numpy()
                rec["rs"], rec["rn"] = M1.T @ avg, M1.sum(0)
                if "is_minority_label" in dg:
                    mi = (rows["minority"] == True).to_numpy().astype(float)  # noqa: E712
                    rec["ms"], rec["mn"] = M1.T @ (avg * mi), M1.T @ mi
                if self.pair_ok:
                    C = dg.pivot_table(index="row_id", columns="attempt", values="correct", aggfunc="first")
                    C = C.reindex(index=rows.index, columns=att).astype(float).to_numpy()
                    S = (C @ C.T) / len(att)
                    lab = rows["label"].to_numpy()
                    L = (lab[:, None] != lab[None, :]).astype(float)
                    rec["Xn"], rec["Xd"] = M1.T @ (S * L) @ M1, M1.T @ L @ M1
            self.g[g] = rec

    @staticmethod
    def _pairsum(X, m):
        return 0.5 * (m @ X @ m - m @ np.diag(X))

    def stat(self, name: str, draws) -> float:
        if name in ("accuracy", "minority_accuracy"):
            s, c = ("rs", "rn") if name == "accuracy" else ("ms", "mn")
            num = sum(m @ self.g[g][s] for g, m in draws)
            den = sum(m @ self.g[g][c] for g, m in draws)
            return float(num / den) if den > 0 else np.nan
        if name == "group_accuracy":
            v = [(m @ self.g[g]["rs"]) / (m @ self.g[g]["rn"]) for g, m in draws if m @ self.g[g]["rn"] > 0]
            return float(np.mean(v)) if v else np.nan
        if name == "pair_accuracy":
            v = []
            for g, m in draws:
                den = self._pairsum(self.g[g]["Xd"], m)
                if den > 0:
                    v.append(self._pairsum(self.g[g]["Xn"], m) / den)
            return float(np.mean(v)) if v else np.nan
        raise KeyError(name)


_FAST = {accuracy: "accuracy", group_accuracy: "group_accuracy", minority_accuracy: "minority_accuracy",
         pair_accuracy: "pair_accuracy"}


def bootstrap(d: pd.DataFrame, fn: Callable[[pd.DataFrame], float], n: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> Tuple[float, float, float]:
    point = fn(d)
    rng = np.random.default_rng(seed)
    name = _FAST.get(fn)
    if name is not None and len(d):
        groups, pbg = _cluster_layout(d)
        fs = _FastStats(d, groups, pbg)
        if name != "pair_accuracy" or fs.pair_ok:
            vals = [fs.stat(name, _draw_counts(groups, pbg, rng)) for _ in range(n)]
            lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            return point, float(lo), float(hi)
    vals = [fn(_cluster_resample(d, rng)) for _ in range(n)]
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap(da: pd.DataFrame, db: pd.DataFrame, fn: Callable[[pd.DataFrame], float],
                     n: int = 2000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """CI of fn(A) - fn(B) with identical cluster resamples (A, B evaluated on the same episodes)."""
    point = fn(da) - fn(db)
    da = da.assign(_cfg="A")
    db = db.assign(_cfg="B")
    both = pd.concat([da, db], ignore_index=True)
    rng = np.random.default_rng(seed)
    name = _FAST.get(fn)
    if name is not None and len(da) and len(db):
        groups, pbg = _cluster_layout(both)
        fa, fb = _FastStats(da, groups, pbg), _FastStats(db, groups, pbg)
        if name != "pair_accuracy" or (fa.pair_ok and fb.pair_ok):
            vals = []
            for _ in range(n):
                dr = _draw_counts(groups, pbg, rng)
                vals.append(fa.stat(name, dr) - fb.stat(name, dr))
            lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            return point, float(lo), float(hi)
    vals = []
    for _ in range(n):
        r = _cluster_resample(both, rng)
        vals.append(fn(r[r._cfg == "A"]) - fn(r[r._cfg == "B"]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ------------------------------------------------------------------------------- tables

def summary_table(df: pd.DataFrame, n_boot: int = 1000, by_split: bool = False) -> pd.DataFrame:
    """One row per (config, condition[, split]) with point estimates + bootstrap CIs."""
    df = with_experiment(df)
    keys = CONFIG_KEYS + ["condition"] + (["orig_split"] if by_split else [])
    kegg = df[df["row_id"].notna()] if "row_id" in df else df.iloc[0:0]
    out = []
    for k, d in kegg.groupby(keys, dropna=False):
        rec = dict(zip(keys, k))
        for name, fn in (("accuracy", accuracy), ("group_accuracy", group_accuracy),
                         ("pair_accuracy", pair_accuracy), ("minority_accuracy", minority_accuracy)):
            p, lo, hi = bootstrap(d, fn, n=n_boot) if n_boot else (fn(d), np.nan, np.nan)
            rec.update({name: p, f"{name}_lo": lo, f"{name}_hi": hi})
        rec["pair_accuracy_pw"] = pair_accuracy(d, pair_weighted=True)
        rec.update(health(d))
        if "follows_donor" in d and d["follows_donor"].notna().any():
            rec["follows_donor"] = float(d["follows_donor"].mean())
            rec["follows_original"] = float(d["follows_original"].mean())
        out.append(rec)
    t = pd.DataFrame(out)
    if t.empty:
        return t
    # condition contrasts vs wt within the same config
    extra = []
    for k, d in kegg.groupby(CONFIG_KEYS + (["orig_split"] if by_split else []), dropna=False):
        wt = d[d.condition == "wt"]
        for cond, dc in d.groupby("condition"):
            e = dict(zip(CONFIG_KEYS + (["orig_split"] if by_split else []), k))
            e["condition"] = cond
            e["flip_rate_vs_wt"] = flip_rate(dc, wt) if cond != "wt" and len(wt) else np.nan
            if cond == "wt" and (d.condition == "no_dna").any():
                e["text_only_delta"] = accuracy(wt) - accuracy(d[d.condition == "no_dna"])
            extra.append(e)
    return t.merge(pd.DataFrame(extra), on=keys, how="left")


def pilot_table(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df["type"].notna()] if "type" in df else df.iloc[0:0]
    if p.empty:
        return p
    p = p.assign(correct_protocol=p.get("correct_with_protocol", p["correct"]).fillna(p["correct"]))
    for c in ("fallback_answer", "n_code_blocks"):          # absent for mini_swe / older runs
        if c not in p:
            p = p.assign(**{c: float("nan")})
    p = p.assign(fallback=p["fallback_answer"].astype(float), ran_code=(p["n_code_blocks"] > 0).astype(float)
                 .where(p["n_code_blocks"].notna()))
    g = p.groupby(["arm", "model", "type"])
    t = g.agg(correct=("correct", "mean"), correct_protocol=("correct_protocol", "mean"),
              valid=("valid_output", "mean"), fallback=("fallback", "mean"), ran_code=("ran_code", "mean"),
              n=("correct", "size"),
              mean_iters=("n_iterations", "mean"), child_calls=("child_calls", "mean"),
              blocked_child=("blocked_child_calls", "mean")).reset_index()
    return t
