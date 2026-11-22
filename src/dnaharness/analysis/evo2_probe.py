"""Linear probes on frozen Evo2 embeddings (blocks.20.mlp.l3, the input to BioReason's DNA projection):
how much label information does the DNA representation itself carry, compared with what BioReason
extracts from it? Decides whether embedding-reading TOOLS (probe / nearest training variants) are worth
giving the harness.

Regime A (paper protocol): bio-mirage analysis/dna_probe_raw.py, ported with explicit arguments:
  the 165 cohort rows; StandardScaler + LogisticRegression(C=1); 5 folds grouped by DNA pair x 10 seeds;
  within-fold text-majority baseline; within-text label-permutation null; DNA-pair cluster bootstrap.
  Training folds contain OTHER DNA pairs of the same question texts.
Regime B (tool regime): fit on KEGG rows outside the cohort, test on the 165.
  B-strict  manifest role train_pool: no shared DNA pair or question text with cohort/dev rows.
  B-locus   every non-cohort row whose DNA pair differs from all cohort pairs (same-gene variants and
            same-text rows allowed) -- what a nearest-training-variant tool would search.
  Models: the same logistic probe, and k-NN (cosine) as a proxy for a `neighbors()` tool.
  Control: the fitted models applied to the cohort's SCRAMBLED windows (same base composition): if
  accuracy survives, the signal is composition, not sequence.
Features: mean over positions (as bio-mirage) of the variant block, ref||var, and var-ref; raw Evo2 and
  after the BioReason SFT projection (W*mean+b: the mean commutes with the affine map).
Reference: BioReason greedy native direct on the same rows (results/repro_trials.parquet, wt).

    python -m dnaharness.analysis.evo2_probe prepare        # wt windows of all KEGG rows -> <datasets>/kegg-wt-all/_meta
    sbatch slurm/evo2_cache.sbatch <datasets>/kegg-wt-all/_meta
    python -m dnaharness.analysis.evo2_probe run --out results/evo2_probe      # serve env (torch + sklearn)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dnaharness import config

META = "kegg-wt-all/_meta"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ----------------------------------------------------------------------------------------- prepare

def prepare() -> Path:
    from dnaharness.data import conditions
    man = pd.read_parquet(config.path("manifest"))
    out = config.path("datasets") / META
    out.mkdir(parents=True, exist_ok=True)
    seqs, rows = {}, []
    for rid in man["row_id"]:
        ref, var = conditions.build(rid, ["wt"])["wt"].dna
        seqs[_sha(ref)], seqs[_sha(var)] = ref, var
        rows.append({"row_id": rid, "ref_sha": _sha(ref), "var_sha": _sha(var)})
    # the cohort's windows must be byte-identical to the generated tasks' (same Evo2 cache entries)
    idx = [json.loads(l) for l in (config.path("datasets") / "kegg-dx/_meta/index.jsonl").read_text().splitlines()]
    wt = {r["row_id"]: r["dna_shas"] for r in idx if r["condition"] == "wt" and r["interface"] == "raw"}
    n_checked = 0
    for r in rows:
        if r["row_id"] in wt:
            assert [r["ref_sha"], r["var_sha"]] == wt[r["row_id"]], r["row_id"]
            n_checked += 1
    assert n_checked == 165, n_checked
    (out / "sequences.jsonl").write_text("".join(json.dumps({"sha": k, "seq": v}) + "\n" for k, v in seqs.items()))
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} rows, {len(seqs)} distinct windows -> {out} (cohort windows verified: {n_checked})")
    return out


# --------------------------------------------------------------------------------------- features

_POOLED: dict = {}


def _pool(cache: Path, shas) -> np.ndarray:
    import torch
    for s in set(shas) - set(_POOLED):
        _POOLED[s] = torch.load(cache / f"{s}.pt").float().mean(0).numpy().astype(np.float32)
    return np.stack([_POOLED[s] for s in shas])


def features(row_ids, ref_shas, var_shas, cache: Path, W: np.ndarray, b: np.ndarray) -> dict:
    ref, var = _pool(cache, ref_shas), _pool(cache, var_shas)
    pref, pvar = ref @ W.T + b, var @ W.T + b
    return {
        "raw_var": var, "raw_ref||raw_var": np.hstack([ref, var]), "raw_var-raw_ref": var - ref,
        "proj_var": pvar, "proj_ref||proj_var": np.hstack([pref, pvar]),
    }


# ------------------------------------------------------------ regime A (bio-mirage dna_probe_raw.py)

def _folds(groups, k, seed):
    u = np.array(sorted(set(groups)))
    r = np.random.default_rng(seed)
    r.shuffle(u)
    f = {g: i % k for i, g in enumerate(u)}
    fa = np.array([f[g] for g in groups])
    return [(np.where(fa != i)[0], np.where(fa == i)[0]) for i in range(k)]


def _lr(Ftr, ytr, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    return sc, LogisticRegression(max_iter=5000, C=C).fit(sc.transform(Ftr), ytr)


def _fit_predict(Ftr, ytr, Fte, C=1.0):
    sc, clf = _lr(Ftr, ytr, C)
    return clf.predict(sc.transform(Fte))


def _par(jobs):
    """Run independent (Ftr, ytr, Fte) fits in parallel; results in input order."""
    import os
    from joblib import Parallel, delayed
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return Parallel(n_jobs=n)(delayed(_fit_predict)(*j) for j in jobs)


def probe_cv(F, y, grp, tid, kmask, C=1.0, seeds=range(10)):
    o, mi, tb, rowc = [], [], [], np.zeros(len(y))
    splits = {s: _folds(grp, 5, s) for s in seeds}
    preds = iter(_par([(F[tr], y[tr], F[te], C) for s in seeds for tr, te in splits[s]]))
    for s in seeds:
        P = np.empty(len(y), dtype=object)
        TB = np.empty(len(y), dtype=object)
        for tr, te in splits[s]:
            P[te] = next(preds)
            m = {t: collections.Counter(y[tr][tid[tr] == t]).most_common(1)[0][0] for t in set(tid[tr])}
            gm = collections.Counter(y[tr]).most_common(1)[0][0]
            TB[te] = [m.get(t, gm) for t in tid[te]]
        o.append((P == y).mean())
        mi.append((P[kmask == "minor"] == y[kmask == "minor"]).mean())
        tb.append((TB == y).mean())
        rowc += (P == y) / len(seeds)
    return dict(acc=float(np.mean(o)), acc_sd=float(np.std(o)), minor=float(np.mean(mi)),
                text_baseline=float(np.mean(tb))), rowc


def null_within_text(F, y, grp, tid, C=1.0, n=20):
    perms, jobs = [], []
    for k in range(n):
        r2 = np.random.default_rng(100 + k)
        ys = y.copy()
        for t in sorted(set(tid)):        # bio-mirage iterates texts in index order
            mm = np.where(tid == t)[0]
            ys[mm] = r2.permutation(y[mm])
        perms.append((ys, _folds(grp, 5, k)))
        jobs += [(F[tr], ys[tr], F[te], C) for tr, te in perms[-1][1]]
    preds, out = iter(_par(jobs)), []
    for ys, splits in perms:
        P = np.empty(len(ys), dtype=object)
        for tr, te in splits:
            P[te] = next(preds)
        out.append((P == ys).mean())
    return float(np.mean(out)), float(np.std(out))


def boot_genome(rowc, grp, n=2000, seed=0):
    genomes = sorted(set(grp))
    gi = {g: np.where(grp == g)[0] for g in genomes}
    r = np.random.default_rng(seed)
    m = [rowc[np.concatenate([gi[genomes[j]] for j in r.integers(0, len(genomes), len(genomes))])].mean()
         for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def bm_kind(y, tid):
    """bio-mirage majority / minority / tied label per row, within its question text."""
    kind = np.empty(len(y), dtype=object)
    for t in set(tid):
        idx = np.where(tid == t)[0]
        c = collections.Counter(y[idx]).most_common()
        top, second = c[0][1], (c[1][1] if len(c) > 1 else 0)
        for i in idx:
            nn = dict(c)[y[i]]
            kind[i] = "tied" if (nn == top and top == second) else ("major" if nn == top else "minor")
    return kind


# ------------------------------------------------------------------------------------- regime B

def _score_rows(coh: pd.DataFrame, pred) -> dict:
    from dnaharness.analysis import metrics as M
    from dnaharness.data.labels import canonical_gold, normalize
    d = coh.assign(correct=[normalize(p or "") == canonical_gold(g) for p, g in zip(pred, coh["label"])], attempt=0)
    acc, lo, hi = M.bootstrap(d, M.accuracy, n=2000)
    return {"acc": acc, "ci95": [lo, hi], "pair_acc": M.pair_accuracy(d),
            "minority_acc": M.minority_accuracy(d), "group_acc": M.group_accuracy(d)}


def run(out: Path, cache: Path) -> dict:
    from sklearn.neighbors import KNeighborsClassifier
    import torch
    man = pd.read_parquet(config.path("manifest"))
    rows = pd.DataFrame([json.loads(l) for l in (config.path("datasets") / META / "rows.jsonl").read_text().splitlines()])
    df = man.merge(rows, on="row_id")
    miss = [s for s in set(df.ref_sha) | set(df.var_sha) if not (cache / f"{s}.pt").exists()]
    assert not miss, f"{len(miss)} windows not in the Evo2 cache: run slurm/evo2_cache.sbatch on {META}"
    st = torch.load(config.path("ckpts") / "bioreason-1.7b" / "dna_projection.pt", map_location="cpu")
    W, b = st["weight"].float().numpy(), st["bias"].float().numpy()
    feats = features(df.row_id, df.ref_sha, df.var_sha, cache, W, b)
    y = df["label"].to_numpy(dtype=object)
    coh = df.index[df.role == "cohort"].to_numpy()
    res = {"n_rows": int(len(df)), "n_cohort": int(len(coh)), "regime_A": {}, "regime_B": {}}

    # ---- A: bio-mirage protocol within the cohort
    ya, grp = y[coh], df.loc[coh, "pair_id"].to_numpy()
    tid = df.loc[coh, "text_group_id"].astype("category").cat.codes.to_numpy()
    kmask = bm_kind(ya, tid)
    res["regime_A"]["minor_rows"] = int((kmask == "minor").sum())
    for name, F in feats.items():
        r, rowc = probe_cv(F[coh], ya, grp, tid, kmask)
        r["ci95_genome"] = boot_genome(rowc, grp)
        r["null_within_text"] = null_within_text(F[coh], ya, grp, tid)
        res["regime_A"][name] = r
        print(f"A {name:<20} acc {r['acc']:.3f}±{r['acc_sd']:.3f} CI {r['ci95_genome'][0]:.3f}-{r['ci95_genome'][1]:.3f} "
              f"minor {r['minor']:.3f} text {r['text_baseline']:.3f} null {r['null_within_text'][0]:.3f}", flush=True)

    # ---- B: fit outside the cohort, test on it (+ scrambled-window control)
    idx = [json.loads(l) for l in (config.path("datasets") / "kegg-dx/_meta/index.jsonl").read_text().splitlines()]
    scr = {r["row_id"]: r["dna_shas"] for r in idx if r["condition"] == "scramble" and r["interface"] == "raw"}
    c = df.loc[coh]
    scr_feats = features(c.row_id, [scr[r][0] for r in c.row_id], [scr[r][1] for r in c.row_id], cache, W, b)
    coh_pairs = set(c.pair_id)
    coh_df = c[["row_id", "text_group_id", "pair_id", "label", "is_minority_label"]].reset_index(drop=True)
    regimes = {"B-strict": df.index[df.role == "train_pool"].to_numpy(),
               "B-locus": df.index[(df.role != "cohort") & ~df.pair_id.isin(coh_pairs)].to_numpy()}
    for rname, tr in regimes.items():
        cov = float(np.isin(ya, np.unique(y[tr])).mean())
        same_text = {t: collections.Counter(y[tr][df.loc[tr, "text_group_id"].to_numpy() == t]).most_common(1)
                     for t in set(c.text_group_id)}
        gm = collections.Counter(y[tr]).most_common(1)[0][0]
        tb_pred = [(same_text[t][0][0] if same_text[t] else gm) for t in c.text_group_id]
        out_r = {"n_train": int(len(tr)), "cohort_label_coverage": cov,
                 "text_baseline": _score_rows(coh_df, tb_pred)}
        for name in ("raw_var", "raw_ref||raw_var", "proj_ref||proj_var"):
            F = feats[name]
            sc, clf = _lr(F[tr], y[tr])
            knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(sc.transform(F[tr]), y[tr])
            out_r[name] = {
                "probe": _score_rows(coh_df, clf.predict(sc.transform(F[coh]))),
                "probe_on_scrambled": _score_rows(coh_df, clf.predict(sc.transform(scr_feats[name]))),
                "knn5": _score_rows(coh_df, knn.predict(sc.transform(F[coh]))),
            }
            p = out_r[name]
            print(f"{rname} {name:<20} probe {p['probe']['acc']:.3f} (pair {p['probe']['pair_acc']:.3f}) "
                  f"scrambled {p['probe_on_scrambled']['acc']:.3f}  knn5 {p['knn5']['acc']:.3f}  "
                  f"text {out_r['text_baseline']['acc']:.3f}  coverage {cov:.2f}", flush=True)
        res["regime_B"][rname] = out_r

    # ---- reference: BioReason greedy native direct (paper-reproduction run) on the same rows
    rp = Path("results/repro_trials.parquet")
    if rp.exists():
        from dnaharness.analysis import metrics as M
        d = pd.read_parquet(rp)
        ref = {}
        for cond in ("wt", "scramble"):
            x = d[d.condition == cond]
            a, lo, hi = M.bootstrap(x, M.accuracy, n=2000)
            ref[cond] = {"acc": a, "ci95": [lo, hi], "pair_acc": M.pair_accuracy(x), "minority_acc": M.minority_accuracy(x)}
        res["bioreason_greedy"] = ref
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"-> {out / 'probe.json'}")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--out", default="results/evo2_probe")
    r.add_argument("--cache", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare()
    else:
        run(Path(a.out), Path(a.cache) if a.cache else config.path("evo2_cache"))


if __name__ == "__main__":
    main()
