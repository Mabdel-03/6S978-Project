"""Generated tasks: whitelisted inputs, no label anywhere agent-visible, valid Harbor task configs."""
import json
import tomllib
from pathlib import Path

import pytest

from dnaharness.adapters.kegg_dx import adapter as A

AGENT_VISIBLE = ("instruction.md", "environment")


@pytest.fixture(scope="module")
def tasks(manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("kdx")
    A.run(out, interfaces=A.INTERFACES, conditions=("wt", "no_dna", "scramble", "swap_pair"), limit=6)
    return out


@pytest.mark.network
def test_inputs_whitelist_and_no_labels(tasks, manifest):
    from dnaharness.data.kegg import label_vocabulary
    labels = [l.lower() for l in label_vocabulary()]
    index = {json.loads(l)["task_name"]: json.loads(l) for l in (tasks / "_meta" / "index.jsonl").read_text().splitlines()}
    n = 0
    for tdir in tasks.glob("*-*/kdx-*"):
        meta = index[tdir.name]
        inputs = json.loads((tdir / "environment" / "inputs.json").read_text())
        assert set(inputs) <= A.ALLOWED_KEYS[meta["interface"]]
        if meta["condition"] == "no_dna":
            assert set(inputs) == {"question"}
        assert sorted(p.name for p in (tdir / "environment").iterdir()) == ["inputs.json"]
        # nothing agent-visible names the gold label unless the original question text already does --
        # apart from the closed answer list, which is the SAME text in every task (checked here)
        from dnaharness.runtime.prompts import LABEL_LIST
        instr = (tdir / "instruction.md").read_text()
        assert instr.count(LABEL_LIST) == 1
        visible = instr.replace(LABEL_LIST, "").lower() + json.dumps(inputs).lower()
        q = inputs["question"].lower()
        gold = meta["label"].lower()
        if gold not in q:
            assert gold not in visible, tdir
        # opaque names: no row id / condition / label in task name or toml metadata
        toml = tomllib.loads((tdir / "task.toml").read_text())
        blob = json.dumps(toml["metadata"]) + tdir.name
        assert meta["row_id"] not in blob and meta["condition"] not in blob and gold not in blob.lower()
        n += 1
    assert n > 0


@pytest.mark.network
def test_task_toml_validates_with_harbor(tasks):
    harbor = pytest.importorskip("harbor")
    from harbor.models.task.config import TaskConfig
    for tdir in list(tasks.glob("*-*/kdx-*"))[:20]:
        cfg = TaskConfig.model_validate(tomllib.loads((tdir / "task.toml").read_text()))
        assert cfg.environment.workdir == "/task"
        assert cfg.verifier.environment_mode.value == "shared"
        assert cfg.environment.docker_image.endswith(".sif")
        assert cfg.environment.env["NO_PROXY"].startswith("127.0.0.1")


@pytest.mark.network
def test_native_handles_unguessable_and_mapped(tasks):
    handles = json.loads((tasks / "_meta" / "handles.json").read_text())
    seqs = {json.loads(l)["sha"] for l in (tasks / "_meta" / "sequences.jsonl").read_text().splitlines()}
    for tdir in tasks.glob("native-wt/kdx-*"):
        inputs = json.loads((tdir / "environment" / "inputs.json").read_text())
        hs = inputs["dna_handles"]
        assert "reference_sequence" not in inputs
        for h in hs.values():
            assert h in handles and handles[h] in seqs
    # same sequence in different episodes -> different handles (no cross-episode reuse)
    by_sha = {}
    for h, sha in handles.items():
        by_sha.setdefault(sha, []).append(h)
    assert any(len(v) > 1 for v in by_sha.values())
