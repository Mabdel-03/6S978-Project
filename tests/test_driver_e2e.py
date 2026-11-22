"""End-to-end: the in-sandbox rlms driver (run_rlm) <-> the real gateway <-> a scripted mock model.

Checks the controlled-comparison guarantees on the actual code path:
  * rlm arm: llm_query goes to the /child route and its value is usable; answer submitted via answer[]
  * coding arm: llm_query is NOT defined in the REPL (NameError) and no child call reaches the model
"""
import asyncio
import json
import socket
import threading
import time

import httpx
import pytest

pytest.importorskip("rlm")
pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from dnaharness import config  # noqa: E402
from dnaharness.runtime import run_rlm  # noqa: E402
from dnaharness.serving import gateway as G  # noqa: E402

RLM_SCRIPT = [
    "Let me look.\n```repl\nprint(list(context))\nr = llm_query('Reply with exactly: HELPER_OK')\nprint('child said', r)\n```",
    "```repl\nanswer['content'] = 'prion disease'\nanswer['ready'] = True\n```",
]
CODING_SCRIPT = [
    "```repl\ntry:\n    llm_query('x')\nexcept NameError as e:\n    print('NO_SUBCALLS', e)\n"
    "print(find_differences(context['reference_sequence'], context['variant_sequence'])['type'])\n```",
    "```repl\nanswer['content'] = 'melanoma'\nanswer['ready'] = True\n```",
]


class TextStub:
    native = False
    stop_token_ids = [1]

    def build(self, res, enable_thinking=True):
        text = "\n".join(f"[{m['role']}] " + "".join(t for _, t in m["parts"]) for m in res.messages)
        return {"prompt": text, "n_prompt": len(text.split()), "n_dna": 0}

    def count(self, text):
        return len(text.split())


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def server(tmp_path):
    config.load_secrets(create=True)
    state = {"root_turn": 0, "child_calls": 0, "script": RLM_SCRIPT}

    def backend(request):
        body = json.loads(request.content)
        prompt = body["prompt"]
        if "HELPER_OK" in prompt and prompt.count("[user]") == 1 and "[system]" not in prompt:
            state["child_calls"] += 1
            text = "<think>\nok\n</think>\n\nHELPER_OK"
        else:
            i = min(state["root_turn"], len(state["script"]) - 1)
            state["root_turn"] += 1
            text = f"<think>\nplan\n</think>\n\n{state['script'][i]}"
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": "stop"}],
                                         "usage": {"completion_tokens": len(text.split())}})

    g = G.Gateway.__new__(G.Gateway)
    g.secrets, g.policy_name, g.policies = config.load_secrets(), "sampled", config.models()["decoding"]
    g.handle_map, g.trials, g.session_defaults, g.log_messages = {}, {}, {}, False
    g.log_path = tmp_path / "calls.jsonl"
    g._log_lock = asyncio.Lock()
    g.client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    g.backends = {"m": {"cfg": {}, "renderer": TextStub(), "url": "http://vllm/v1/completions"}}
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(G.create_app(g), host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    yield g, f"http://127.0.0.1:{port}", state
    srv.should_exit = True
    th.join(timeout=5)


def _run(g, url, tmp_path, arm, inputs):
    g.register({"token": arm, "model": "m", "arm": arm, "subcalls": arm == "rlm", "max_generated_tokens": 10**6})
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    out = tmp_path / f"out-{arm}"
    rc = run_rlm.main(["--inputs", str(tmp_path / "inputs.json"), "--arm", arm, "--model", "m", "--gateway", url,
                       "--token", arm, "--max-iterations", "4", "--out", str(out)])
    assert rc == 0
    return out


def test_rlm_arm_subcall_and_submit(server, tmp_path):
    g, url, state = server
    out = _run(g, url, tmp_path, "rlm", {"question": "Which disease?", "reference_sequence": "ACGT", "variant_sequence": "ACTT"})
    assert (out / "answer.txt").read_text() == "prion disease"
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["status"] == "ok"
    assert state["child_calls"] == 1 and g.trials["rlm"].n_calls["child"] == 1
    log = "".join(p.read_text() for p in out.glob("rlm_*.jsonl"))
    assert "child said HELPER_OK" in log


def test_coding_arm_has_no_subcalls(server, tmp_path):
    g, url, state = server
    state["script"], state["root_turn"] = CODING_SCRIPT, 0
    out = _run(g, url, tmp_path, "coding_repl", {"question": "Which disease?", "reference_sequence": "ACGTA", "variant_sequence": "ACTTA"})
    assert (out / "answer.txt").read_text() == "melanoma"
    log = "".join(p.read_text() for p in out.glob("rlm_*.jsonl"))
    assert "NO_SUBCALLS" in log and "SNV" in log         # llm_query undefined; seqtools available
    st = g.trials["coding_repl"]
    assert st.n_calls["child"] == 0 and state["child_calls"] == 0
    # the system prompt the root saw never mentions llm_query
    calls = [json.loads(l) for l in (tmp_path.parent / tmp_path.name / "calls.jsonl").read_text().splitlines()]
    assert all(c["route"] == "root" for c in calls if c["trial"] == "coding_repl")


def test_unfenced_code_is_a_protocol_failure_not_a_submission(server, tmp_path):
    """Smoke-run regression: a 1.7B model wrote `answer["content"] = ...` WITHOUT a ```repl fence every
    turn. Nothing executes; rlms then makes its own out-of-turns "final answer" call and logs it as
    iteration max+1 with final_answer set. That must be reported as max_iterations/fallback, not ok."""
    g, url, state = server
    state["script"], state["root_turn"] = ['answer["content"] = "melanoma"'], 0
    out = _run(g, url, tmp_path, "coding_repl", {"question": "Which disease?", "reference_sequence": "ACGTA", "variant_sequence": "ACTTA"})
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["status"] == "max_iterations" and meta["fallback_answer"] is True
    assert meta["n_iterations"] == 4 and meta["n_code_blocks"] == 0
    assert "melanoma" in (out / "answer.txt").read_text()     # the fallback text is still scored


def test_bioreason_tool_works_without_dna(server, tmp_path):
    """Exp B no_dna condition: the tool must still answer (canonical prompt = question only)."""
    g, url, state = server
    state["script"], state["root_turn"] = ["```repl\nr = bioreason_answer()\nanswer['content'] = 'prion disease'\nanswer['ready'] = True\n```"], 0
    g.register({"token": "tool", "model": "m", "arm": "coding_repl+bioreason_tool", "subcalls": False, "tool": True,
                "max_generated_tokens": 10**6})
    (tmp_path / "inputs.json").write_text(json.dumps({"question": "Which disease?"}))
    out = tmp_path / "out-tool"
    rc = run_rlm.main(["--inputs", str(tmp_path / "inputs.json"), "--arm", "coding_repl", "--model", "m", "--gateway", url,
                       "--token", "tool", "--bioreason-tool", "--max-iterations", "3", "--out", str(out)])
    assert rc == 0
    assert json.loads((out / "run_meta.json").read_text())["status"] == "ok"
    assert g.trials["tool"].n_calls["tool"] == 1
