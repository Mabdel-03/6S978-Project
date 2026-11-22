"""Gateway policy tests with a stub renderer and a mock vLLM backend (no GPU, no models)."""
import asyncio
import json

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dnaharness import config  # noqa: E402
from dnaharness.serving import gateway as G  # noqa: E402
from dnaharness.serving import native as N  # noqa: E402


class StubRenderer:
    native = False
    stop_token_ids = [1]

    def build(self, res, enable_thinking=True):
        if res.dna_shas:
            raise N.HandleError("no native")
        text = " ".join(t for m in res.messages for _, t in m["parts"])
        return {"prompt": list(range(len(text.split()))), "n_prompt": len(text.split()), "n_dna": 0}

    def count(self, text):
        return len(text.split())


@pytest.fixture()
def gw(tmp_path):
    config.load_secrets(create=True)
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["Authorization"] == f"Bearer {config.load_secrets()['vllm_api_key']}"
        return httpx.Response(200, json={"choices": [{"text": "<think>\nhmm\n</think>\n\nAnswer: x", "finish_reason": "stop"}],
                                         "usage": {"completion_tokens": 7}})

    g = G.Gateway.__new__(G.Gateway)
    g.secrets = config.load_secrets()
    g.policy_name = "sampled"
    g.policies = config.models()["decoding"]
    g.handle_map = {}
    g.trials = {}
    g.session_defaults = {"arm": "mini_swe", "max_generated_tokens": 20}
    g.log_messages = False
    g.log_path = tmp_path / "calls.jsonl"
    g._log_lock = asyncio.Lock()
    g.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    g.backends = {"m": {"cfg": {}, "renderer": StubRenderer(), "url": "http://vllm/v1/completions"}}
    g._seen = seen
    return g


def _admin():
    return {"X-Admin-Key": config.load_secrets()["gateway_admin_key"]}


def _chat(c, token, route, **body):
    return c.post(f"/t/{token}/{route}/v1/chat/completions",
                  json={"model": "m", "messages": [{"role": "user", "content": "hello there"}], **body})


def test_registration_requires_admin_key(gw):
    c = TestClient(G.create_app(gw))
    assert c.post("/admin/trials", json={"token": "t", "model": "m", "arm": "rlm"}).status_code == 401
    assert c.post("/admin/trials", json={"token": "t", "model": "m", "arm": "rlm"}, headers=_admin()).status_code == 200
    assert _chat(c, "unknown", "root").status_code == 401


def test_coding_arm_child_blocked_and_counted(gw):
    c = TestClient(G.create_app(gw))
    c.post("/admin/trials", json={"token": "t", "model": "m", "arm": "coding_repl", "subcalls": False}, headers=_admin())
    assert _chat(c, "t", "child").status_code == 403
    assert _chat(c, "t", "root").status_code == 200
    st = c.get("/admin/trials/t", headers=_admin()).json()
    assert st["blocked_child_calls"] == 1 and st["n_calls"]["child"] == 0


def test_sampling_is_enforced_and_think_split(gw):
    c = TestClient(G.create_app(gw))
    c.post("/admin/trials", json={"token": "t", "model": "m", "arm": "rlm", "subcalls": True}, headers=_admin())
    r = _chat(c, "t", "root", temperature=1.9, top_p=0.1, max_tokens=99999).json()
    sent = gw._seen[-1]
    pol = config.models()["decoding"]["sampled"]
    assert (sent["temperature"], sent["top_p"], sent["top_k"]) == (pol["temperature"], pol["top_p"], pol["top_k"])
    assert sent["max_tokens"] <= pol["max_tokens"]
    msg = r["choices"][0]["message"]
    assert msg["content"] == "Answer: x" and msg["reasoning_content"] == "hmm"


def test_budget_exhaustion_429(gw):
    c = TestClient(G.create_app(gw))
    c.post("/admin/trials", json={"token": "t", "model": "m", "arm": "rlm", "subcalls": True,
                                  "max_generated_tokens": 10, "max_child_calls": 1}, headers=_admin())
    assert _chat(c, "t", "child").status_code == 200          # 7 tokens
    assert _chat(c, "t", "child").status_code == 429          # child-call cap
    assert _chat(c, "t", "root").status_code == 200          # 14 > 10 now
    assert _chat(c, "t", "root").status_code == 429
    assert c.get("/admin/trials/t", headers=_admin()).json()["budget_exhausted"] is True


def test_unique_seeds_for_concurrent_calls(gw):
    gw.register({"token": "t", "model": "m", "arm": "vote", "max_generated_tokens": 10**6})
    st = gw.trials["t"]

    async def go():
        body = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
        await asyncio.gather(*[gw.chat(st, "direct", body) for _ in range(8)])
    asyncio.run(go())
    seeds = [b["seed"] for b in gw._seen]
    assert len(set(seeds)) == 8


def test_session_route_for_mini_swe(gw):
    c = TestClient(G.create_app(gw))
    assert c.post("/v1/chat/completions", json={"model": "openai/m", "messages": []}).status_code == 401
    r = c.post("/v1/chat/completions", headers={"X-Session-ID": "trialX__env"},
               json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "session:trialX__env" in gw.trials


def test_max_tokens_clamped_to_context_room_and_full_window_is_429(gw):
    """max_tokens never exceeds what fits in vLLM's --max-model-len; a full window ends the trial (429)."""
    gw.max_model_len = 100
    c = TestClient(G.create_app(gw))
    c.post("/admin/trials", json={"token": "t", "model": "m", "arm": "rlm", "subcalls": True,
                                  "max_generated_tokens": 10**6}, headers=_admin())
    assert _chat(c, "t", "root").status_code == 200                  # 2-token prompt
    assert gw._seen[-1]["max_tokens"] == 100 - 2 - 8
    long = {"role": "user", "content": " ".join(["w"] * 60)}         # 60 + 8 > 100 - 64
    r = c.post("/t/t/root/v1/chat/completions", json={"model": "m", "messages": [long]})
    assert r.status_code == 429 and len(gw._seen) == 1             # never forwarded to vLLM
    assert c.get("/admin/trials/t", headers=_admin()).json()["budget_exhausted"] is True
