"""Per-allocation port blocks: several allocations may share a node (sandboxes use the host network)."""
import pytest

from dnaharness import config


@pytest.fixture()
def offset(monkeypatch):
    def set_(v):
        if v is None:
            monkeypatch.delenv("DNAH_PORT_OFFSET", raising=False)
        else:
            monkeypatch.setenv("DNAH_PORT_OFFSET", str(v))
        config.paths.cache_clear()
        config.models.cache_clear()
    yield set_
    monkeypatch.delenv("DNAH_PORT_OFFSET", raising=False)
    config.paths.cache_clear()
    config.models.cache_clear()


def test_offset_shifts_every_port(offset):
    offset(None)
    base = config.paths()
    b_gw, b_v = base["gateway_port"], dict(base["vllm_ports"])
    b_m = config.model_port("bioreason-1.7b")
    offset(300)
    p = config.paths()
    assert p["gateway_port"] == b_gw + 300
    assert p["vllm_ports"] == {k: v + 300 for k, v in b_v.items()}
    assert config.model_port("bioreason-1.7b") == b_m + 300


def test_runtime_gateway_url_wins_over_baked(monkeypatch):
    pytest.importorskip("harbor")
    from dnaharness.harbor_ext import agents as A
    monkeypatch.delenv("DNAH_GATEWAY_URL", raising=False)
    assert A.gateway_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    monkeypatch.setenv("DNAH_GATEWAY_URL", "http://127.0.0.1:8300")
    assert A.gateway_url("http://127.0.0.1:8000") == "http://127.0.0.1:8300"


def test_mini_swe_subclass_rewrites_only_api_base(monkeypatch, tmp_path):
    pytest.importorskip("harbor")
    from dnaharness.harbor_ext import agents as A
    monkeypatch.setenv("DNAH_GATEWAY_URL", "http://127.0.0.1:8300")
    cfg = {"agent": {"step_limit": 12}, "model": {"model_class": "litellm_textbased",
                                                  "model_kwargs": {"api_base": "http://127.0.0.1:8000/v1", "api_key": "sandbox"}}}
    ag = A.MiniSweGatewayAgent(logs_dir=tmp_path, model_name="openai/qwen3-1.7b", config=cfg, skip_install=True)
    import yaml
    seen = yaml.safe_load(ag._config_yaml)
    assert seen["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:8300/v1"
    assert seen["model"]["model_class"] == "litellm_textbased" and seen["agent"] == cfg["agent"]
    assert cfg["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:8000/v1"     # caller's dict untouched
    assert ag.name() == "mini-swe-agent"
