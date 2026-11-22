"""Tests run against a throwaway storage root (secrets, manifest, generated tasks) but reuse the real
HF download cache so the dataset is not re-fetched."""
import os
import re
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_real_root = re.search(r"^root:\s*(\S+)", (_REPO / "configs" / "paths.yaml").read_text(), re.M).group(1)
os.environ.setdefault("DNAH_HF_HOME", os.environ.get("DNAH_TEST_HF_HOME", f"{_real_root}/hf_cache"))
os.environ["DNAH_ROOT"] = tempfile.mkdtemp(prefix="dnah-test-")
os.environ.setdefault("HF_HOME", os.environ["DNAH_HF_HOME"])


@pytest.fixture(scope="session")
def manifest():
    from dnaharness import config
    from dnaharness.data import cohort
    try:
        df = cohort.build_manifest()
    except Exception as e:  # offline / dataset unavailable
        pytest.skip(f"KEGG dataset unavailable: {e}")
    p = config.path("manifest")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df
