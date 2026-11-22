"""Convenience verifier (runs in the sandbox AFTER the agent finished; /tests is uploaded only then).

Writes /logs/verifier/reward.json = {correct, valid_output, mapped} (+ prediction.json). The
authoritative score is recomputed on the host from /logs/agent/answer.txt.
"""
import json
import os
import sys

sys.path.insert(0, "/opt/dnah")
from dnah_runtime.scoring import score  # noqa: E402

gold = json.load(open("/tests/gold.json"))
ans_path = "/logs/agent/answer.txt"
answer = open(ans_path).read().strip() if os.path.exists(ans_path) else None
valid = bool(answer)
res = score(answer, gold["label"], gold["labels"]) if valid else {
    "predicted_label": None, "correct": False, "mapped": False, "segment": ""}
os.makedirs("/logs/verifier", exist_ok=True)
json.dump({"correct": int(res["correct"]), "valid_output": int(valid), "mapped": int(res["mapped"])},
          open("/logs/verifier/reward.json", "w"))
json.dump({"answer": answer, **res}, open("/logs/verifier/prediction.json", "w"))
print(json.dumps({"valid": valid, **res}))
