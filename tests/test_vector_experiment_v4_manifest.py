import json
from pathlib import Path


def test_v4_changes_only_reinjection_topology_from_v3():
    root = Path(__file__).resolve().parents[1]
    v3 = json.loads((root / "experiments" / "spm0_vector_state_bstar_c_v3.json").read_text())
    v4 = json.loads((root / "experiments" / "spm0_vector_state_bstar_c_v4.json").read_text())

    for key in ("subject", "arms", "precision", "optimization", "data", "decision_rules"):
        assert v4[key] == v3[key]

    a3 = dict(v3["architecture"])
    a4 = dict(v4["architecture"])
    assert a4.pop("reinjection_mode") == "broadcast_real_tokens_add"
    assert a4 == a3

    assert v4["predecessor"]["result_commit"] == "539c0fec882efe8585b5ab23ff2523a640686b9c"
    assert v4["predecessor"]["change_class"] == "REINJECTION_TOPOLOGY_ONLY"
    assert v4["mechanism_commit"] == "5c9f966d16eb5ae51252ce9d454ea080fc2e0963"
    assert v4["retune_after_comparison"] is False
