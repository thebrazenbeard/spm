import json
from pathlib import Path


def test_v3_changes_only_injection_depth_from_v2():
    root = Path(__file__).resolve().parents[1]
    v2 = json.loads((root / "experiments" / "spm0_vector_state_bstar_c_v2.json").read_text())
    v3 = json.loads((root / "experiments" / "spm0_vector_state_bstar_c_v3.json").read_text())

    for key in ("subject", "arms", "precision", "optimization", "data", "decision_rules"):
        assert v3[key] == v2[key]

    a2 = dict(v2["architecture"])
    a3 = dict(v3["architecture"])
    assert a2.pop("injection_layer_index") == 31
    assert a3.pop("injection_layer_index") == 23
    assert a2.pop("remaining_backbone_blocks") == [32, 33, 34, 35]
    assert a3.pop("remaining_backbone_blocks") == list(range(24, 36))
    assert a3 == a2

    assert v3["predecessor"]["result_commit"] == "51ccd4fc5d5d3a287546f9de6e2bdd120eac8052"
    assert v3["predecessor"]["change_class"] == "INJECTION_DEPTH_ONLY"
    assert v3["retune_after_comparison"] is False
