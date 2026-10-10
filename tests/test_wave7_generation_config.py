import math
import pytest
from spm_bench.baseline_manifest import ArtifactIdentity, BaselineSubject

def test_nonfinite_generation_config_cannot_be_digested_as_standard_json():
    model=ArtifactIdentity("base", inventory_digest="a"*64)
    tokenizer=ArtifactIdentity("tokenizer", inventory_digest="b"*64)
    with pytest.raises(ValueError,match="JSON-serializable|finite"):
        BaselineSubject(model=model,tokenizer=tokenizer,parameter_count=100, generation_config={"temperature":math.nan})
