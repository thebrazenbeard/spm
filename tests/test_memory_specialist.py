import pytest

from spm_bench.memory_specialist import memory_adapter_active


def test_memory_adapter_activation_depends_only_on_retrieval_state():
    assert memory_adapter_active(0) is False
    assert memory_adapter_active(1) is True
    assert memory_adapter_active(7) is True


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_memory_adapter_activation_fails_closed_on_invalid_count(value):
    with pytest.raises((TypeError, ValueError)):
        memory_adapter_active(value)
