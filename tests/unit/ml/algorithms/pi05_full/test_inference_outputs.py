"""Inference returns joint/gripper actions plus a SUBTASK_LANGUAGE field."""

from neuracore_types import DataType
from neuracore_types.batched_nc_data.batched_language_data import BatchedLanguageData

from .conftest import PI05_FULL_TEST_ARGS


def test_forward_returns_subtask_language(
    pi05_full_cls, model_init_description_with_subtask, synthetic_inference_batch
):
    model = pi05_full_cls(model_init_description_with_subtask, **PI05_FULL_TEST_ARGS)
    out = model.forward(synthetic_inference_batch)
    assert DataType.SUBTASK_LANGUAGE in out
    items = out[DataType.SUBTASK_LANGUAGE]
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, BatchedLanguageData)
    assert item.input_ids.shape[0] == synthetic_inference_batch.batch_size
