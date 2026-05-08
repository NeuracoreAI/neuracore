"""Construction-time validation for Pi05Full."""

import pytest

from .conftest import PI05_FULL_TEST_ARGS


def test_missing_subtask_input_raises(
    pi05_full_cls, model_init_description_no_subtask_input
):
    with pytest.raises(ValueError, match="SUBTASK_LANGUAGE"):
        pi05_full_cls(model_init_description_no_subtask_input, **PI05_FULL_TEST_ARGS)


def test_missing_subtask_output_raises(
    pi05_full_cls, model_init_description_no_subtask_output
):
    with pytest.raises(ValueError, match="SUBTASK_LANGUAGE"):
        pi05_full_cls(model_init_description_no_subtask_output, **PI05_FULL_TEST_ARGS)


def test_negative_loss_weight_raises(
    pi05_full_cls, model_init_description_with_subtask
):
    with pytest.raises(ValueError, match="non-negative"):
        pi05_full_cls(
            model_init_description_with_subtask,
            **PI05_FULL_TEST_ARGS,
            subtask_loss_weight=-1.0,
        )


def test_all_zero_loss_weights_raise(
    pi05_full_cls, model_init_description_with_subtask
):
    with pytest.raises(ValueError, match="At least one loss weight"):
        pi05_full_cls(
            model_init_description_with_subtask,
            **PI05_FULL_TEST_ARGS,
            subtask_loss_weight=0.0,
            fast_token_loss_weight=0.0,
            flow_matching_loss_weight=0.0,
        )


def test_warns_when_subtask_loss_with_action_expert_only(
    caplog, pi05_full_cls, model_init_description_with_subtask
):
    pi05_full_cls(
        model_init_description_with_subtask,
        **PI05_FULL_TEST_ARGS,
        finetune_action_expert_only=True,
        subtask_loss_weight=10.0,
    )
    assert "subtask CE loss has no effect" in caplog.text
