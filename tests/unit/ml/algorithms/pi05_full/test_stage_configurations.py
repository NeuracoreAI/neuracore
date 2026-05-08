"""Stage 1, Stage 2, and joint training configurations all run."""

from .conftest import PI05_FULL_TEST_ARGS


def test_stage1_config_runs_training_step(
    pi05_full_cls, model_init_description_with_subtask, synthetic_training_batch
):
    model = pi05_full_cls(
        model_init_description_with_subtask,
        **PI05_FULL_TEST_ARGS,
        flow_matching_loss_weight=0.0,
        subtask_loss_weight=10.0,
        fast_token_loss_weight=1.0,
        finetune_action_expert_only=False,
        knowledge_insulation=False,
    )
    out = model.training_step(synthetic_training_batch)
    assert out.losses["loss"].item() != 0


def test_stage2_config_runs_training_step(
    pi05_full_cls, model_init_description_with_subtask, synthetic_training_batch
):
    model = pi05_full_cls(
        model_init_description_with_subtask,
        **PI05_FULL_TEST_ARGS,
        flow_matching_loss_weight=1.0,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
        finetune_action_expert_only=True,
    )
    out = model.training_step(synthetic_training_batch)
    assert out.losses["loss"].item() != 0


def test_joint_config_runs_training_step(
    pi05_full_cls, model_init_description_with_subtask, synthetic_training_batch
):
    model = pi05_full_cls(model_init_description_with_subtask, **PI05_FULL_TEST_ARGS)
    out = model.training_step(synthetic_training_batch)
    assert out.losses["loss"].item() != 0
