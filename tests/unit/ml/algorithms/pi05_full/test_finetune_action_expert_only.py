"""Stage 2: VLM params get requires_grad=False, weights unchanged after step."""

import torch

from .conftest import PI05_FULL_TEST_ARGS


def test_action_expert_only_freezes_vlm(
    pi05_full_cls, model_init_description_with_subtask
):
    model = pi05_full_cls(
        model_init_description_with_subtask,
        **PI05_FULL_TEST_ARGS,
        finetune_action_expert_only=True,
    )
    expert_param_substr = (
        "gemma_expert",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    )
    for name, param in model.model.named_parameters():
        if any(s in name for s in expert_param_substr):
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name


def test_action_expert_only_step_does_not_change_vlm(
    pi05_full_cls, model_init_description_with_subtask, synthetic_training_batch
):
    model = pi05_full_cls(
        model_init_description_with_subtask,
        **PI05_FULL_TEST_ARGS,
        finetune_action_expert_only=True,
        subtask_loss_weight=0.0,
        fast_token_loss_weight=0.0,
    )
    optimizer = model.configure_optimizers()[0]
    vlm_param = model.model.paligemma_with_expert.paligemma.language_model.layers[
        0
    ].self_attn.k_proj.weight
    before = vlm_param.detach().clone()

    out = model.training_step(synthetic_training_batch)
    out.losses["loss"].backward()
    optimizer.step()

    assert torch.allclose(vlm_param.detach(), before)
