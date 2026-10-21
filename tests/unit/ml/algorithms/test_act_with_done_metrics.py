import torch
from torch import nn

from neuracore.ml.algorithms.act_with_done.act_with_done import ACTWithDone


def _pool_stub(detach: bool) -> ACTWithDone:
    model = ACTWithDone.__new__(ACTWithDone)
    nn.Module.__init__(model)
    model.detach_done_encoder = detach
    model.done_spatial_score = nn.Linear(4, 1)
    return model


def test_done_metrics_ignore_masked_features_and_score_at_one_half():
    logits = torch.tensor([[[2.0, -2.0, 2.0]]])
    targets = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    metrics = ACTWithDone._done_classification_metrics(logits, targets, mask)

    assert metrics["done_accuracy"].item() == 1.0
    assert metrics["done_precision"].item() == 1.0
    assert metrics["done_recall"].item() == 1.0


def test_done_metrics_are_zero_when_the_positive_sets_are_empty():
    logits = torch.tensor([[[-1.0, -1.0]]])
    targets = torch.tensor([[[1.0, 0.0]]])
    mask = torch.tensor([[1.0, 1.0]])

    metrics = ACTWithDone._done_classification_metrics(logits, targets, mask)

    assert metrics["done_accuracy"].item() == 0.5
    assert metrics["done_precision"].item() == 0.0
    assert metrics["done_recall"].item() == 0.0


def test_detach_done_encoder_stops_gradients_into_the_feature_map():
    model = _pool_stub(detach=True)
    features = torch.randn(2, 4, 3, 3, requires_grad=True)
    pos = torch.randn(2, 4, 3, 3, requires_grad=True)

    pooled = model._spatial_attention_pool(*model._done_encoder_tensors(features, pos))
    pooled.sum().backward()

    assert features.grad is None
    assert pos.grad is None
    assert model.done_spatial_score.weight.grad is not None


def test_attached_done_encoder_trains_the_feature_map():
    model = _pool_stub(detach=False)
    features = torch.randn(2, 4, 3, 3, requires_grad=True)
    pos = torch.randn(2, 4, 3, 3, requires_grad=True)

    pooled = model._spatial_attention_pool(*model._done_encoder_tensors(features, pos))
    pooled.sum().backward()

    assert features.grad is not None
    assert torch.count_nonzero(features.grad) > 0


def _proprio_stub(detach: bool) -> ACTWithDone:
    model = ACTWithDone.__new__(ACTWithDone)
    nn.Module.__init__(model)
    model.detach_done_encoder = detach
    model.done_head = nn.Linear(4, 1)
    model.done_proprio_to_logits = nn.Linear(2, 1)
    nn.init.constant_(model.done_proprio_to_logits.weight, 0.5)
    # Columns 0 and 2 stand in for joint position and gripper. Column 1 is
    # another proprio channel the done head must ignore.
    model.done_proprio_slices = [(0, 1), (2, 3)]
    return model


def test_detach_done_encoder_stops_gradients_into_joint_and_gripper_state():
    model = _proprio_stub(detach=True)
    vision = torch.randn(2, 4, requires_grad=True)
    proprio = torch.randn(2, 4, requires_grad=True)

    model._predict_done_logits(vision, proprio).sum().backward()

    assert proprio.grad is None
    assert model.done_proprio_to_logits.weight.grad is not None


def test_attached_done_head_trains_only_the_joint_and_gripper_columns():
    model = _proprio_stub(detach=False)
    vision = torch.randn(2, 4, requires_grad=True)
    proprio = torch.randn(2, 4, requires_grad=True)

    model._predict_done_logits(vision, proprio).sum().backward()

    assert proprio.grad is not None
    assert torch.count_nonzero(proprio.grad[:, 0]) > 0
    assert torch.count_nonzero(proprio.grad[:, 2]) > 0
    assert torch.count_nonzero(proprio.grad[:, 1]) == 0
