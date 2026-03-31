import torch
from neuracore_types import BatchedJointData, CrossEmbodimentDescription, DataType

from neuracore.ml.utils.policy_inference import PolicyInference


def _make_policy_inference(
    output_embodiment_description: CrossEmbodimentDescription,
) -> PolicyInference:
    policy_inference = PolicyInference.__new__(PolicyInference)
    policy_inference.output_embodiment_description = output_embodiment_description
    policy_inference.input_embodiment_description = {}
    policy_inference.org_id = "test_org"
    policy_inference.job_id = None
    policy_inference.device = torch.device("cpu")
    policy_inference.model = None
    policy_inference.input_dataset_statistics = {}
    policy_inference.prediction_horizon = 1
    return policy_inference


def _joint_prediction(value: float) -> BatchedJointData:
    return BatchedJointData(value=torch.full((1, 1, 1), value))


def _indexed_names(*names: str) -> dict[int, str]:
    return dict(enumerate(names))


def test_assign_names_to_model_outputs_drops_extra_padded_tensors() -> None:
    policy_inference = _make_policy_inference({
        DataType.JOINT_TARGET_POSITIONS: _indexed_names("joint1", "joint2"),
    })

    first_prediction = _joint_prediction(0.1)
    second_prediction = _joint_prediction(0.2)
    extra_prediction = _joint_prediction(0.3)

    outputs = policy_inference._assign_names_to_model_outputs({
        DataType.JOINT_TARGET_POSITIONS: [
            first_prediction,
            second_prediction,
            extra_prediction,
        ]
    })

    assert list(outputs[DataType.JOINT_TARGET_POSITIONS].keys()) == [
        "joint1",
        "joint2",
    ]
    assert outputs[DataType.JOINT_TARGET_POSITIONS]["joint1"] is first_prediction
    assert outputs[DataType.JOINT_TARGET_POSITIONS]["joint2"] is second_prediction
    assert extra_prediction not in outputs[DataType.JOINT_TARGET_POSITIONS].values()
