"""Contains the DatasetDescription class.

Note we have to use NamedTuple instead of dataclass because of torchscript.
"""

from typing import NamedTuple

import torch


class DatasetDescription(NamedTuple):
    max_num_cameras: int
    max_state_size: int
    max_action_size: int
    action_mean: torch.tensor
    action_std: torch.tensor
    state_mean: torch.tensor
    state_std: torch.tensor
    action_prediction_horizon: int = 1
