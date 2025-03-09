"""Contains the BatchOutput class.

Note we have to use NamedTuple instead of dataclass because of torchscript.
"""

from typing import NamedTuple

import torch


class BatchOutput(NamedTuple):
    action_predicitons: torch.FloatTensor
    losses: dict[str, torch.FloatTensor]
    metrics: dict[str, torch.FloatTensor]
