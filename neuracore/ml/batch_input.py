"""Contains the BatchInput class.

Note we have to use NamedTuple instead of dataclass because of torchscript.
"""

from typing import NamedTuple

import torch


class BatchInput(NamedTuple):
    states: torch.FloatTensor
    states_mask: torch.FloatTensor
    camera_images: torch.FloatTensor
    camera_images_mask: torch.FloatTensor
    actions: torch.FloatTensor
    actions_mask: torch.FloatTensor
    actions_sequence_mask: torch.FloatTensor

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchInput(
            states=self.states.to(device),
            states_mask=self.states_mask.to(device),
            camera_images=self.camera_images.to(device),
            camera_images_mask=self.camera_images_mask.to(device),
            actions=self.actions.to(device),
            actions_mask=self.actions_mask.to(device),
            actions_sequence_mask=self.actions_sequence_mask.to(device),
        )

    def __len__(self):
        if self.states is not None:
            return self.states.size(0)
        if self.camera_images is not None:
            return self.camera_images.size(0)
        if self.actions is not None:
            return self.actions.size(0)
        raise ValueError("No tensor found in the batch input")
