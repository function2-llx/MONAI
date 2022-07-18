from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

@dataclass
class UEncoderOutput:
    cls_feature: torch.Tensor = None
    hidden_states: list[torch.Tensor] = field(default_factory=list)

class UEncoderBase(nn.Module):
    def forward(self, img: torch.Tensor, *args, **kwargs) -> UEncoderOutput:
        raise NotImplementedError

@dataclass
class UDecoderOutput:
    # low->high resolution
    feature_maps: list[torch.Tensor]

class UDecoderBase(nn.Module):
    def forward(self, img: torch.Tensor, encoder_hidden_states: list[torch.Tensor]) -> UDecoderOutput:
        raise not NotImplementedError
