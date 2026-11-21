# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .fam import FAPNAlignment, _disable_autocast, _positive_int


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.out_channels = _positive_int("out_channels", out_channels)
        self.conv_atten = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=False)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Match the c2_xavier_fill used by the official FaPN FSM.
        nn.init.kaiming_uniform_(self.conv_atten.weight, a=1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)

    def init_weights(self):
        """Restore the official FaPN convolution initialization."""
        self.reset_parameters()

    def forward(self, x):
        attention = self.conv_atten(F.adaptive_avg_pool2d(x, 1)).sigmoid()
        return self.conv(x + x * attention)


class DirectionAwareBranch(nn.Module):
    def __init__(self, channels, hidden_channels=32, zero_init=True):
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.hidden_channels = _positive_int("hidden_channels", hidden_channels)
        if not isinstance(zero_init, bool):
            raise TypeError("zero_init must be a bool.")
        self.zero_init = zero_init
        self.reduce = nn.Conv2d(2 * self.channels, self.hidden_channels, 1, bias=False)
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    (1, 7),
                    padding=(0, 3),
                    groups=self.hidden_channels,
                    bias=False,
                ),
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    (7, 1),
                    padding=(3, 0),
                    groups=self.hidden_channels,
                    bias=False,
                ),
                nn.Conv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    3,
                    padding=2,
                    dilation=2,
                    groups=self.hidden_channels,
                    bias=False,
                ),
            ]
        )
        self.weight_logits = nn.Conv2d(self.hidden_channels, 3, 1, bias=True)
        self.project = nn.Conv2d(self.hidden_channels, self.channels, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.reduce.reset_parameters()
        for branch in self.branches:
            branch.reset_parameters()
        nn.init.zeros_(self.weight_logits.weight)
        nn.init.zeros_(self.weight_logits.bias)
        if self.zero_init:
            nn.init.zeros_(self.project.weight)
        else:
            nn.init.kaiming_uniform_(self.project.weight, a=math.sqrt(5))

    def init_weights(self):
        """Reset kernels and restore uniform mixing and configured residual."""
        self.reset_parameters()

    @staticmethod
    def _convolve(layer, x, force_fp32):
        if not force_fp32:
            return layer(x)
        bias = None if layer.bias is None else layer.bias.float()
        return F.conv2d(
            x,
            layer.weight.float(),
            bias,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
        )

    def forward(self, combined, force_fp32=False):
        if not isinstance(force_fp32, bool):
            raise TypeError("force_fp32 must be a bool.")
        if not isinstance(combined, torch.Tensor):
            raise TypeError("Direction-aware input must be a torch tensor.")
        if (
            combined.ndim != 4
            or min(combined.shape) <= 0
            or combined.shape[1] != 2 * self.channels
        ):
            raise ValueError("Direction-aware input must have shape [B,2*C,H,W].")
        if not combined.is_floating_point():
            raise TypeError("Direction-aware input must be floating-point.")
        context = (
            _disable_autocast(combined.device.type) if force_fp32 else nullcontext()
        )
        with context:
            if force_fp32:
                combined = combined.float()
            reduced = self._convolve(self.reduce, combined, force_fp32)
            logits = self._convolve(self.weight_logits, reduced, force_fp32)
            weights = logits.softmax(dim=1)
            directions = torch.stack(
                [
                    self._convolve(branch, reduced, force_fp32)
                    for branch in self.branches
                ],
                dim=1,
            )
            mixed = (directions * weights.unsqueeze(2)).sum(dim=1)
            return self._convolve(self.project, mixed, force_fp32)

    def extra_repr(self):
        return "channels={}, hidden_channels={}, zero_init={}".format(
            self.channels, self.hidden_channels, self.zero_init
        )


class DAFAPNAlignment(FAPNAlignment):
    def __init__(
        self,
        channels,
        deform_groups=8,
        backend="mmcv",
        force_fp32=True,
        dab_hidden_channels=32,
        dab_zero_init=True,
    ):
        dab_hidden_channels = _positive_int("dab_hidden_channels", dab_hidden_channels)
        if not isinstance(dab_zero_init, bool):
            raise TypeError("dab_zero_init must be a bool.")
        super().__init__(
            channels,
            deform_groups=deform_groups,
            backend=backend,
            force_fp32=force_fp32,
        )
        self.directional_branch = DirectionAwareBranch(
            self.channels, dab_hidden_channels, zero_init=dab_zero_init
        )

    def _make_condition(self, fine, coarse):
        combined = torch.cat((fine, 2 * coarse), dim=1)
        if self.force_fp32:
            original = F.conv2d(combined, self.condition.weight.float(), bias=None)
        else:
            original = self.condition(combined)
        return original + self.directional_branch(combined, force_fp32=self.force_fp32)

    def init_weights(self):
        """Reset the base alignment and the added directional branch."""
        super().init_weights()
        self.directional_branch.init_weights()
