# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import numbers
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return int(value)


def _disable_autocast(device_type):
    """Cover modern PyTorch and the CUDA AMP API used by legacy MMRotate."""
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device_type, enabled=False)
    if device_type == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _load_mmcv_operator():
    # Do not import MMCV at module import time: the original, FAM-disabled
    # pure-PyTorch neck and the small-input reference must remain importable.
    try:
        from mmcv.ops import modulated_deform_conv2d
    except (ImportError, OSError, RuntimeError) as error:
        raise ImportError(
            'FAM backend="mmcv" requires mmcv-full with compiled DCNv2 '
            "operators matching your PyTorch/CUDA installation. "
            'backend="reference" is only for small correctness tests; '
            "it is not an automatic training fallback."
        ) from error
    return modulated_deform_conv2d


def modulated_deform_conv2d_reference(
    input, offset, mask, weight, bias=None, deform_groups=1
):
    deform_groups = _positive_int("deform_groups", deform_groups)
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError("Reference input/weight must be four-dimensional.")
    batch, channels, height, width = input.shape
    if min(batch, channels, height, width) <= 0:
        raise ValueError("Reference input dimensions must be nonempty.")
    if channels % deform_groups:
        raise ValueError("Input channels must be divisible by deform_groups.")
    if tuple(weight.shape[1:]) != (channels, 3, 3):
        raise ValueError("Reference weight must have shape [C_out,C_in,3,3].")
    if tuple(offset.shape) != (batch, 18 * deform_groups, height, width):
        raise ValueError("Reference offset must have shape [B,18*G,H,W].")
    if tuple(mask.shape) != (batch, 9 * deform_groups, height, width):
        raise ValueError("Reference mask must have shape [B,9*G,H,W].")
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError("Reference bias must have shape [C_out].")
    tensors = [input, offset, mask, weight] + ([] if bias is None else [bias])
    if any(not t.is_floating_point() for t in tensors):
        raise TypeError("Reference operands must be floating-point tensors.")
    if any(t.device != input.device or t.dtype != input.dtype for t in tensors):
        raise ValueError("Reference operands must share one device and dtype.")
    sampled_elements = batch * channels * height * width * 9
    multiply_adds = sampled_elements * weight.shape[0]
    if sampled_elements > 1000000 or multiply_adds > 160000000 or height * width > 4096:
        raise RuntimeError(
            "The FAM reference backend only supports small correctness-test "
            'tensors. Use backend="mmcv" for training or full-size features.'
        )

    ys = torch.arange(height, device=input.device, dtype=input.dtype)
    xs = torch.arange(width, device=input.device, dtype=input.dtype)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:  # PyTorch versions before meshgrid(indexing=...)
        yy, xx = torch.meshgrid(ys, xs)
    kernel_y = input.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    kernel_x = input.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    offsets = offset.reshape(batch, deform_groups, 9, 2, height, width)
    sample_y = (
        yy.view(1, 1, 1, height, width)
        + kernel_y.view(1, 1, 9, 1, 1)
        + offsets[:, :, :, 0]
    )
    sample_x = (
        xx.view(1, 1, 1, height, width)
        + kernel_x.view(1, 1, 9, 1, 1)
        + offsets[:, :, :, 1]
    )
    grid = torch.stack(
        (2 * (sample_x + 0.5) / width - 1, 2 * (sample_y + 0.5) / height - 1), dim=-1
    )
    # Native DCNv2 explicitly excludes coordinates <=-1 or >=H/W; the mask
    # also matches its offset derivative at those exact boundary coordinates.
    valid = (sample_y > -1) & (sample_y < height) & (sample_x > -1) & (sample_x < width)
    grouped = input.reshape(
        batch * deform_groups, channels // deform_groups, height, width
    )
    sampled = F.grid_sample(
        grouped,
        grid.reshape(batch * deform_groups, 9 * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.reshape(
        batch, deform_groups, channels // deform_groups, 9, height, width
    )
    modulation = mask.reshape(batch, deform_groups, 9, height, width) * valid.to(
        input.dtype
    )
    sampled = sampled * modulation.unsqueeze(2)
    sampled = sampled.reshape(batch, channels, 9, height, width)
    output = torch.einsum(
        "ock,bckhw->bohw", weight.reshape(weight.shape[0], channels, 9), sampled
    )
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


class FAPNAlignment(nn.Module):
    """Align already-upsampled coarse features using a fine-level reference.

    ``forward(fine, upsampled_coarse)`` returns only ``ReLU(DCNv2(coarse))``.
    The caller retains ownership of the upsampling and the final fine+coarse
    fusion. There is no FSM, geometry head, or extra supervision here.

    The official two-input design is preserved: ``[fine, 2*coarse]`` passes
    through a 1x1 condition projection; a separate 3x3 convolution predicts
    offsets and masks from that condition, not from coarse features alone.
    Zero-initialized offsets/mask logits do not make this module an identity:
    the DCN convolution and ReLU remain, and initial masks equal 0.5.
    """

    def __init__(self, channels, deform_groups=8, backend="mmcv", force_fp32=True):
        super().__init__()
        self.channels = _positive_int("channels", channels)
        self.deform_groups = _positive_int("deform_groups", deform_groups)
        if self.channels % self.deform_groups:
            raise ValueError("channels must be divisible by deform_groups.")
        if backend not in ("mmcv", "reference"):
            raise ValueError('FAM backend must be "mmcv" or "reference".')
        if not isinstance(force_fp32, bool):
            raise TypeError("force_fp32 must be a bool.")
        self.backend = backend
        self.force_fp32 = force_fp32
        self._mmcv_operator = _load_mmcv_operator() if backend == "mmcv" else None
        self.condition = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.offset_mask = nn.Conv2d(
            channels, 27 * self.deform_groups, 3, padding=1, bias=True
        )
        self.weight = nn.Parameter(torch.empty(channels, channels, 3, 3))
        self.bias = nn.Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self):
        # fvcore.c2_xavier_fill uses kaiming_uniform_(a=1), not torch.xavier_.
        nn.init.kaiming_uniform_(self.condition.weight, a=1)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        bound = 1.0 / math.sqrt(9 * self.channels)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def init_weights(self):
        """Restore the official initialization if a framework requests it."""
        self.reset_parameters()

    def _validate_inputs(self, fine, coarse):
        if not isinstance(fine, torch.Tensor) or not isinstance(coarse, torch.Tensor):
            raise TypeError("FAM inputs must be torch tensors.")
        if fine.ndim != 4 or coarse.ndim != 4 or fine.shape != coarse.shape:
            raise ValueError("FAM inputs must have identical [B,C,H,W] shapes.")
        if min(fine.shape) <= 0 or fine.shape[1] != self.channels:
            raise ValueError("FAM input dimensions/channels do not match the module.")
        if not fine.is_floating_point() or not coarse.is_floating_point():
            raise TypeError("FAM inputs must be floating-point tensors.")
        if fine.device != coarse.device or fine.dtype != coarse.dtype:
            raise ValueError("FAM inputs must have the same device and dtype.")

    def _make_condition(self, fine, coarse):
        """Project ``[fine, 2*coarse]`` into C conditioning channels.

        This feature tensor is not a physical sampling offset. The separate
        ``offset_mask`` layer converts it to DCNv2 offsets and mask logits.
        Keeping this hook separate lets variants refine the condition while
        retaining the original sampling and precision behavior.
        """
        combined = torch.cat((fine, 2 * coarse), dim=1)
        if self.force_fp32:
            return F.conv2d(combined, self.condition.weight.float(), bias=None)
        return self.condition(combined)

    def _forward(self, fine, coarse):
        if self.force_fp32:
            fine, coarse = fine.float(), coarse.float()
            condition = self._make_condition(fine, coarse)
            fields = F.conv2d(
                condition,
                self.offset_mask.weight.float(),
                self.offset_mask.bias.float(),
                padding=1,
            )
            weight, bias = self.weight.float(), self.bias.float()
        else:
            condition = self._make_condition(fine, coarse)
            fields = self.offset_mask(condition)
            # MMCV similarly adopts the offset dtype under AMP; keep the
            # explicit reference path consistent with that convention.
            coarse = coarse.to(dtype=fields.dtype)
            weight = self.weight.to(dtype=fields.dtype)
            bias = self.bias.to(dtype=fields.dtype)
        # Keep the first 18G channels in their existing order. FaPN's
        # chunk(o1,o2,mask) followed by cat(o1,o2) does not rearrange them.
        offset = fields[:, : 18 * self.deform_groups].contiguous()
        mask = fields[:, 18 * self.deform_groups :].sigmoid().contiguous()
        if self.backend == "mmcv":
            # Functional is autograd.Function.apply in MMCV 1.x: use
            # positional arguments, including ordinary groups=1.
            aligned = self._mmcv_operator(
                coarse.contiguous(),
                offset,
                mask,
                weight.contiguous(),
                bias,
                1,
                1,
                1,
                1,
                self.deform_groups,
            )
        else:
            aligned = modulated_deform_conv2d_reference(
                coarse, offset, mask, weight, bias, self.deform_groups
            )
        return F.relu(aligned, inplace=False)

    def forward(self, fine, upsampled_coarse):
        self._validate_inputs(fine, upsampled_coarse)
        context = (
            _disable_autocast(fine.device.type) if self.force_fp32 else nullcontext()
        )
        with context:
            aligned = self._forward(fine, upsampled_coarse)
        return aligned.to(dtype=upsampled_coarse.dtype)

    def extra_repr(self):
        return "channels={}, deform_groups={}, backend={!r}, force_fp32={}".format(
            self.channels, self.deform_groups, self.backend, self.force_fp32
        )
