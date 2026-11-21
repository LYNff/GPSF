# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Adapted from LeapLabTHU/ARC (ICCV 2023), Apache-2.0.
# Upstream commit: 382fcdb616d544c63e73d22c3cf3a429f06bd1eb
# Source files: mmdet/models/backbones/modules/adaptive_rotated_conv.py
#               mmdet/models/backbones/modules/routing_function.py
# License: ../licenses/ARC-APACHE-2.0.txt
# Changes: combine the two files; use permute instead of einops; self-contained
# trunc_normal fallback; input validation; dtype-safe rotation constants;
# build the official routing module internally; default to four kernel banks.
"""Official ARC kernel rotation/routing adapted for an HMPN reconstruction.

This is a true input-conditioned rotated kernel bank, not a fixed directional
convolution. Routing is per input image, not a dense pixel-wise angle field.
Only the upstream analytic 3x3 kernel rotation is supported. No CUDA extension
or MMCV op is required. The HMPN integration is an independent reconstruction,
not official OriMamba code.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = [
    'AdaptiveRotatedConv2d', 'ARConv', 'RoutingFunction', 'RountingFunction',
    'LayerNormProxy', 'batch_rotate_multiweight',
]


def _trunc_normal_(tensor, std=0.02):
    """timm/PyTorch truncated normal, with a PyTorch 1.x fallback."""
    if hasattr(nn.init, 'trunc_normal_'):
        return nn.init.trunc_normal_(tensor, std=std)
    with torch.no_grad():
        # Same absolute [-2, 2] bounds as upstream trunc_normal_.
        low = (1.0 + math.erf(-2.0 / std / math.sqrt(2.0))) / 2.0
        high = (1.0 + math.erf(2.0 / std / math.sqrt(2.0))) / 2.0
        tensor.uniform_(2.0 * low - 1.0, 2.0 * high - 1.0)
        tensor.erfinv_().mul_(std * math.sqrt(2.0)).clamp_(-2.0, 2.0)
    return tensor


class LayerNormProxy(nn.Module):
    """Official ARC channel LayerNorm at each spatial location."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class RoutingFunction(nn.Module):
    """Official ARC routing, producing [B, N] mixing gates and radians."""

    def __init__(self, in_channels, kernel_number, dropout_rate=0.2,
                 proportion=40.0):
        super().__init__()
        if not 0.0 < proportion <= 45.0:
            raise ValueError('ARC analytic rotation requires 0 < angle range <= 45 degrees')
        self.kernel_number = kernel_number
        self.dwc = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_alpha = nn.Linear(in_channels, kernel_number, bias=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_theta = nn.Linear(in_channels, kernel_number, bias=False)
        self.act_func = nn.Softsign()
        self.proportion = proportion / 180.0 * math.pi
        _trunc_normal_(self.dwc.weight, std=0.02)
        _trunc_normal_(self.fc_alpha.weight, std=0.02)
        _trunc_normal_(self.fc_theta.weight, std=0.02)

    def forward(self, x):
        x = self.relu(self.norm(self.dwc(x)))
        x = self.avg_pool(x).flatten(1)
        alphas = torch.sigmoid(self.fc_alpha(self.dropout1(x)))
        angles = self.act_func(self.fc_theta(self.dropout2(x)))
        return alphas, angles * self.proportion

    def extra_repr(self):
        return 'kernel_number={}, angle_range_degrees={:.1f}'.format(
            self.kernel_number, self.proportion * 180.0 / math.pi)


# Keep the spelling used in the original repository for source compatibility.
RountingFunction = RoutingFunction


def _get_rotation_matrix(thetas):
    bs, g = thetas.shape
    # Upstream defaults to float32 constants. Match the input dtype so that
    # float64 verification and explicit half-precision models remain valid.
    def zeros(*shape):
        return thetas.new_zeros(shape)

    def ones(*shape):
        return thetas.new_ones(shape)
    thetas = thetas.reshape(-1)  # [bs, n] --> [bs x n]
    
    x = torch.cos(thetas)
    y = torch.sin(thetas)
    x = x.unsqueeze(0).unsqueeze(0)  # shape = [1, 1, bs * g]
    y = y.unsqueeze(0).unsqueeze(0)
    a = x - y
    b = x * y
    c = x + y

    rot_mat_positive = torch.cat((
        torch.cat((a, 1-a, zeros(1, 7, bs*g)), dim=1),
        torch.cat((zeros(1, 1, bs*g), x-b, b, zeros(1, 1, bs*g), 1-c+b, y-b, zeros(1, 3, bs*g)), dim=1),
        torch.cat((zeros(1, 2, bs*g), a, zeros(1, 2, bs*g), 1-a, zeros(1, 3, bs*g)), dim=1),
        torch.cat((b, y-b, zeros(1,1 , bs*g), x-b, 1-c+b, zeros(1, 4, bs*g)), dim=1),
        torch.cat((zeros(1, 4, bs*g), ones(1, 1, bs*g), zeros(1, 4, bs*g)), dim=1),
        torch.cat((zeros(1, 4, bs*g), 1-c+b, x-b, zeros(1, 1, bs*g), y-b, b), dim=1),
        torch.cat((zeros(1, 3, bs*g), 1-a, zeros(1, 2, bs*g), a, zeros(1, 2, bs*g)), dim=1),
        torch.cat((zeros(1, 3, bs*g), y-b, 1-c+b, zeros(1, 1, bs*g), b, x-b, zeros(1, 1, bs*g)), dim=1),
        torch.cat((zeros(1, 7, bs*g), 1-a, a), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    rot_mat_negative = torch.cat((
        torch.cat((c, zeros(1, 2, bs*g), 1-c, zeros(1, 5, bs*g)), dim=1),
        torch.cat((-b, x+b, zeros(1, 1, bs*g), b-y, 1-a-b, zeros(1, 4, bs*g)), dim=1),
        torch.cat((zeros(1, 1, bs*g), 1-c, c, zeros(1, 6, bs*g)), dim=1),
        torch.cat((zeros(1, 3, bs*g), x+b, 1-a-b, zeros(1, 1, bs*g), -b, b-y, zeros(1, 1, bs*g)), dim=1),
        torch.cat((zeros(1, 4, bs*g), ones(1, 1, bs*g), zeros(1, 4, bs*g)), dim=1),
        torch.cat((zeros(1, 1, bs*g), b-y, -b, zeros(1, 1, bs*g), 1-a-b, x+b, zeros(1, 3, bs*g)), dim=1),
        torch.cat((zeros(1, 6, bs*g), c, 1-c, zeros(1, 1, bs*g)), dim=1),
        torch.cat((zeros(1, 4, bs*g), 1-a-b, b-y, zeros(1, 1, bs*g), x+b, -b), dim=1),
        torch.cat((zeros(1, 5, bs*g), 1-c, zeros(1, 2, bs*g), c), dim=1)
    ), dim=0)  # shape = [k^2, k^2, bs*g]

    mask = (thetas >= 0).unsqueeze(0).unsqueeze(0)
    mask = mask.to(dtype=thetas.dtype)                                                   # shape = [1, 1, bs*g]
    rot_mat = mask * rot_mat_positive + (1 - mask) * rot_mat_negative     # shape = [k*k, k*k, bs*g]
    rot_mat = rot_mat.permute(2, 0, 1)                                    # shape = [bs*g, k*k, k*k]
    rot_mat = rot_mat.reshape(bs, g, rot_mat.shape[1], rot_mat.shape[2])  # shape = [bs, g, k*k, k*k]
    return rot_mat


def batch_rotate_multiweight(weights, lambdas, thetas):
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        thetas: tensor of thetas,  shape = [batch_size, kernel_number]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    if weights.ndim != 5 or weights.shape[-2:] != (3, 3):
        raise ValueError('Official ARC rotation supports 3x3 kernel banks only')
    if thetas.ndim != 2 or thetas.shape != lambdas.shape:
        raise ValueError('thetas and lambdas must have the same [B, N] shape')
    if lambdas.shape[1] != weights.shape[0]:
        raise ValueError('Routing kernel count does not match the weight bank')
    # Keep parameter dtype for kernel synthesis; autocast may still select its
    # convolution/matmul dtype. These casts preserve routing gradients.
    thetas = thetas.to(device=weights.device, dtype=weights.dtype)
    lambdas = lambdas.to(device=weights.device, dtype=weights.dtype)

    b = thetas.shape[0]
    n = thetas.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    # Stage 1:
    # input: thetas: [b, n]
    #        lambdas: [b, n]
    # output: rotation_matrix: [b, n, 9, 9] (with gate) --> [b*9, n*9]

    #       Sub_Stage 1.1:
    #       input: [b, n] kernel
    #       output: [b, n, 9, 9] rotation matrix
    rotation_matrix = _get_rotation_matrix(thetas)

    #       Sub_Stage 1.2:
    #       input: [b, n, 9, 9] rotation matrix
    #              [b, n] lambdas
    #          --> [b, n, 1, 1] lambdas
    #          --> [b, n, 1, 1] lambdas dot [b, n, 9, 9] rotation matrix
    #          --> [b, n, 9, 9] rotation matrix with gate (done)
    #       output: [b, n, 9, 9] rotation matrix with gate
    lambdas = lambdas.unsqueeze(2).unsqueeze(3)
    rotation_matrix = torch.mul(rotation_matrix, lambdas)

    #       Sub_Stage 1.3: Reshape
    #       input: [b, n, 9, 9] rotation matrix with gate
    #       output: [b*9, n*9] rotation matrix with gate
    rotation_matrix = rotation_matrix.permute(0, 2, 1, 3)
    rotation_matrix = rotation_matrix.reshape(b*9, n*9)

    # Stage 2: Reshape 
    # input: weights: [n, Cout, Cin, 3, 3]
    #             --> [n, 3, 3, Cout, Cin]
    #             --> [n*9, Cout*Cin] done
    # output: weights: [n*9, Cout*Cin]
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.contiguous().view(n*9, Cout*Cin)


    # Stage 3: torch.mm
    # [b*9, n*9] x [n*9, Cout*Cin]
    # --> [b*9, Cout*Cin]
    weights = torch.mm(rotation_matrix, weights)

    # Stage 4: Reshape Back
    # input: [b*9, Cout*Cin]
    #    --> [b, 3, 3, Cout, Cin]
    #    --> [b, Cout, Cin, 3, 3]
    #    --> [b * Cout, Cin, 3, 3] done
    # output: [b * Cout, Cin, 3, 3]
    weights = weights.contiguous().view(b, 3, 3, Cout, Cin)
    weights = weights.permute(0, 3, 4, 1, 2)
    weights = weights.reshape(b * Cout, Cin, 3, 3)

    return weights


class AdaptiveRotatedConv2d(nn.Module):
    """ARC: rotate a bank of 3x3 kernels, mix it, convolve each image.

    Args follow the official class. Unlike upstream, routing is instantiated
    internally unless supplied. ``kernel_number=4`` follows the released n4
    configuration (the upstream class itself defaults to 1). ``angle_range``
    is in degrees and defaults to the official routing value of 40.
    ``rounting_func`` preserves the misspelled upstream keyword.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, kernel_number=4,
                 routing_func=None, rounting_func=None,
                 rotate_func=batch_rotate_multiweight, dropout_rate=0.2,
                 angle_range=40.0):
        super().__init__()
        if kernel_size != 3:
            raise ValueError('Official ARC analytic kernel rotation supports kernel_size=3 only')
        if bias:
            raise ValueError('This source-faithful ARC implementation supports bias=False only')
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError('groups must divide in_channels and out_channels')
        if kernel_number < 1:
            raise ValueError('kernel_number must be positive')
        if routing_func is not None and rounting_func is not None:
            raise ValueError('Pass only one routing function keyword')
        self.kernel_number = kernel_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = False
        routing_func = routing_func if routing_func is not None else rounting_func
        self.rounting_func = routing_func if routing_func is not None else RoutingFunction(
            in_channels, kernel_number, dropout_rate=dropout_rate,
            proportion=angle_range)
        self.rotate_func = rotate_func
        self.weight = nn.Parameter(torch.empty(
            kernel_number, out_channels, in_channels // groups, 3, 3))
        # Preserve the exact 5-D fan_out initialization used upstream.
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('ARC expects [B, {}, H, W] input'.format(self.in_channels))
        batch_size, channels, height, width = x.shape
        if batch_size == 0:
            raise ValueError('ARC does not accept an empty image batch')
        alphas, angles = self.rounting_func(x)
        rotated_weight = self.rotate_func(self.weight, alphas, angles)
        packed = x.reshape(1, batch_size * channels, height, width)
        out = F.conv2d(
            packed, rotated_weight, bias=None, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
            groups=batch_size * self.groups)
        return out.reshape(batch_size, self.out_channels, *out.shape[2:])

    def extra_repr(self):
        return ('{}, {}, kernel_size=3, stride={}, padding={}, dilation={}, '
                'groups={}, kernel_number={}, bias=False').format(
                    self.in_channels, self.out_channels, self.stride,
                    self.padding, self.dilation, self.groups, self.kernel_number)


ARConv = AdaptiveRotatedConv2d
