import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


def _positive_channels(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return int(value)


class FrequencyGuidanceGate(nn.Module):
    def __init__(self, channels, kernel_size=3, norm_eps=1e-5):
        super().__init__()
        self.channels = _positive_channels("frequency channels", channels) if self.channels < 2:
            raise ValueError(
                "Frequency guidance requires at least 2 hidden "
                "channels; ChannelNorm would erase a single channel."
            )
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, numbers.Integral)
            or kernel_size < 3
            or kernel_size % 2 == 0
        ):
            raise ValueError("frequency_kernel must be an odd integer >= 3.")
        if (
            isinstance(norm_eps, bool)
            or not isinstance(norm_eps, numbers.Real)
            or not math.isfinite(float(norm_eps))
            or norm_eps <= 0
        ):
            raise ValueError("frequency_norm_eps must be finite and positive.")
        self.kernel_size = int(kernel_size)
        self.norm_eps = float(norm_eps)
        self.projection = nn.Conv2d(2 * self.channels, self.channels, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def highpass(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "Frequency input must be [B, {}, H, W].".format(self.channels)
            )
        if not x.is_floating_point():
            raise TypeError("Frequency input must be floating point.")
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = F.avg_pool2d(
            work,
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            count_include_pad=False,
        )
        return work - mean

    def forward(self, x):
        high = self.highpass(x)
        normalized = F.layer_norm(
            high.permute(0, 2, 3, 1),
            (self.channels,),
            weight=None,
            bias=None,
            eps=self.norm_eps,
        )
        normalized = normalized.permute(0, 3, 1, 2).contiguous().to(x.dtype)
        logits = self.projection(torch.cat((x, normalized), dim=1))
        # Sigmoid in FP32 avoids avoidable loss of resolution in mixed precision.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return (2.0 * logits.sigmoid()).to(x.dtype)


class LowLevelDetailAdapter(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels=32,
        beta_init=0.1,
        learnable_beta=True,
        zero_init=True,
        use_frequency=False,
        frequency_kernel=3,
        frequency_norm_eps=1e-5,
    ):
        super().__init__()
        self.channels = _positive_channels("channels", channels)
        self.hidden_channels = _positive_channels("hidden_channels", hidden_channels)
        if not isinstance(learnable_beta, bool) or not isinstance(zero_init, bool):
            raise ValueError("learnable_beta and zero_init must be booleans.")
        if (
            isinstance(beta_init, bool)
            or not isinstance(beta_init, numbers.Real)
            or not math.isfinite(float(beta_init))
        ):
            raise ValueError("beta_init must be a finite real number.")
        if zero_init and beta_init == 0:
            raise ValueError(
                "Use nonzero beta_init with a zero-initialized up "
                "projection; zeroing both prevents this branch learning."
            )
        if not learnable_beta and beta_init == 0:
            raise ValueError(
                "A fixed zero beta disables all learning in this branch; "
                "use detail_levels=() to disable adapters instead."
            )
        self.beta_init = float(beta_init)
        self.learnable_beta = learnable_beta
        self.zero_init = zero_init
        if not isinstance(use_frequency, bool):
            raise ValueError("use_frequency must be a boolean.")
        self.use_frequency = use_frequency
        self.down = nn.Conv2d(self.channels, self.hidden_channels, 1, bias=False)
        self.local = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            3,
            padding=1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.up = nn.Conv2d(self.hidden_channels, self.channels, 1, bias=False)
        beta = torch.tensor(self.beta_init)
        if learnable_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)
        # Keep the original local adapter's RNG/initialization order. Create
        # the optional gate only after all original parameters are initialized.
        self.frequency_gate = None
        self.reset_parameters()
        if self.use_frequency:
            self.frequency_gate = FrequencyGuidanceGate(
                self.hidden_channels, frequency_kernel, frequency_norm_eps
            )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.kaiming_uniform_(self.local.weight, a=1)
        if self.zero_init:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.xavier_uniform_(self.up.weight)
        with torch.no_grad():
            self.beta.fill_(self.beta_init)
        if self.frequency_gate is not None:
            self.frequency_gate.reset_parameters()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "Expected a [B, {}, H, W] fine feature.".format(self.channels)
            )
        low = self.down(x)
        local = F.silu(self.local(low))
        if self.frequency_gate is not None:
            local = self.frequency_gate(low) * local
        residual = self.up(local)
        # Keep the fine-feature dtype, including mixed-precision autocast.
        return x + self.beta.to(dtype=x.dtype) * residual.to(dtype=x.dtype)

    def extra_repr(self):
        return (
            "channels={}, hidden_channels={}, beta_init={}, "
            "learnable_beta={}, zero_init={}, use_frequency={}".format(
                self.channels,
                self.hidden_channels,
                self.beta_init,
                self.learnable_beta,
                self.zero_init,
                self.use_frequency,
            )
        )
