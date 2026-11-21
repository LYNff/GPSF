import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .arconv import AdaptiveRotatedConv2d
from .detail import LowLevelDetailAdapter
from .scan import GeometryLineScan


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        y = F.layer_norm(
            x.permute(0, 2, 3, 1).float(),
            (x.shape[1],),
            self.weight.float(),
            self.bias.float(),
            self.eps,
        )
        return y.permute(0, 3, 1, 2).contiguous().to(x.dtype)


def _init_conv(conv, zero=False):
    if zero:
        nn.init.zeros_(conv.weight)
    else:
        nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def normalize_direction(raw, eps=1e-4):
    raw = raw.float()
    norm = torch.norm(raw, p=2, dim=1, keepdim=True)
    fallback = torch.cat((torch.ones_like(norm), torch.zeros_like(norm)), 1)
    return torch.where(norm >= eps, raw / norm.clamp(min=eps), fallback)


class ARCAdapter(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_number=2,
        dropout=0.0,
        angle_range=40.0,
        residual_init=0.1,
        zero_init_residual=True,
    ):
        super().__init__()
        self.down = nn.Conv2d(channels, hidden_channels, 1)
        self.arc = AdaptiveRotatedConv2d(
            hidden_channels,
            hidden_channels,
            kernel_number=kernel_number,
            dropout_rate=dropout,
            angle_range=angle_range,
        )
        self.up = nn.Conv2d(hidden_channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(float(residual_init)))
        _init_conv(self.down)
        _init_conv(self.up, zero=zero_init_residual)

    def forward(self, x):
        residual = self.up(self.arc(F.silu(self.down(x))))
        return x + self.gamma.to(x.dtype) * residual


class TransverseRelation(nn.Module):
    def __init__(self, channels, scales=(1.5, 3.0), max_radius=16.0):
        super().__init__()
        if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("context_scales must be finite and positive.")
        if not math.isfinite(max_radius) or max_radius <= 0:
            raise ValueError("context_max_radius must be positive.")
        self.scales = tuple(float(s) for s in scales)
        self.max_radius = float(max_radius)
        # Center, side mean, center-minus-side, side absolute difference,
        # and two validity indicators (count/asymmetry) per sampling scale.
        self.project = nn.Conv2d(
            channels * (1 + 3 * len(scales)) + 2 * len(scales), channels, 1
        )
        self.norm = ChannelNorm(channels)
        _init_conv(self.project)

    @staticmethod
    def _sample_pair(x, q, radius):
        batch, channels, height, width = x.shape
        # q represents an unoriented axis. theta +/- pi swaps the two samples;
        # the relation below is symmetric to that swap.
        theta = 0.5 * torch.atan2(q[:, 1], q[:, 0])
        nx, ny = -theta.sin(), theta.cos()
        ys = torch.arange(height, device=x.device, dtype=torch.float32)
        xs = torch.arange(width, device=x.device, dtype=torch.float32)
        try:
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:  # older PyTorch in legacy MMRotate environments
            yy, xx = torch.meshgrid(ys, xs)
        dx, dy = radius[:, 0] * nx, radius[:, 0] * ny
        sample_x = torch.stack((xx + dx, xx - dx), dim=-1)
        sample_y = torch.stack((yy + dy, yy - dy), dim=-1)
        grid = torch.stack(
            (2 * (sample_x + 0.5) / width - 1, 2 * (sample_y + 0.5) / height - 1),
            dim=-1,
        )
        # A strict center-domain mask excludes synthetic zero-padding values.
        valid = (
            (sample_x >= 0)
            & (sample_x <= width - 1)
            & (sample_y >= 0)
            & (sample_y <= height - 1)
        )
        sampled = F.grid_sample(
            x.float(),
            grid.reshape(batch, height, width * 2, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(batch, channels, height, width, 2)
        valid = valid.unsqueeze(1).float()
        return sampled * valid, valid

    def forward(self, x, q, log_width, foreground):
        fields = [x.float()]
        rho = log_width.float().exp()
        for scale in self.scales:
            radius = (0.5 * scale * rho).clamp(min=0.5, max=self.max_radius)
            pair, valid = self._sample_pair(x, q.float(), radius)
            count = valid.sum(-1)
            mean = pair.sum(-1) / count.clamp(min=1)
            has_any = (count > 0).float()
            both = valid[..., 0] * valid[..., 1]
            difference = (pair[..., 0] - pair[..., 1]).abs() * both
            fields.extend(
                (
                    mean,
                    (x.float() - mean) * has_any,
                    difference,
                    count / 2,
                    (valid[..., 0] - valid[..., 1]).abs(),
                )
            )
        relation = self.project(torch.cat(fields, 1).to(x.dtype))
        relation = F.silu(self.norm(relation))
        return relation * foreground.to(x.dtype)


class GSSFFusion(nn.Module):
    """One fine/coarse fusion with a full-channel additive base path."""

    def __init__(
        self,
        channels,
        hidden_channels=64,
        d_state=16,
        dt_rank="auto",
        use_geometry=True,
        use_context=False,
        context_scales=(1.5, 3.0),
        context_max_radius=16.0,
        geometry_eta_init=0.1,
        residual_init=0.1,
        zero_init_residual=True,
        scan_backend="mamba_ssm",
        reference_max_length=256,
        num_scan_directions=4,
    ):
        super().__init__()
        self.fine_proj = nn.Conv2d(channels, hidden_channels, 1)
        self.coarse_proj = nn.Conv2d(channels, hidden_channels, 1)
        self.condition = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, 1),
            ChannelNorm(hidden_channels),
            nn.SiLU(),
        )
        self.fg_head = nn.Conv2d(hidden_channels, 1, 1)
        self.direction_head = nn.Conv2d(hidden_channels, 2, 1)
        self.width_head = nn.Conv2d(hidden_channels, 1, 1)
        self.scan = GeometryLineScan(
            hidden_channels,
            d_state=d_state,
            dt_rank=dt_rank,
            backend=scan_backend,
            reference_max_length=reference_max_length,
            geometry_eta_init=geometry_eta_init,
            use_geometry=use_geometry,
            num_scan_directions=num_scan_directions,
        )
        self.use_context = bool(use_context)
        if self.use_context:
            self.context = TransverseRelation(
                hidden_channels, context_scales, context_max_radius
            )
            self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.output = nn.Conv2d(hidden_channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(float(residual_init)))
        for conv in (self.fine_proj, self.coarse_proj, self.condition[0]):
            _init_conv(conv)
        for conv in (self.fg_head, self.direction_head, self.width_head):
            nn.init.normal_(conv.weight, std=0.001)
            nn.init.zeros_(conv.bias)
        nn.init.constant_(self.fg_head.bias, -2.0)
        with torch.no_grad():
            self.direction_head.bias[0] = 1.0
            self.width_head.bias.fill_(math.log(2.0))
        _init_conv(self.output, zero=zero_init_residual)

    def forward(self, fine, upsampled_coarse):
        x = self.fine_proj(fine)
        z = self.coarse_proj(upsampled_coarse)
        condition = self.condition(torch.cat((x, z), 1))
        fg_logits = self.fg_head(condition).float()
        q_raw = self.direction_head(condition).float()
        log_width = self.width_head(condition).float()
        q = normalize_direction(q_raw)
        foreground = fg_logits.sigmoid()
        # Auxiliary loss supervises unclamped raw log-width. Only the forward
        # condition is bounded, for safe exp/sampling at initialization.
        safe_width = log_width.clamp(math.log(0.25), math.log(256.0))
        if self.use_context:
            relation = self.context(x, q, safe_width, foreground)
            condition = condition + self.context_scale.to(x.dtype) * relation
        y = self.scan(x, condition, q, safe_width, foreground)
        output = fine + upsampled_coarse + self.gamma.to(fine.dtype) * self.output(y)
        predictions = dict(fg_logits=fg_logits, q_raw=q_raw, log_width=log_width)
        return output, predictions


class GSSFNeckCore(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        feature_strides=(4, 8, 16, 32),
        gssf_levels=(1, 2),
        hidden_channels=64,
        d_state=16,
        dt_rank="auto",
        use_geometry=True,
        use_context=False,
        context_scales=(1.5, 3.0),
        context_max_radius=16.0,
        use_arc=True,
        arc_hidden_channels=128,
        arc_kernel_number=2,
        arc_dropout=0.0,
        arc_angle_range=40.0,
        geometry_eta_init=0.1,
        residual_init=0.1,
        zero_init_residual=True,
        scan_backend="mamba_ssm",
        reference_max_length=256,
        upsample_cfg=None,
        add_extra_convs=False,
        num_scan_directions=4,
        fam_levels=(),
        fam_deform_groups=8,
        fam_backend="mmcv",
        fam_force_fp32=True,
        detail_levels=(),
        detail_hidden_channels=32,
        detail_beta_init=0.1,
        detail_learnable_beta=True,
        detail_zero_init=True,
        detail_fam_reference=True,
        detail_use_frequency=False,
        detail_frequency_kernel=3,
        detail_frequency_norm_eps=1e-5,
        da_fam_levels=(),
        dab_hidden_channels=32,
        dab_zero_init=True,
        fsm_levels=(),
    ):
        super().__init__()
        if (
            isinstance(num_scan_directions, bool)
            or not isinstance(num_scan_directions, int)
            or num_scan_directions not in (4, 8)
        ):
            raise ValueError("num_scan_directions must be integer 4 or 8.")
        self.num_scan_directions = num_scan_directions
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = len(in_channels) if end_level == -1 else end_level
        if (
            not in_channels
            or min(in_channels) < 1
            or out_channels < 1
            or hidden_channels < 1
            or arc_hidden_channels < 1
        ):
            raise ValueError("All channel counts must be positive.")
        if not 0 <= start_level < self.end_level <= len(in_channels):
            raise ValueError("Invalid start/end levels; end_level is exclusive.")
        if len(feature_strides) != len(in_channels) or min(feature_strides) <= 0:
            raise ValueError("feature_strides must match all backbone inputs.")
        if any(b <= a for a, b in zip(feature_strides, feature_strides[1:])):
            raise ValueError("feature_strides must be strictly increasing.")
        self.feature_strides = tuple(feature_strides[start_level : self.end_level])
        self.n_levels = self.end_level - start_level
        if num_outs < self.n_levels:
            raise ValueError("num_outs must cover selected backbone levels.")
        if add_extra_convs:
            raise ValueError(
                "This version uses pooled extra levels; set add_extra_convs=False."
            )
        levels = tuple(gssf_levels)
        if len(set(levels)) != len(levels) or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < self.n_levels - 1
            for i in levels
        ):
            raise ValueError(
                "gssf_levels must be unique fine-node indices before the deepest level."
            )
        self.gssf_levels = tuple(sorted(levels))
        self.use_geometry = bool(use_geometry)
        self.use_context = bool(use_context)
        if not math.isfinite(residual_init) or (
            zero_init_residual and residual_init == 0
        ):
            raise ValueError(
                "Use finite residual_init, nonzero when output projections start at zero."
            )
        self.upsample_cfg = deepcopy(upsample_cfg or dict(mode="nearest"))
        if "size" in self.upsample_cfg or "scale_factor" in self.upsample_cfg:
            raise ValueError(
                "Upsampling infers size from the fine layer; omit size/scale_factor."
            )
        selected_channels = in_channels[start_level : self.end_level]
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in selected_channels]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in selected_channels
            ]
        )
        for conv in list(self.lateral_convs) + list(self.output_convs):
            _init_conv(conv)
        self.arc_adapter = (
            ARCAdapter(
                out_channels,
                arc_hidden_channels,
                arc_kernel_number,
                arc_dropout,
                arc_angle_range,
                residual_init,
                zero_init_residual,
            )
            if use_arc
            else nn.Identity()
        )
        self.fusions = nn.ModuleDict(
            {
                str(i): GSSFFusion(
                    out_channels,
                    hidden_channels,
                    d_state,
                    dt_rank,
                    use_geometry,
                    use_context,
                    context_scales,
                    context_max_radius,
                    geometry_eta_init,
                    residual_init,
                    zero_init_residual,
                    scan_backend,
                    reference_max_length,
                    num_scan_directions=num_scan_directions,
                )
                for i in self.gssf_levels
            }
        )
        # Optional FAM acts on each selected upsampled coarse feature BEFORE
        # either GSSF or plain lateral addition. Empty levels preserve the old
        # initialization order, state_dict keys and forward path exactly.
        fam_levels = tuple(fam_levels)
        if len(set(fam_levels)) != len(fam_levels) or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < self.n_levels - 1
            for i in fam_levels
        ):
            raise ValueError(
                "fam_levels must be unique fine-node indices before the deepest level."
            )
        self.fam_levels = tuple(sorted(fam_levels))
        self.aligners = nn.ModuleDict()
        if self.fam_levels:
            from .fam import FAPNAlignment

            for i in self.fam_levels:
                self.aligners[str(i)] = FAPNAlignment(
                    out_channels,
                    deform_groups=fam_deform_groups,
                    backend=fam_backend,
                    force_fp32=fam_force_fp32,
                )
        # Append AFTER all pre-existing modules to preserve their parameters
        # and RNG order. Disabled adapters have no state_dict entries.
        detail_levels = tuple(detail_levels)
        if len(set(detail_levels)) != len(detail_levels) or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < self.n_levels - 1
            for i in detail_levels
        ):
            raise ValueError(
                "detail_levels must be unique fine-node indices "
                "before the deepest level (0=T2, 1=T3, 2=T4 "
                "when using C2-C5)."
            )
        if not isinstance(detail_fam_reference, bool):
            raise ValueError("detail_fam_reference must be a boolean.")
        if not isinstance(detail_use_frequency, bool):
            raise ValueError("detail_use_frequency must be a boolean.")
        self.detail_levels = tuple(sorted(detail_levels))
        self.detail_fam_reference = detail_fam_reference
        self.detail_use_frequency = detail_use_frequency
        self.detail_adapters = nn.ModuleDict(
            {
                str(i): LowLevelDetailAdapter(
                    out_channels,
                    hidden_channels=detail_hidden_channels,
                    beta_init=detail_beta_init,
                    learnable_beta=detail_learnable_beta,
                    zero_init=detail_zero_init,
                    use_frequency=detail_use_frequency,
                    frequency_kernel=detail_frequency_kernel,
                    frequency_norm_eps=detail_frequency_norm_eps,
                )
                for i in self.detail_levels
            }
        )

        # Add optional DA/FSM components only AFTER every existing component
        # has been initialized. In particular, a disabled option leaves old
        # state_dict keys, parameters and RNG consumption unchanged.
        da_fam_levels = tuple(da_fam_levels)
        if len(set(da_fam_levels)) != len(da_fam_levels) or any(
            isinstance(i, bool) or not isinstance(i, int) or i not in self.fam_levels
            for i in da_fam_levels
        ):
            raise ValueError("da_fam_levels must be unique indices in fam_levels.")
        fsm_levels = tuple(fsm_levels)
        if len(set(fsm_levels)) != len(fsm_levels) or any(
            isinstance(i, bool)
            or not isinstance(i, int)
            or not 0 <= i < self.n_levels - 1
            for i in fsm_levels
        ):
            raise ValueError(
                "fsm_levels must be unique fine-node indices "
                "before the deepest selected backbone level."
            )
        if (
            isinstance(dab_hidden_channels, bool)
            or not isinstance(dab_hidden_channels, int)
            or dab_hidden_channels < 1
        ):
            raise ValueError("dab_hidden_channels must be a positive integer.")
        if not isinstance(dab_zero_init, bool):
            raise TypeError("dab_zero_init must be a bool.")
        self.da_fam_levels = tuple(sorted(da_fam_levels))
        self.fsm_levels = tuple(sorted(fsm_levels))
        if self.da_fam_levels or self.fsm_levels:
            from .da_fam import DAFAPNAlignment, FeatureSelectionModule

            for i in self.da_fam_levels:
                original = self.aligners[str(i)]
                directional = DAFAPNAlignment(
                    out_channels,
                    deform_groups=fam_deform_groups,
                    backend=fam_backend,
                    force_fp32=fam_force_fp32,
                    dab_hidden_channels=dab_hidden_channels,
                    dab_zero_init=dab_zero_init,
                )
                # Preserve the exact existing FAM initialization. Only the
                # newly introduced directional branch has missing keys.
                directional.load_state_dict(original.state_dict(), strict=False)
                self.aligners[str(i)] = directional
            for i in self.fsm_levels:
                original = self.lateral_convs[i]
                selection = FeatureSelectionModule(selected_channels[i], out_channels)
                # Reuse the baseline projection weight. FSM replaces that
                # projection, rather than stacking another 256->256 layer.
                with torch.no_grad():
                    selection.conv.weight.copy_(original.weight)
                self.lateral_convs[i] = selection

    def forward(self, inputs, return_geometry=False):
        if len(inputs) != len(self.in_channels):
            raise ValueError("Backbone input count does not match in_channels.")
        if any(
            x.ndim != 4 or x.shape[1] != c for x, c in zip(inputs, self.in_channels)
        ):
            raise ValueError("Expected NCHW backbone inputs with configured channels.")
        selected = inputs[self.start_level : self.end_level]
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, selected)]
        states = [None] * self.n_levels
        states[-1] = self.arc_adapter(laterals[-1])
        predictions = []
        for i in range(self.n_levels - 2, -1, -1):
            original_fine = laterals[i]
            fine = (
                self.detail_adapters[str(i)](original_fine)
                if str(i) in self.detail_adapters
                else original_fine
            )
            coarse = F.interpolate(
                states[i + 1], size=fine.shape[-2:], **self.upsample_cfg
            )
            if str(i) in self.aligners:
                reference = fine if self.detail_fam_reference else original_fine
                coarse = self.aligners[str(i)](reference, coarse)
            if str(i) in self.fusions:
                states[i], prediction = self.fusions[str(i)](fine, coarse)
                prediction.update(level=i, stride=self.feature_strides[i])
                predictions.append(prediction)
            else:
                states[i] = fine + coarse
        outputs = [conv(x) for conv, x in zip(self.output_convs, states)]
        while len(outputs) < self.num_outs:
            outputs.append(F.max_pool2d(outputs[-1], kernel_size=1, stride=2))
        # Stable fine-to-coarse auxiliary order, independent of traversal order.
        predictions.sort(key=lambda item: item["level"])
        if return_geometry:
            return tuple(outputs), predictions
        return tuple(outputs)
