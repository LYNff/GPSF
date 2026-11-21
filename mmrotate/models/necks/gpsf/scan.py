import math

import torch
from torch import nn
from torch.nn import functional as F


def selective_scan_reference(
    u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True
):
    if u.ndim != 3 or delta.shape != u.shape or A.ndim != 2:
        raise ValueError("Expected u/delta (B,D,L) and A (D,N).")
    if A.is_complex() or B.is_complex() or C.is_complex():
        raise ValueError("Only real-valued selective states are supported.")
    if B.ndim == 3:
        B = B.unsqueeze(1)
    if C.ndim == 3:
        C = C.unsqueeze(1)
    if B.ndim != 4 or C.shape != B.shape:
        raise ValueError("B and C need equal (B,G,N,L) shapes.")
    batch, channels, length = u.shape
    groups, states = B.shape[1:3]
    if (
        groups < 1
        or channels % groups != 0
        or A.shape != (channels, states)
        or B.shape[0] != batch
        or B.shape[-1] != length
        or length < 1
    ):
        raise ValueError("Incompatible selective scan tensor shapes.")
    if D is not None and D.shape != (channels,):
        raise ValueError("D must have shape (channels,).")
    if delta_bias is not None and delta_bias.shape != (channels,):
        raise ValueError("delta_bias must have shape (channels,).")
    u, delta, A, B, C = (value.float() for value in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, channels, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    per_group = channels // groups
    u_group = u.reshape(batch, groups, per_group, length)
    dt_group = delta.reshape(batch, groups, per_group, length)
    a_group = A.reshape(groups, per_group, states)
    hidden = u.new_zeros(batch, groups, per_group, states)
    result = []
    for index in range(length):
        dt = dt_group[..., index].unsqueeze(-1)
        hidden = torch.exp(dt * a_group.unsqueeze(0)) * hidden + dt * B[
            ..., index
        ].unsqueeze(2) * u_group[..., index].unsqueeze(-1)
        result.append((hidden * C[..., index].unsqueeze(2)).sum(-1))
    output = torch.stack(result, dim=-1).reshape(batch, channels, length)
    if D is not None:
        output = output + u * D.float().view(1, channels, 1)
    return output


def _load_mamba_scan():
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "backend='mamba_ssm' needs a working CUDA build of mamba-ssm "
            "matched to your Python/PyTorch/CUDA versions. The module will "
            "not silently use a slow fallback. backend='reference' is for "
            "small correctness tests only."
        ) from error
    return selective_scan_fn


class GeometryLineScan(nn.Module):
    def __init__(
        self,
        channels,
        d_state=16,
        dt_rank="auto",
        backend="mamba_ssm",
        reference_max_length=256,
        geometry_eta_init=0.1,
        use_geometry=True,
        num_scan_directions=4,
    ):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer.")
        if not isinstance(d_state, int) or not 1 <= d_state <= 256:
            raise ValueError("d_state must be in [1, 256].")
        if backend not in ("mamba_ssm", "reference"):
            raise ValueError("backend must be 'mamba_ssm' or 'reference'.")
        if not math.isfinite(float(geometry_eta_init)) or float(geometry_eta_init) < 0:
            raise ValueError("geometry_eta_init must be finite and nonnegative.")
        if int(reference_max_length) < 1:
            raise ValueError("reference_max_length must be positive.")
        if (
            isinstance(num_scan_directions, bool)
            or not isinstance(num_scan_directions, int)
            or num_scan_directions not in (4, 8)
        ):
            raise ValueError("num_scan_directions must be integer 4 or 8.")
        self.num_scan_directions = num_scan_directions
        self.channels = channels
        self.d_state = d_state
        self.dt_rank = (
            int(math.ceil(channels / 16.0)) if dt_rank == "auto" else int(dt_rank)
        )
        if self.dt_rank < 1:
            raise ValueError("dt_rank must be positive or auto.")
        self.backend = backend
        self.reference_max_length = int(reference_max_length)
        self.use_geometry = bool(use_geometry)
        self._scan_fn = None

        parameter_heads = [
            nn.Linear(channels, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(num_scan_directions)
        ]
        self.condition_proj_weight = nn.Parameter(
            torch.stack([head.weight for head in parameter_heads], dim=0)
        )
        dt_heads = [
            self._make_dt_projection(self.dt_rank, channels)
            for _ in range(num_scan_directions)
        ]
        self.dt_proj_weight = nn.Parameter(
            torch.stack([head.weight for head in dt_heads], dim=0)
        )
        self.dt_proj_bias = nn.Parameter(
            torch.stack([head.bias for head in dt_heads], dim=0)
        )
        self.dt_proj_bias._no_reinit = True
        a_values = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_logs = nn.Parameter(
            a_values.log().view(1, 1, -1).repeat(num_scan_directions, channels, 1)
        )
        self.Ds = nn.Parameter(torch.ones(num_scan_directions, channels))
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True
        # softplus cannot represent exact zero with a finite, trainable logit.
        # use_geometry=False is the exact no-geometry ablation; eta_init=0 is
        # represented by a tiny positive 1e-8 rather than a dead -inf parameter.
        eta_initial = max(float(geometry_eta_init), 1e-8)
        eta_raw = eta_initial + math.log(-math.expm1(-eta_initial))
        self.eta_raw = nn.Parameter(torch.tensor(eta_raw, dtype=torch.float32))
        # The no-geometry ablation still has the same checkpoint structure, but
        # eta is not used by its forward graph. Freeze it so DDP does not see an
        # unused trainable parameter when find_unused_parameters=False.
        self.eta_raw.requires_grad_(self.use_geometry)
        self.eta_raw._no_weight_decay = True
        self.out_norm = nn.LayerNorm(channels, eps=1e-5)

    @property
    def geometry_eta(self):
        return F.softplus(self.eta_raw.float())

    @staticmethod
    def _make_dt_projection(rank, channels):
        head = nn.Linear(rank, channels, bias=True)
        bound = rank**-0.5
        nn.init.uniform_(head.weight, -bound, bound)
        dt = torch.exp(
            torch.rand(channels) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            head.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return head

    @staticmethod
    def _diagonal_layout(height, width, device, axis):
        if axis not in ("diagonal45", "diagonal135"):
            raise ValueError("Unknown diagonal axis: {}".format(axis))
        length = min(height, width)
        offsets = torch.arange(1 - width, height, device=device)
        start_y = offsets.clamp(min=0)
        start_x = (-offsets).clamp(min=0)
        lengths = torch.minimum(height - start_y, width - start_x)
        steps = torch.arange(length, device=device)
        valid = steps.unsqueeze(0) < lengths.unsqueeze(1)
        ys = start_y.unsqueeze(1) + steps
        xs = start_x.unsqueeze(1) + steps
        if axis == "diagonal135":
            xs = width - 1 - xs
        # Invalid entries gather pixel zero temporarily, then are zeroed by
        # the validity mask before they can enter the recurrence.
        indices = (ys * width + xs).masked_fill(~valid, 0).long()
        flat = torch.arange(height * width, device=device)
        pixel_y = flat // width
        pixel_x = flat % width
        if axis == "diagonal135":
            pixel_x = width - 1 - pixel_x
        line = pixel_y - pixel_x + width - 1
        step = torch.minimum(pixel_y, pixel_x)
        inverse = (line * length + step).long()
        return indices, valid, inverse

    @staticmethod
    def _pack_lines(value, axis, layout=None):
        """Each physical line becomes its own independent batch entry."""
        batch, channels, height, width = value.shape
        if axis == "horizontal":
            return (
                value.permute(0, 2, 1, 3)
                .reshape(batch * height, channels, width)
                .contiguous()
            )
        if axis == "vertical":
            return (
                value.permute(0, 3, 1, 2)
                .reshape(batch * width, channels, height)
                .contiguous()
            )
        if axis in ("diagonal45", "diagonal135"):
            if layout is None:
                layout = GeometryLineScan._diagonal_layout(
                    height, width, value.device, axis
                )
            indices, valid, _ = layout
            line_count, length = indices.shape
            packed = value.flatten(2).index_select(2, indices.reshape(-1))
            packed = packed.reshape(batch, channels, line_count, length)
            packed = packed.masked_fill(~valid[None, None], 0)
            return (
                packed.permute(0, 2, 1, 3)
                .reshape(batch * line_count, channels, length)
                .contiguous()
            )
        raise ValueError("Unknown scan axis: {}".format(axis))

    @staticmethod
    def _restore_lines(value, axis, batch, height, width, layout=None):
        channels = value.shape[1]
        if axis == "horizontal":
            return (
                value.reshape(batch, height, channels, width)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        if axis == "vertical":
            return (
                value.reshape(batch, width, channels, height)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
        if axis in ("diagonal45", "diagonal135"):
            if layout is None:
                layout = GeometryLineScan._diagonal_layout(
                    height, width, value.device, axis
                )
            indices, _, inverse = layout
            line_count, length = indices.shape
            packed = value.reshape(batch, line_count, channels, length)
            packed = packed.permute(0, 2, 1, 3).reshape(batch, channels, -1)
            # Inverse gather selects only real pixels, so padding cannot
            # overwrite a valid coordinate (as naive scatter can).
            return (
                packed.index_select(2, inverse)
                .reshape(batch, channels, height, width)
                .contiguous()
            )
        raise ValueError("Unknown scan axis: {}".format(axis))

    def _geometry_bias(self, q, log_width, foreground, axis):
        if not self.use_geometry:
            return torch.zeros_like(log_width, dtype=torch.float32)
        q_cos = q[:, :1].float().clamp(-1.0, 1.0)
        if axis == "horizontal":
            mismatch = (1.0 - q_cos) * 0.5
        elif axis == "vertical":
            mismatch = (1.0 + q_cos) * 0.5
        elif axis == "diagonal45":
            mismatch = (1.0 - q[:, 1:2].float().clamp(-1.0, 1.0)) * 0.5
        elif axis == "diagonal135":
            mismatch = (1.0 + q[:, 1:2].float().clamp(-1.0, 1.0)) * 0.5
        else:
            raise ValueError("Unknown scan axis: {}".format(axis))
        # sigmoid(-log_width) is stable even for extreme predicted log widths.
        return (
            self.geometry_eta
            * foreground.float()
            * mismatch
            * torch.sigmoid(-log_width.float())
        )

    def _selective_scan(self, xs, delta_logits, A, Bs, Cs, Ds, dt_bias):
        if self.backend == "reference":
            if xs.shape[-1] > self.reference_max_length:
                raise RuntimeError(
                    "Reference line length {} exceeds test-only limit {}. "
                    "Use backend=mamba_ssm on CUDA for detector training.".format(
                        xs.shape[-1], self.reference_max_length
                    )
                )
            return selective_scan_reference(
                xs, delta_logits, A, Bs, Cs, Ds, dt_bias, True
            )
        if not xs.is_cuda:
            raise RuntimeError(
                "backend='mamba_ssm' requires CUDA tensors. Explicitly use "
                "backend='reference' only for small CPU correctness tests."
            )
        if self._scan_fn is None:
            self._scan_fn = _load_mamba_scan()
        return self._scan_fn(
            xs,
            delta_logits,
            A,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        )

    def _scan_axis(self, x, condition, geometry_bias, axis):
        batch, channels, height, width = x.shape
        route_starts = dict(horizontal=0, vertical=2, diagonal45=4, diagonal135=6)
        if axis not in route_starts or route_starts[axis] >= self.num_scan_directions:
            raise ValueError(
                "Axis {} unavailable with {} scan directions.".format(
                    axis, self.num_scan_directions
                )
            )
        layout = (
            self._diagonal_layout(height, width, x.device, axis)
            if axis.startswith("diagonal")
            else None
        )
        x_lines = self._pack_lines(x, axis, layout)
        condition_lines = self._pack_lines(condition, axis, layout)
        geometry_lines = self._pack_lines(geometry_bias, axis, layout)
        # Reversing the padded sequence moves zero padding to the front.
        # Every independent line starts at zero state and its padded x is
        # strictly zero, so these leading positions cannot write any state.
        xs = torch.stack((x_lines, x_lines.flip(-1)), dim=1)
        conditions = torch.stack((condition_lines, condition_lines.flip(-1)), dim=1)
        biases = torch.stack((geometry_lines, geometry_lines.flip(-1)), dim=1)
        route_slice = slice(route_starts[axis], route_starts[axis] + 2)
        # Parameter generation also runs in fp32. An enclosing AMP context can
        # autocast einsum, so forward() disables autocast for this whole path.
        projected = torch.einsum(
            "bkcl,kpc->bkpl",
            conditions.float(),
            self.condition_proj_weight[route_slice].float(),
        )
        low_dt, Bs, Cs = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dt = (
            torch.einsum(
                "bkrl,kcr->bkcl", low_dt, self.dt_proj_weight[route_slice].float()
            )
            + biases
        )
        line_batch, _, _, length = xs.shape
        result = self._selective_scan(
            xs.reshape(line_batch, 2 * channels, length).float().contiguous(),
            dt.reshape(line_batch, 2 * channels, length).float().contiguous(),
            -self.A_logs[route_slice].float().exp().reshape(2 * channels, self.d_state),
            Bs.float().contiguous(),
            Cs.float().contiguous(),
            self.Ds[route_slice].float().reshape(2 * channels),
            self.dt_proj_bias[route_slice].float().reshape(2 * channels),
        )
        result = result.reshape(line_batch, 2, channels, length)
        forward = self._restore_lines(result[:, 0], axis, batch, height, width, layout)
        reverse = self._restore_lines(
            result[:, 1].flip(-1), axis, batch, height, width, layout
        )
        return forward, reverse

    def forward(self, x, condition, q, log_width, foreground):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("x must be BCHW with {} channels.".format(self.channels))
        if condition.shape != x.shape:
            raise ValueError("condition must have the same shape as x.")
        batch, _, height, width = x.shape
        if min(height, width) < 1:
            raise ValueError("Spatial dimensions must be nonempty.")
        for name, value, channels in (
            ("q", q, 2),
            ("log_width", log_width, 1),
            ("foreground", foreground, 1),
        ):
            if value.shape != (batch, channels, height, width):
                raise ValueError("{} has an incompatible shape.".format(name))
            if value.device != x.device:
                raise ValueError("{} must be on x.device.".format(name))
        if condition.device != x.device:
            raise ValueError("condition must be on x.device.")
        if (
            self.backend == "reference"
            and max(height, width) > self.reference_max_length
        ):
            raise RuntimeError(
                "Reference line length {} exceeds test-only limit {}.".format(
                    max(height, width), self.reference_max_length
                )
            )
        dtype_in = x.dtype
        # New PyTorch supports the common CPU/CUDA API; older MMRotate stacks
        # use torch.cuda.amp. Either context prevents AMP downcasting einsum.
        if hasattr(torch, "autocast"):
            autocast_context = torch.autocast(device_type=x.device.type, enabled=False)
        else:
            autocast_context = torch.cuda.amp.autocast(enabled=False)
        with autocast_context:
            horizontal_bias = self._geometry_bias(
                q, log_width, foreground, "horizontal"
            )
            vertical_bias = self._geometry_bias(q, log_width, foreground, "vertical")
            row_forward, row_reverse = self._scan_axis(
                x.float(), condition.float(), horizontal_bias, "horizontal"
            )
            col_forward, col_reverse = self._scan_axis(
                x.float(), condition.float(), vertical_bias, "vertical"
            )
            if self.num_scan_directions == 4:
                # Preserve the exact original arithmetic and state_dict for
                # existing four-route checkpoints and controlled comparisons.
                merged = (row_forward + row_reverse + col_forward + col_reverse) * 0.25
            else:
                diagonal45_bias = self._geometry_bias(
                    q, log_width, foreground, "diagonal45"
                )
                diagonal135_bias = self._geometry_bias(
                    q, log_width, foreground, "diagonal135"
                )
                d45_forward, d45_reverse = self._scan_axis(
                    x.float(), condition.float(), diagonal45_bias, "diagonal45"
                )
                d135_forward, d135_reverse = self._scan_axis(
                    x.float(), condition.float(), diagonal135_bias, "diagonal135"
                )
                merged = (
                    row_forward
                    + row_reverse
                    + col_forward
                    + col_reverse
                    + d45_forward
                    + d45_reverse
                    + d135_forward
                    + d135_reverse
                ) * 0.125
            output = F.layer_norm(
                merged.permute(0, 2, 3, 1),
                self.out_norm.normalized_shape,
                self.out_norm.weight.float(),
                self.out_norm.bias.float(),
                self.out_norm.eps,
            )
        return output.permute(0, 3, 1, 2).contiguous().to(dtype=dtype_in)

    def extra_repr(self):
        return (
            "channels={}, d_state={}, dt_rank={}, backend={!r}, "
            "use_geometry={}, num_scan_directions={}"
        ).format(
            self.channels,
            self.d_state,
            self.dt_rank,
            self.backend,
            self.use_geometry,
            self.num_scan_directions,
        )
