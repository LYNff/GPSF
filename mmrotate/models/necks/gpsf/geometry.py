import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
import torch.distributed as dist
from torch.nn import functional as F


def _real(value, name, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(
            "{} must be a finite {} number".format(
                name, "positive" if positive else "nonnegative"
            )
        )
    return float(value)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return int(value)


def _boxes(value, device):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != 5:
        raise ValueError("Each GT/ignore item must be an (N, 5) tensor")
    value = value.detach().to(device=device, dtype=torch.float32)
    valid = torch.isfinite(value).all(dim=1) & (value[:, 2] > 0) & (value[:, 3] > 0)
    return value, valid


def _properties(boxes):
    width, height, theta = boxes[:, 2], boxes[:, 3], boxes[:, 4]
    # In image coordinates increasing theta rotates the width axis clockwise.
    theta_long = theta + (height > width).to(theta.dtype) * (math.pi / 2)
    ux, uy = theta_long.cos(), theta_long.sin()
    short = torch.minimum(width, height)
    long = torch.maximum(width, height)
    q = torch.stack((ux.square() - uy.square(), 2 * ux * uy), dim=1)
    return ux, uy, short, long, q


def _pairwise_inside(boxes, x, y):
    ux, uy, short, long, _ = _properties(boxes)
    dx = x[None, :] - boxes[:, 0:1]
    dy = y[None, :] - boxes[:, 1:2]
    along = dx * ux[:, None] + dy * uy[:, None]
    across = -dx * uy[:, None] + dy * ux[:, None]
    return (along.abs() <= 0.5 * long[:, None]) & (across.abs() <= 0.5 * short[:, None])


@torch.no_grad()
def build_geometry_targets(
    gt_bboxes,
    img_metas,
    feature_size,
    stride,
    device=None,
    gt_labels=None,
    gt_bboxes_ignore=None,
    bridge_label=0,
    direction_min_aspect=1.2,
    max_boxes_per_chunk=16,
    max_elements=262144,
    thin_center_fallback=True,
):
    """Rasterize boxes at feature-cell centres, with explicit thin-GT fallback.

    Outputs: foreground (B1HW float), valid (B1HW bool), q (B2HW float),
    log_width (B1HW float), geometry_valid/direction_valid (B1HW bool).
    ``valid`` excludes padding and ignored OBBs.  Foreground is bridge OBB
    occupancy, including overlap; geometry is supervised only where exactly
    one bridge claims a cell.  Near-square boxes supervise width, not angle.

    A valid bridge whose full OBB contains no valid cell centre may claim the
    nearest nonignored valid cell to its centre (centre must be in the image),
    provided squared distance is at most stride**2 / 2 plus numeric tolerance.
    This bound prevents ignored nearest cells from moving a target far away.
    This is an explicit supervision convention for sub-stride targets, not
    an enlargement of the box or a claim that the cell is bridge segmentation.
    Fallback collisions with other bridges are excluded from geometry losses.
    Pairwise box-point temporaries are bounded by ``max_elements``.
    """
    if (
        not isinstance(gt_bboxes, (list, tuple))
        or not gt_bboxes
        or len(gt_bboxes) != len(img_metas)
    ):
        raise ValueError("GT and img_metas need the same nonzero batch size")
    batch = len(gt_bboxes)
    for name, value in (
        ("gt_labels", gt_labels),
        ("gt_bboxes_ignore", gt_bboxes_ignore),
    ):
        if value is not None and len(value) != batch:
            raise ValueError("{} needs one item per image".format(name))
    if len(feature_size) != 2:
        raise ValueError("feature_size must be (height, width)")
    height = _integer(feature_size[0], "feature height")
    width = _integer(feature_size[1], "feature width")
    stride = _real(stride, "stride", positive=True)
    bridge_label = _integer(bridge_label, "bridge_label", minimum=0)
    direction_min_aspect = _real(
        direction_min_aspect, "direction_min_aspect", positive=True
    )
    if direction_min_aspect < 1:
        raise ValueError("direction_min_aspect must be >= 1")
    box_limit = _integer(max_boxes_per_chunk, "max_boxes_per_chunk")
    element_limit = _integer(max_elements, "max_elements")
    if not isinstance(thin_center_fallback, bool):
        raise ValueError("thin_center_fallback must be bool")
    if device is None:
        if not isinstance(gt_bboxes[0], torch.Tensor):
            raise ValueError("GT entries must be tensors")
        device = gt_bboxes[0].device
    x = (
        ((torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride)
        .view(1, width)
        .expand(height, width)
        .reshape(-1)
    )
    y = (
        ((torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride)
        .view(height, 1)
        .expand(height, width)
        .reshape(-1)
    )
    output = {
        key: []
        for key in (
            "foreground",
            "valid",
            "q",
            "log_width",
            "geometry_valid",
            "direction_valid",
        )
    }
    for index, (raw_boxes, meta) in enumerate(zip(gt_bboxes, img_metas)):
        if not isinstance(meta, Mapping) or "img_shape" not in meta:
            raise ValueError("Each img_meta must contain img_shape")
        if len(meta["img_shape"]) < 2:
            raise ValueError("img_shape needs height and width")
        image_h = _real(meta["img_shape"][0], "image height")
        image_w = _real(meta["img_shape"][1], "image width")
        image_valid = (x < image_w) & (y < image_h)
        valid = image_valid.clone()
        ignored = torch.zeros_like(valid)
        ignore_boxes = torch.empty((0, 5), device=device)
        if gt_bboxes_ignore is not None and gt_bboxes_ignore[index] is not None:
            ignore_boxes, good = _boxes(gt_bboxes_ignore[index], device)
            ignore_boxes = ignore_boxes[good]
            bc = min(box_limit, element_limit)
            pc = max(1, element_limit // bc)
            for b0 in range(0, ignore_boxes.shape[0], bc):
                for p0 in range(0, x.numel(), pc):
                    ps = slice(p0, min(p0 + pc, x.numel()))
                    hit = _pairwise_inside(ignore_boxes[b0 : b0 + bc], x[ps], y[ps])
                    ignored[ps] |= hit.any(dim=0)
        valid &= ~ignored
        boxes, good = _boxes(raw_boxes, device)
        if gt_labels is not None:
            labels = gt_labels[index]
            integer_types = (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
            if (
                not isinstance(labels, torch.Tensor)
                or labels.ndim != 1
                or labels.shape[0] != boxes.shape[0]
                or labels.dtype not in integer_types
            ):
                raise ValueError("Each gt_labels item needs N integer labels")
            good &= labels.detach().to(device=device) == bridge_label
        boxes = boxes[good]
        _, _, short, long, q = _properties(boxes)
        box_log_width = (short / stride).clamp_min(1e-12).log()
        stable_direction = long >= direction_min_aspect * short
        count = torch.zeros_like(x, dtype=torch.int32)
        q_target = torch.zeros((2, x.numel()), device=device)
        width_target = torch.zeros_like(x)
        direction_count = torch.zeros_like(count)
        seen = torch.zeros(boxes.shape[0], device=device, dtype=torch.bool)
        bc = min(box_limit, element_limit)
        pc = max(1, element_limit // bc)
        for b0 in range(0, boxes.shape[0], bc):
            bs = slice(b0, min(b0 + bc, boxes.shape[0]))
            for p0 in range(0, x.numel(), pc):
                ps = slice(p0, min(p0 + pc, x.numel()))
                raw_hit = _pairwise_inside(boxes[bs], x[ps], y[ps])
                # An OBB covered by ignored cells must not reappear outside
                # the ignored region via the thin-object fallback.
                seen[bs] |= (raw_hit & image_valid[None, ps]).any(dim=1)
                hit = raw_hit & valid[None, ps]
                hit_float = hit.float()
                count[ps] += hit.sum(dim=0).to(count.dtype)
                q_target[:, ps] += q[bs].t().matmul(hit_float)
                width_target[ps] += (box_log_width[bs, None] * hit_float).sum(0)
                direction_count[ps] += (hit & stable_direction[bs, None]).sum(0)
        if thin_center_fallback:
            eligible = (
                ~seen
                & (boxes[:, 0] >= 0)
                & (boxes[:, 0] < image_w)
                & (boxes[:, 1] >= 0)
                & (boxes[:, 1] < image_h)
            )
            for b0 in range(0, ignore_boxes.shape[0], bc):
                for p0 in range(0, boxes.shape[0], pc):
                    ps = slice(p0, min(p0 + pc, boxes.shape[0]))
                    center_ignored = _pairwise_inside(
                        ignore_boxes[b0 : b0 + bc], boxes[ps, 0], boxes[ps, 1]
                    )
                    eligible[ps] &= ~center_ignored.any(dim=0)
            fallback_boxes = torch.nonzero(eligible, as_tuple=False).flatten()
            # Each box's nearest valid point is found with bounded temporaries.
            for b0 in range(0, fallback_boxes.numel(), bc):
                ids = fallback_boxes[b0 : b0 + bc]
                best = x.new_full((ids.numel(),), float("inf"))
                best_index = torch.zeros(ids.numel(), device=device, dtype=torch.long)
                for p0 in range(0, x.numel(), pc):
                    ps = slice(p0, min(p0 + pc, x.numel()))
                    distance = (boxes[ids, 0:1] - x[None, ps]).square() + (
                        boxes[ids, 1:2] - y[None, ps]
                    ).square()
                    distance.masked_fill_(~valid[None, ps], float("inf"))
                    local_best, local_index = distance.min(dim=1)
                    improve = local_best < best
                    best_index = torch.where(improve, local_index + p0, best_index)
                    best = torch.minimum(best, local_best)
                # The nearest centre of an unconstrained square grid is at
                # most stride/sqrt(2) away. Ignored/padded neighbours must
                # not send a sub-stride target to a remote valid location.
                max_distance = 0.5 * stride * stride
                tolerance = max(1e-6, max_distance * 1e-6)
                usable = torch.isfinite(best) & (best <= max_distance + tolerance)
                ids, points = ids[usable], best_index[usable]
                count.index_add_(0, points, torch.ones_like(points, dtype=count.dtype))
                q_target.index_add_(1, points, q[ids].t())
                width_target.index_add_(0, points, box_log_width[ids])
                direction_count.index_add_(
                    0, points, stable_direction[ids].to(count.dtype)
                )
        geometry_valid = valid & (count == 1)
        direction_valid = geometry_valid & (direction_count == 1)
        q_target.masked_fill_(~direction_valid[None, :], 0)
        width_target.masked_fill_(~geometry_valid, 0)
        values = {
            "foreground": (count > 0).float().reshape(1, height, width),
            "valid": valid.reshape(1, height, width),
            "q": q_target.reshape(2, height, width),
            "log_width": width_target.reshape(1, height, width),
            "geometry_valid": geometry_valid.reshape(1, height, width),
            "direction_valid": direction_valid.reshape(1, height, width),
        }
        for key, value in values.items():
            output[key].append(value)
    return {key: torch.stack(values) for key, values in output.items()}


def _level_weights(number, level_weights):
    if level_weights is None:
        values = [1.0] * number
    elif isinstance(level_weights, Mapping):
        if set(level_weights) != set(range(number)) or any(
            isinstance(k, bool) or not isinstance(k, Integral) for k in level_weights
        ):
            raise ValueError("level_weights mapping must use prediction-list indices")
        values = [level_weights[i] for i in range(number)]
    else:
        if isinstance(level_weights, (str, bytes)):
            raise ValueError("level_weights must be a sequence or mapping")
        try:
            values = list(level_weights)
        except TypeError as error:
            raise ValueError("level_weights must be a sequence or mapping") from error
        if len(values) != number:
            raise ValueError("One level weight is required per prediction")
    values = [_real(v, "level weight") for v in values]
    if sum(values) <= 0:
        raise ValueError("At least one level weight must be positive")
    return values


def multilevel_geometry_loss(
    predictions,
    img_metas,
    gt_bboxes,
    gt_labels,
    gt_bboxes_ignore=None,
    bridge_label=0,
    foreground_weight=0.1,
    direction_weight=0.1,
    width_weight=0.05,
    level_weights=None,
    target_cfg=None,
):
    if not isinstance(predictions, (list, tuple)) or not predictions:
        raise ValueError("predictions must be a nonempty list/tuple")
    task_weights = [
        _real(v, n)
        for v, n in (
            (foreground_weight, "foreground_weight"),
            (direction_weight, "direction_weight"),
            (width_weight, "width_weight"),
        )
    ]
    weights = _level_weights(len(predictions), level_weights)
    if target_cfg is not None and not isinstance(target_cfg, Mapping):
        raise ValueError("target_cfg must be a mapping")
    cfg = {} if target_cfg is None else dict(target_cfg)
    allowed = {
        "direction_min_aspect",
        "max_boxes_per_chunk",
        "max_elements",
        "thin_center_fallback",
    }
    if set(cfg) - allowed:
        raise ValueError(
            "Unknown geometry target settings: {}".format(sorted(set(cfg) - allowed))
        )
    numerators, counts, device = [], [], None
    for prediction in predictions:
        required = {"fg_logits", "q_raw", "log_width", "stride"}
        if not isinstance(prediction, Mapping) or not required.issubset(prediction):
            raise ValueError("Each prediction needs fg_logits/q_raw/log_width/stride")
        fg, q, width = (prediction[k] for k in ("fg_logits", "q_raw", "log_width"))
        for value, channels, name in (
            (fg, 1, "fg_logits"),
            (q, 2, "q_raw"),
            (width, 1, "log_width"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 4
                or value.shape[1] != channels
                or not value.is_floating_point()
            ):
                raise ValueError("{} needs floating Bx{}xHxW".format(name, channels))
            if (
                value.shape[0] != fg.shape[0]
                or value.shape[-2:] != fg.shape[-2:]
                or value.device != fg.device
            ):
                raise ValueError("Prediction shapes/devices differ")
        if device is None:
            device = fg.device
        if fg.device != device:
            raise ValueError("All prediction levels need the same device")
        target = build_geometry_targets(
            gt_bboxes,
            img_metas,
            fg.shape[-2:],
            prediction["stride"],
            device=device,
            gt_labels=gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            bridge_label=bridge_label,
            **cfg,
        )
        if target["foreground"].shape != fg.shape:
            raise ValueError("Prediction batch size differs from GT/img_metas")
        fg_error = F.binary_cross_entropy_with_logits(
            fg.float(), target["foreground"], reduction="none"
        )
        direction_error = F.smooth_l1_loss(
            q.float(), target["q"], reduction="none"
        ).mean(1, keepdim=True)
        width_error = F.smooth_l1_loss(
            width.float(), target["log_width"], reduction="none"
        )
        pos = target["valid"] & (target["foreground"] > 0)
        neg = target["valid"] & ~pos
        masks = (pos, neg, target["direction_valid"], target["geometry_valid"])
        errors = (fg_error, fg_error, direction_error, width_error)
        numerators.append(
            torch.stack(
                [
                    torch.where(mask, error, torch.zeros_like(error)).sum()
                    for mask, error in zip(masks, errors)
                ]
            )
        )
        counts.append(torch.stack([mask.sum() for mask in masks]).float())
    numerator = torch.stack(numerators)
    count = torch.stack(counts).detach()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        count /= dist.get_world_size()
    active = count > 0
    # Counts can be fractional after DDP averaging: do not clamp positive
    # counts to one, or a rank with fewer positives would be underweighted.
    means = numerator / torch.where(active, count, torch.ones_like(count))
    fg_per_level = means[:, :2].sum(1) / active[:, :2].sum(1).clamp_min(1)
    values = torch.stack((fg_per_level, means[:, 2], means[:, 3]), dim=1)
    task_active = torch.stack((active[:, :2].any(1), active[:, 2], active[:, 3]), 1)
    level = count.new_tensor(weights)[:, None]
    denominator = (level * task_active.float()).sum(0)
    total = (level * values).sum(0) / denominator.clamp_min(1e-12)
    return {
        "loss_gssf_fg": total[0] * task_weights[0],
        "loss_gssf_dir": total[1] * task_weights[1],
        "loss_gssf_width": total[2] * task_weights[2],
    }
