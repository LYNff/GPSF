# GPSF for Strip R-CNN

GPSF replaces the standard FPN neck in Strip R-CNN with a bridge-oriented feature pyramid. It combines **GSSF** for geometry-conditioned feature propagation, **DAM** for direction-aware alignment during top-down fusion, and **RASA** for adaptive orientation processing of the deepest feature.

## Architecture

```text
StripNet: C2, C3, C4, C5
             │ 1×1 lateral projections
             ├── C5 ── RASA / ARC ──┐
             │                      ↓ upsample
             ├── C4 ── DAM ── GSSF (optional by level)
             │                      ↓ upsample
             ├── C3 ── DAM ── GSSF (optional by level)
             │                      ↓ upsample
             └── C2 ── DAM ── ordinary fusion when GSSF is disabled there
                                    ↓
                         standard pyramid outputs
                                    ↓
                         unchanged RPN and RoI head
```

The exact modules used at each fusion node are selected by configuration. The supplied template enables GSSF and DAM at the C3/C4 nodes and applies RASA to the deepest lateral feature.

### GSSF

At a selected fine-to-coarse fusion node, the low-level feature and upsampled high-level feature form a joint condition. Three prediction heads estimate foreground applicability, an unoriented direction, and short-side width. The predicted condition controls a four-route, line-independent selective state scan (horizontal and vertical, forward and reverse). Its output is added to the ordinary lateral fusion path.

Training also applies foreground, direction, and width auxiliary losses using rotated GT boxes. GT boxes build detached loss targets; they are not inputs to the feature-forward predictions. During inference the geometry is predicted from features.

### DAM

DAM aligns the upsampled coarse feature to the fine feature using a direction-aware condition and a modulated deformable convolution. In the source, the implementation retains the internal configuration names `fam_levels` and `da_fam_levels`; configure both, with `da_fam_levels` a subset of `fam_levels`.

### RASA

RASA is enabled by `use_arc=True`. In this implementation it uses an ARC adapter on the deepest projected lateral feature before top-down propagation. Its configuration names remain `use_arc` and `arc_*`.

## Source layout

After integrating GPSF into the MMRotate source tree, the relevant paths are:

```text
mmrotate/models/necks/gpsf/
├── __init__.py       # GSSFNeck registration and MMCV wrapper
├── neck.py           # GPSF pyramid and fusion nodes
├── scan.py           # four/eight-route selective state scan
├── geometry.py       # rotated-box targets and auxiliary losses
├── da_fam.py         # DAM directional alignment branch
├── fam.py            # modulated deformable alignment operator
├── arconv.py         # ARC operator used by RASA
└── detail.py         # optional detail adapter; disabled in the template

mmrotate/models/detectors/gssf_strip_rcnn.py  # StripRCNN + GSSF losses
mmrotate/models/necks/__init__.py             # registers GSSFNeck
mmrotate/models/detectors/__init__.py         # registers GSSFStripRCNN
```

The detector subclasses the existing `StripRCNN`. It preserves the standard RPN, RoI head, and inference path; the original `strip_rcnn.py` remains available for baseline runs. With the in-tree registration imports in place, the config does not need `custom_imports`.

## Configuration

Use [`configs/gpsf_model_override.py`](configs/gpsf_model_override.py) as the model override. In the Strip R-CNN repository, the corresponding template is `configs/strip_rcnn/strip_rcnn_s_gpsf_template.py`.

The template uses:

```python
model = dict(
    type='GSSFStripRCNN',
    bridge_label=0,
    gssf_loss_weights=dict(foreground=0.1, direction=0.1, width=0.05),
    gssf_target_cfg=dict(
        direction_min_aspect=1.2,
        max_boxes_per_chunk=16,
        max_elements=262144,
        thin_center_fallback=True),
    roi_head=dict(bbox_head=dict(num_classes=1)),
    neck=dict(
        _delete_=True,
        type='GSSFNeck',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5,
        feature_strides=(4, 8, 16, 32),
        gssf_levels=(1, 2),
        use_geometry=True,
        use_context=False,
        use_arc=True,
        num_scan_directions=4,
        fam_levels=(1, 2),
        da_fam_levels=(1, 2),
        fam_backend='mmcv',
        fam_force_fp32=True,
        # Other dimensions and initialization settings are in the template.
    ))
```

Level indices refer to the selected backbone inputs `[C2, C3, C4, C5]`:

| Index | Feature | Top-down fusion |
|---:|---|---|
| 0 | C2 / P2 | C3 → C2 |
| 1 | C3 / P3 | C4 → C3 |
| 2 | C4 / P4 | C5 → C4 |
| 3 | C5 / P5 | deepest feature; no coarser fusion |

Thus `gssf_levels=(1, 2)` enables GSSF at C3 and C4. The template also enables DAM at these two nodes. To enable DAM at all three top-down connections, set **both** `fam_levels` and `da_fam_levels` to `(0, 1, 2)`. GSSF cannot be enabled at index 3 because that node has no coarser feature to fuse.

For a one-class bridge dataset, use `bridge_label=0` and `num_classes=1`. Keep the dataset paths, split definitions, image pipeline, optimizer, schedule, checkpoint initialization, and evaluation settings from the actual MBDD or GLH config. Change the template's `_base_` to that dataset config before training; the template otherwise inherits the official DOTA example.

The nested neck uses `_delete_=True` to replace the FPN configuration cleanly. Keep `num_outs=5` and `add_extra_convs=False` for this implementation.

## Dependencies

- A compatible PyTorch / MMCV 1.x / MMDetection 2.x / MMRotate 0.x environment.
- `mamba_ssm` built for the active PyTorch and CUDA versions when `scan_backend='mamba_ssm'`.
- MMCV compiled with `mmcv.ops.modulated_deform_conv2d` when DAM uses `fam_backend='mmcv'`.

## Training

From the Strip R-CNN repository root, after setting the template base to the actual MBDD/GLH config:

```bash
python tools/train.py configs/strip_rcnn/strip_rcnn_s_gpsf_template.py \
  --work-dir work_dirs/gpsf_mbdd --seed 42
```

Change the config and work directory for GLH as needed. This README documents the in-tree integration; the actual data paths and experiment settings remain defined by your dataset config.

