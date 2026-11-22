# GPSF for Strip R-CNN

> **Status: Under Construction / Manuscript in Preparation for Submission**
>
> This repository hosts the official implementation of **GPSF** (*Geometry-Preserving Selective Fusion for Oriented Small Bridge Detection in Remote Sensing Images*). The paper is currently being prepared for submission, and this repository is still under active organization. Complete training/evaluation scripts, configuration files, and pretrained checkpoints will be released upon paper acceptance.

GPSF replaces the standard FPN neck in Strip R-CNN with a bridge-oriented feature pyramid. It combines **GSSF** for geometry-conditioned feature propagation, **DAM** for direction-aware alignment during top-down fusion, and **RASA** for adaptive orientation processing of the deepest feature.

## Architecture

```text
StripNet: C2, C3, C4, C5
             │ 1×1 lateral projections
             ├── C5 ── RASA / ARC ──┐
             │                      ↓ upsample
             ├── C4 ──   DAM ──   GSSF 
             │                      ↓ upsample
             ├── C3 ──   DAM ──   GSSF 
             │                      ↓ upsample
             └── C2 ──   DAM ── ordinary fusion when GSSF is disabled there
                                    ↓
                         standard pyramid outputs
                                    ↓
                         unchanged RPN and RoI head
```

<!-- ### GSSF

At a selected fine-to-coarse fusion node, the low-level feature and upsampled high-level feature form a joint condition. Three prediction heads estimate foreground applicability, an unoriented direction, and short-side width. The predicted condition controls a four-route, line-independent selective state scan (horizontal and vertical, forward and reverse). Its output is added to the ordinary lateral fusion path.

Training also applies foreground, direction, and width auxiliary losses using rotated GT boxes. GT boxes build detached loss targets; they are not inputs to the feature-forward predictions. During inference the geometry is predicted from features.

### DAM

DAM aligns the upsampled coarse feature to the fine feature using a direction-aware condition and a modulated deformable convolution. In the source, the implementation retains the internal configuration names `fam_levels` and `da_fam_levels`; configure both, with `da_fam_levels` a subset of `fam_levels`.

### RASA

RASA is enabled by `use_arc=True`. In this implementation it uses an ARC adapter on the deepest projected lateral feature before top-down propagation.  -->

## Dependencies

- A compatible PyTorch / MMCV 1.x / MMDetection 2.x / MMRotate 0.x environment.
- `mamba_ssm` built for the active PyTorch and CUDA versions when `scan_backend='mamba_ssm'`.
- MMCV compiled with `mmcv.ops.modulated_deform_conv2d` when DAM uses `fam_backend='mmcv'`.

## 待完善......