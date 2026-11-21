"""Auxiliary-loss integration; the original RPN/RoI paths are preserved."""

import copy
import math

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.strip_rcnn import StripRCNN

from ..necks.gpsf.geometry import multilevel_geometry_loss


def _loss_weights(value):
    result = dict(foreground=0.1, direction=0.1, width=0.05)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('gssf_loss_weights supports foreground/direction/width only.')
        result.update(value)
    for key, weight in result.items():
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError('gssf_loss_weights values must be finite and nonnegative.')
        result[key] = float(weight)
    return result


@ROTATED_DETECTORS.register_module()
class GSSFStripRCNN(StripRCNN):
    """StripRCNN with explicit geometry losses, using one backbone/neck pass.

    Forward features always use predicted m/q/w, both in training and testing.
    GT OBBs are supplied only to the detached target builder after extraction.
    simple_test/aug_test/extract_feat are inherited from the original model.
    """

    def __init__(self, gssf_loss_weights=None, gssf_level_weights=None,
                 gssf_target_cfg=None, bridge_label=0, **kwargs):
        self.gssf_loss_weights = _loss_weights(gssf_loss_weights)
        if isinstance(bridge_label, bool) or not isinstance(bridge_label, int) or bridge_label < 0:
            raise ValueError('bridge_label must be a nonnegative integer class index.')
        self.bridge_label = bridge_label
        self.gssf_level_weights = copy.deepcopy(gssf_level_weights)
        self.gssf_target_cfg = copy.deepcopy(gssf_target_cfg)
        # Consume custom arguments here, never forward them to StripRCNN.
        super().__init__(**kwargs)
        from ..necks.gpsf import GSSFNeck
        if not self.with_neck or not isinstance(self.neck, GSSFNeck):
            raise TypeError('GSSFStripRCNN requires neck.type="GSSFNeck".')
        if not self.neck.gssf_levels:
            raise ValueError('GSSFStripRCNN requires at least one GSSF fusion node.')
        core = self.neck.core
        if ((core.use_geometry or core.use_context)
                and any(weight <= 0 for weight in self.gssf_loss_weights.values())):
            raise ValueError('Predicted geometry/context requires positive foreground, '
                             'direction and width loss weights. For the no-aux ablation '
                             'disable both use_geometry and use_context.')
        head = getattr(self.roi_head, 'bbox_head', None)
        classes = getattr(head, 'num_classes', None)
        if isinstance(classes, int) and bridge_label >= classes:
            raise ValueError('bridge_label is outside bbox_head.num_classes.')

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None,
                      **kwargs):
        backbone_features = self.backbone(img)
        features, predictions = self.neck(backbone_features, return_geometry=True)
        weights = self.gssf_loss_weights
        losses = multilevel_geometry_loss(
            predictions, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore, bridge_label=self.bridge_label,
            foreground_weight=weights['foreground'],
            direction_weight=weights['direction'], width_weight=weights['width'],
            level_weights=self.gssf_level_weights,
            target_cfg=self.gssf_target_cfg)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                features, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(
            features, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        return losses
