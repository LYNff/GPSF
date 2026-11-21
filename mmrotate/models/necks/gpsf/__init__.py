"""GPSF neck for MMRotate 0.x / MMDetection 2.x."""

from mmcv.runner import BaseModule, auto_fp16
from mmrotate.models.builder import ROTATED_NECKS

from .neck import GSSFNeckCore


@ROTATED_NECKS.register_module()
class GSSFNeck(BaseModule):
    """StripNet-compatible GPSF feature pyramid."""

    def __init__(self, in_channels, out_channels=256, num_outs=5,
                 start_level=0, end_level=-1, add_extra_convs=False,
                 upsample_cfg=None, conv_cfg=None, norm_cfg=None,
                 act_cfg=None, init_cfg=None, **gssf_cfg):
        if conv_cfg is not None or norm_cfg is not None or act_cfg is not None:
            raise ValueError(
                'GSSFNeck uses its own ChannelNorm; set conv/norm/act_cfg=None.')
        if init_cfg is not None:
            raise ValueError(
                'Leave GSSFNeck init_cfg=None to preserve its initializers.')
        super().__init__(init_cfg=None)
        self.fp16_enabled = False
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.core = GSSFNeckCore(
            in_channels, out_channels=out_channels, num_outs=num_outs,
            start_level=start_level, end_level=end_level,
            add_extra_convs=add_extra_convs, upsample_cfg=upsample_cfg,
            **gssf_cfg)

    @property
    def gssf_levels(self):
        return self.core.gssf_levels

    @auto_fp16()
    def forward(self, inputs, return_geometry=False):
        return self.core(inputs, return_geometry=return_geometry)


__all__ = ['GSSFNeck']
