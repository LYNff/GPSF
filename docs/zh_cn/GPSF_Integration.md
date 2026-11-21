# GPSF 已并入 MMRotate 源码：文件位置与配置方法

## 新增源码

- `mmrotate/models/necks/gpsf/`：GSSF、DAM、RASA/ARC 及 GPSF neck。
- `mmrotate/models/detectors/gssf_strip_rcnn.py`：继承 StripRCNN，训练时增加 GSSF 的几何辅助损失。

## 注册改动

- `mmrotate/models/necks/__init__.py` 导入并注册 `GSSFNeck`。
- `mmrotate/models/detectors/__init__.py` 导入并注册 `GSSFStripRCNN`。
- 原 `strip_rcnn.py`、RPN、RoI head 不需要改动；模型构建通过 MMRotate registry 找到新类。配置中不再需要 `custom_imports`。

## 配置

`configs/strip_rcnn/strip_rcnn_s_gpsf_template.py` 是模型覆盖模板。用于 MBDD/GLH 时，把 `_base_` 改成你实际的数据集配置文件，然后保留该配置中的 dataset/split/pipeline/schedule/optimizer 设置。示例里的 GSSF 节点为 `(1,2)`，DAM 也在 `(1,2)`；若最终实验 DAM 用于所有自顶向下连接，将 `fam_levels`、`da_fam_levels` 同时设为 `(0,1,2)`。

`use_arc=True` 启用最深层 ARC 适配分支（论文中记为 RASA）；`num_scan_directions=4` 保持四路扫描。此训练实现使用 OBB GT 构造辅助几何损失，目标类别索引由 `bridge_label` 指定；单类桥梁数据集取 0。

DAM 需要 MMCV 编译的 `mmcv.ops.modulated_deform_conv2d`；GSSF 训练需要与你的 PyTorch/CUDA 相匹配的 `mamba_ssm`。

## 启动

```bash
python tools/train.py configs/strip_rcnn/strip_rcnn_s_gpsf_template.py \
  --work-dir work_dirs/gpsf_mbdd --seed 42
```

先把模板的 `_base_` 指向你正在使用的 MBDD 或 GLH 配置，再运行此命令；否则它仍会继承 DOTA 示例的数据配置。
