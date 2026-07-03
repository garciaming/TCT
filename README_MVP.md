# TCT MVP Pipeline

当前项目主线：

```text
TCT WSI -> 512 patch UniCAS feature -> Attention-MIL -> Top-K ROI -> YOLO11 二分类异常细胞检测头 -> ROI guidance score
```

根目录只保留训练入口；模型结构放在 `model/`，数据处理、ROI 导出、评估和可视化脚本放在 `tools/`。

## 项目结构

```text
mvp/
  train_tct_mil.py              # Attention-MIL 训练入口
  train_yolo11_detector.py      # YOLO11 二分类检测头训练入口
  model/
    attention_mil.py            # Attention-MIL 模型与 bag dataset
  tools/
    prepare_detector_coco_yolo.py
    prepare_tct_mvp.py
    extract_tct_features.py
    export_topk_rois.py
    prepare_cell_detection_data.py
    prepare_yolo_detection_data.py
    run_yolo11_roi_detector.py
    qc_patch_rois.py
    evaluate_roi_hits.py
    visualize_attention_heatmap.py
    parse_cell_annotations.py
    build_data_label_catalog.py
  data/
    datasets/                   # 原始数据和生成数据集
    labels/                     # 标签与检测框标注
    manifests/                  # WSI、特征、划分目录 CSV
  features/Pathology_TCT_MVP_patch_512/
  runs/tct_mvp_512/
  weights/pretrained/
  docs/
```

核心文档：

- [模型说明](docs/MODEL说明.md)
- [当前模型图](docs/MODEL图.md)
- [数据与标签目录](docs/DATA_LABEL目录.md)

## 环境

```powershell
conda activate prompt
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

## 1. 准备 TCT WSI 元数据

```powershell
python tools/prepare_tct_mvp.py
```

默认输出：

```text
data/datasets/tct_mvp/tct_mvp_all.csv
data/datasets/tct_mvp/tct_mvp_train.csv
data/datasets/tct_mvp/tct_mvp_val.csv
data/datasets/tct_mvp/tct_mvp_test.csv
data/datasets/tct_mvp/tct_mvp_summary.json
```

## 2. 提取 512 Patch UniCAS 特征

```powershell
python tools/extract_tct_features.py --weights weights/pretrained/UniCAS.pth --patch-size 1024 --encoder-input-size 224 --max-patches 512 --stride 1024 --batch-size 8 --feature-root features/Pathology_TCT_MVP_patch_512 --skip-existing
```

输出：

```text
features/Pathology_TCT_MVP_patch_512/<slide_stem>/torch/images.pt
features/Pathology_TCT_MVP_patch_512/<slide_stem>/torch/coords.csv
```

每个 MIL patch 对应 WSI 上 `1024 x 1024` 的区域；送入 UniCAS 前 resize 到 `224 x 224`。
UniCAS ViT 内部的 `patch_size=16` 是 resized 224 输入上的 token patch，不是 WSI 上的 MIL patch 尺寸。

## 3. 训练 Attention-MIL

```powershell
python train_tct_mil.py
```

输出：

```text
runs/tct_mvp_512/best.pt
runs/tct_mvp_512/last.pt
runs/tct_mvp_512/metrics.json
runs/tct_mvp_512/val_predictions.csv
```

## 4. 导出 Top-K ROI

```powershell
python tools/export_topk_rois.py
```

输出：

```text
runs/tct_mvp_512/topk_rois_all/topk_rois.csv
runs/tct_mvp_512/topk_rois_all/thumbnails/<slide_stem>/*.png
```

## 5. 构建 YOLO11 二分类检测数据

原始检测数据放到：

```text
data/datasets/detector/raw/ComparisonDetectorDataset
```

整理 ComparisonDetectorDataset，并把病变相关类别合并成二分类 YOLO 数据：

```powershell
python tools/prepare_detector_coco_yolo.py
```

如果后续还要使用旧 VOC 来源数据，可继续使用：

```powershell
python tools/prepare_cell_detection_data.py
python tools/prepare_yolo_detection_data.py --overwrite
```

输出：

```text
data/datasets/detector/yolo_binary/cell_detector.yaml
```

检测类别：

```text
0 = abnormal_cell
```

## 6. 训练 YOLO11 检测头

8GB 显存默认配置：

```powershell
python train_yolo11_detector.py
```

等价关键参数：

```powershell
python train_yolo11_detector.py --data data/datasets/detector/yolo_binary/cell_detector.yaml --model weights/pretrained/yolo11m.pt --name yolo11m_binary_1024_guidance --epochs 60 --imgsz 1024 --batch 4 --device 0 --workers 0 --patience 18
```

显存不够时：

```powershell
python train_yolo11_detector.py --model weights/pretrained/yolo11n.pt --name yolo11n_binary_1024_guidance --batch 4
```

## 7. 生成 ROI Guidance

```powershell
python tools/run_yolo11_roi_detector.py --save-annotated
```

输出：

```text
runs/tct_mvp_512/yolo11_binary_roi_guidance/roi_detections.csv
runs/tct_mvp_512/yolo11_binary_roi_guidance/roi_scores.csv
runs/tct_mvp_512/yolo11_binary_roi_guidance/roi_detections_meta.json
```

`roi_scores.csv` 中的 `detector_score_max`、`detector_score_sum`、`detector_count` 是后续引导 Attention-MIL 注意力分数的检测头信号。
