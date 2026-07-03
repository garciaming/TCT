# Project Structure

```text
mvp/
  README_MVP.md
  PROJECT_STRUCTURE.md
  train_tct_mil.py
  train_yolo11_detector.py
  model/
    __init__.py
    attention_mil.py
  tools/
    build_data_label_catalog.py
    evaluate_roi_hits.py
    export_topk_rois.py
    extract_tct_features.py
    parse_cell_annotations.py
    prepare_cell_detection_data.py
    prepare_detector_coco_yolo.py
    prepare_tct_mvp.py
    prepare_yolo_detection_data.py
    qc_patch_rois.py
    run_yolo11_roi_detector.py
    visualize_attention_heatmap.py
  data/
    datasets/
      detector/
        raw/ComparisonDetectorDataset/
        yolo_binary/
      TCT_Slides/
      tct_mvp/
    labels/
      files/
      annotations/
      annotations_yx/
    manifests/
  features/
    Pathology_TCT_MVP_patch_512/
  runs/
    tct_mvp_512/
    yolo11_cell_detector/
  weights/
    pretrained/
      UniCAS.pth
      yolo11m.pt
      yolo11n.pt
      yolo11s.pt
```

主目录下只保留训练入口：

```text
train_tct_mil.py
train_yolo11_detector.py
```

模型代码：

```text
model/attention_mil.py
```

工具脚本：

```text
tools/*.py
```
