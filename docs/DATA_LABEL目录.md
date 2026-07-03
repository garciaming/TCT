# MVP 数据文件与标签文件目录

这个文件由 `tools/build_data_label_catalog.py` 生成。当前清理后推荐目录如下：

```text
data/manifests/
  wsi_manifest.csv
  feature_manifest_512.csv
  split_manifest.csv
  DATA_LABEL_CATALOG.csv
  DATA_LABEL_CATALOG.json

data/labels/files/
  tct_binary_labels.csv
  tct_multiclass_labels.csv
  unicas_slide_level_labels.csv
  diagnosis_text_labels.csv
```

重新生成：

```powershell
python tools/build_data_label_catalog.py
```

标签定义：

```text
0 = normal = NILM/Benign
1 = abnormal = ASC-US / LSIL / ASC-H / HSIL
```
