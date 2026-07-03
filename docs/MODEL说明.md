# 模型说明

## 当前定位

当前 MVP 是 TCT-only 的 WSI 二分类框架，并在 Attention-MIL 后接入一个独立训练的 YOLO11 二分类异常细胞检测头。检测头输出不直接替代 MIL 分类，而是作为 ROI 级别的引导信号，让 Attention-MIL 的 patch attention 更关注异常细胞区域。

当前主线：

```text
TCT WSI
  -> 512 个均匀采样 MIL patch，每个 patch 是 WSI 上 1024 x 1024 区域
  -> resize 到 224 x 224 后送入 UniCAS
  -> UniCAS encoder 提取 patch feature
  -> Attention-MIL 输出 WSI normal/abnormal
  -> 导出 attention Top-K ROI
  -> YOLO11 检测 1024x1024 ROI 内异常细胞
  -> roi_scores.csv 作为后续 attention guidance 输入
```

## Attention-MIL 内部结构

输入：

```text
features: [B, N, 1024]
mask:     [B, N]
label:    [B]
```

结构：

```text
Patch feature 1024
  -> Linear(1024, 256)
  -> LayerNorm(256)
  -> GELU
  -> Dropout
  -> Attention MLP:
       Linear(256, 128)
       Tanh
       Linear(128, 1)
  -> masked softmax 得到每个 patch 的 attention
  -> WSI feature = sum(attention_i * patch_embedding_i)
  -> Linear(256, 2)
  -> normal / abnormal logits
```

损失：

```text
Weighted CrossEntropyLoss
```

类别权重从训练集标签分布自动计算，用来缓解正常/异常比例不均衡。

## YOLO11 二分类检测头

当前推荐检测头训练数据：

```text
data/datasets/detector/yolo_binary/cell_detector.yaml
```

数据来源是 `ComparisonDetectorDataset` 的 COCO 标注，已整理为 YOLO 二分类数据。以下 COCO 类别被合并成 `abnormal_cell`：

```text
ascus, asch, lsil, hsil, scc, agc
```

以下类别被忽略，不作为异常细胞正例：

```text
trichomonas, candida, flora, herps, actinomyces
```

检测类别：

```text
0 = abnormal_cell
```

整理命令：

```powershell
python tools/prepare_detector_coco_yolo.py
```

训练命令：

```powershell
python train_yolo11_detector.py
```

## 检测头如何引导 Attention-MIL

1. Attention-MIL 先给每张 WSI 的 512 个 patch 打分，导出 Top-K ROI。
2. YOLO11 在每个 `1024 x 1024` ROI 上检测异常细胞。
3. 对每个 ROI 汇总检测信号：

```text
detector_score_max = ROI 内最高检测置信度
detector_score_sum = ROI 内所有检测置信度之和
detector_count     = ROI 内检测框数量
```

4. 将 ROI 检测分数映射回对应 patch，形成 guidance target。
5. 后续可在 MIL loss 上加入额外约束：

```text
loss = classification_loss + lambda * attention_guidance_loss
```

可选的 guidance loss：

```text
MSE(attention, normalized_detector_score)
KL(attention_distribution, detector_score_distribution)
Rank loss: 检测分高的 ROI attention 应高于检测分低的 ROI
```

生成新的 `roi_scores.csv` 后，下一步是在 `train_tct_mil.py` 中实现 guidance loss，把检测头 guidance 接回 Attention-MIL 训练。
