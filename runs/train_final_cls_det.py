from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import AttentionMIL, FinalDataPatchBagDataset, collate_patch_bags


# =========================================================
# 训练配置区
# =========================================================
# 服务器正式训练时只改这里，然后直接运行：
#
#   python runs/train_final_cls_det.py
#
# 不需要在命令行额外输入参数。

RUN_CONFIG = {
    # -------------------------
    # 数据路径
    # -------------------------
    # 1024 patch 根目录，目录结构应为：
    #   positive/<slide_name>/*.jpg
    #   negative/<slide_name>/*.jpg
    #   patch_1024_metadata.csv
    "patch_root": "database/datasets/final_data/1024_patch",

    # patch metadata。留空时默认使用 patch_root/patch_1024_metadata.csv。
    "metadata_csv": "",

    # json_global_to_1024_yolo.py 生成的 YOLO txt 目录。
    # 训练阶段的检测监督来自这里的框标注：patch_has_cell -> attention_guidance_loss。
    # 如果暂时不想用细胞框 attention guidance，可以保留该路径但把 box_guidance_lambda 设为 0。
    "yolo_label_dir": "database/datasets/final_data/json_to_1024_output/yolo_labels",

    # 可选：手工划分 train/val/test 的 CSV。
    # 留空时脚本按 slide 自动分层随机划分。
    # CSV 至少包含 split + slide_name，或 split + bag_key。
    "split_csv": "",

    # 输出目录。stage2 启用 detection/box attention guidance，避免覆盖 stage1 baseline。
    "out_dir": "runs/final_cls_det_stage2_m64_guide",

    # -------------------------
    # 分类模型结构
    # -------------------------
    # 二分类：negative=0, positive=1。
    "num_classes": 2,
    "embed_dim": 1024,
    "hidden_dim": 256,
    "dropout": 0.1,

    # UniCAS/encoder 权重。
    "encoder_weights": "weights/pretrained/UniCAS.pth",

    # 1024 patch 输入 encoder 前 resize 到 224。
    "patch_input_mode": "subpatch_4x4",
    "encoder_input_size": 224,
    "subpatch_grid_size": 4,
    "subpatch_tile_size": 256,
    "center_crop_size": 224,
    "expected_patch_size": 1024,

    # encoder 内部小批量大小。显存不够就调小，比如 4 或 8。
    "encoder_batch_size": 256,

    # False：冻结 encoder，只训练 MIL 分类头，推荐先这样跑通。
    # True：端到端微调 encoder + MIL，显存和时间开销更大。
    "train_encoder": False,

    # -------------------------
    # MIL bag 设置
    # -------------------------
    # 每张 slide 最多采样多少个 1024 patch。
    "max_patches": 64,

    # train 阶段 random；val/test 阶段会自动用 uniform，保证评估稳定。
    "sample_mode": "random",

    # MIL 通常 batch_size=1 起步；显存足够再调大。
    "batch_size": 1,
    "num_workers": 8,

    # 没有 split_csv 时使用这两个比例自动划分。
    "val_fraction": 0.2,
    "test_fraction": 0.0,

    # -------------------------
    # 训练超参数
    # -------------------------
    # False：跳过分类训练，直接加载 classification_checkpoint 或 out_dir/best.pt，
    # 然后完成 Top-K ROI + detector 导出。
    "run_training": False,

    # run_training=True 时才会训练这么多轮。
    "epochs": 15,

    # 留空时默认加载 out_dir/best.pt。
    "classification_checkpoint": "",

    "lr": 1e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 1e-4,

    # False：根据训练集类别分布自动加 class weight。
    "no_class_weights": False,

    # 细胞框对 attention 的辅助约束强度。
    # 0.0 表示只做 slide 分类，不使用检测框监督。
    # 0.05 是较稳的起步值；如果 attention 仍偏离细胞框，可再尝试 0.1。
    "box_guidance_lambda": 0.1,

    # CUDA 上建议 True。
    "amp": False,

    # auto/cpu/cuda/cuda:0 等。
    "device": "0",
    "seed": 2026,

    # -------------------------
    # 检测器与 Top-K ROI
    # -------------------------
    # 分类模型训练完后，取 attention 最高的 topk 个 patch 跑 detector。
    "topk": 128,

    # 你的预训练检测器权重。可以写具体 .pt 文件，也可以写目录。
    # 注意：这个权重只用于训练结束后的 Top-K ROI detector 推理导出；
    # 当前训练 loss 中的检测监督来自 yolo_label_dir 中的 txt 标注，而不是实时跑 detector。
    # 这里兼容用户目录 /weights/pretrianed/best.pt，也会自动 fallback 到项目内 weights/pretrianed/best.pt
    # 或常见拼写 weights/pretrained/best.pt。
    # 如果暂时只想看分类效果，把这里改成空字符串 ""。
    "detector_weights": "/weights/pretrained/best.pt",

    # auto：优先按 ultralytics YOLO 加载，失败后尝试 torch module。
    # 可选：auto / ultralytics / torch_module / torchscript。
    "detector_backend": "ultralytics",

    "detector_imgsz": 1024,
    "detector_conf": 0.20,
    "detector_iou": 0.7,
    "detector_batch_size": 4,
    "detector_device": "0",

    # 0：只在训练结束后用 best checkpoint 跑一次 detector。
    # 1：每个 epoch 都额外跑一次 detector 导出 Top-K 结果，会明显变慢。
    # 这个参数不控制训练 loss；训练中的 guidance 由 box_guidance_lambda + yolo_label_dir 控制。
    "detector_every": 0,
}


try:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
except Exception:
    accuracy_score = None
    balanced_accuracy_score = None
    f1_score = None
    roc_auc_score = None


def project_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def resolve_detector_weights(path: str | Path | None) -> Path | None:
    weights = project_path(path)
    if weights is None:
        return None
    if not weights.exists() and str(path):
        raw_path = Path(str(path))
        alternatives: list[Path] = []
        raw_text = str(path)
        if "pretrianed" in raw_text:
            alternatives.append(project_path(raw_text.replace("pretrianed", "pretrained")))
        if "pretrained" in raw_text:
            alternatives.append(project_path(raw_text.replace("pretrained", "pretrianed")))
        if raw_path.is_absolute():
            alternatives.append(PROJECT_ROOT / str(raw_path).lstrip("/\\"))
        for alternative in alternatives:
            if alternative is not None and alternative.exists():
                weights = alternative
                break

    if weights.is_file():
        return weights
    if not weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {weights}")
    if not weights.is_dir():
        raise FileNotFoundError(f"Detector weights path is not a file or directory: {weights}")

    patterns = ["*detector*.pt", "*yolo*.pt", "*best*.pt", "*.pt"]
    candidates: list[Path] = []
    for pattern in patterns:
        for candidate in weights.glob(pattern):
            name = candidate.name.lower()
            if candidate.is_file() and "unicas" not in name:
                candidates.append(candidate)
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(
            f"No detector .pt weights found under {weights}. "
            "Put best.pt/yolo*.pt/detector*.pt there, or set detector_weights to the exact file."
        )

    candidates = sorted(set(candidates), key=lambda item: item.stat().st_mtime, reverse=True)
    selected = candidates[0]
    print(f"Resolved detector weights: {normalize_path(selected)}")
    return selected


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device_arg} was requested, but torch.cuda.is_available() is False."
            )
        device_index = int(device_arg)
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device={device_arg} was requested, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        return torch.device(f"cuda:{device_index}")
    return torch.device(device_arg)


def load_torch(path: Path, map_location: str | torch.device = "cpu", mmap: bool = False) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False, mmap=mmap)
    except TypeError:
        try:
            return torch.load(path, map_location=map_location, mmap=mmap)
        except TypeError:
            return torch.load(path, map_location=map_location)


def build_unicas_encoder(weights_path: Path | None, device: torch.device) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("UniCAS encoder requires timm. Install timm on the server.") from exc

    params = {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "init_values": 1e-5,
        "mlp_ratio": 2.671875 * 2,
        "mlp_layer": functools.partial(timm.layers.mlp.GluMlp, gate_last=False),
        "act_layer": torch.nn.SiLU,
        "no_embed_class": False,
        "img_size": 224,
        "num_classes": 0,
        "in_chans": 3,
    }
    model = timm.models.VisionTransformer(**params)

    if weights_path is not None:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Encoder weights not found: {weights_path}")
        state = load_torch(weights_path, map_location="cpu", mmap=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        load_msg = model.load_state_dict(state, strict=False)
        del state
        gc.collect()
        print(f"Loaded encoder weights: {normalize_path(weights_path)}")
        print(load_msg)

    model.to(device)
    return model


class EndToEndAttentionMIL(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        mil_head: AttentionMIL,
        encoder_batch_size: int,
        freeze_encoder: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mil_head = mil_head
        self.encoder_batch_size = encoder_batch_size
        self.freeze_encoder = freeze_encoder

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def train(self, mode: bool = True) -> "EndToEndAttentionMIL":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode_flat(self, flat_images: torch.Tensor) -> torch.Tensor:
        chunks = []
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            for start in range(0, flat_images.shape[0], self.encoder_batch_size):
                chunk = flat_images[start : start + self.encoder_batch_size]
                pred = self.encoder(chunk)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                if pred.ndim == 3:
                    pred = pred[:, 0]
                chunks.append(pred.float())
        return torch.cat(chunks, dim=0)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_patches, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * num_patches, channels, height, width)
        features = self._encode_flat(flat_images).reshape(batch_size, num_patches, -1)
        logits, attention = self.mil_head(features, mask)
        return logits, attention, features


def compute_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def bag_labels(dataset: FinalDataPatchBagDataset) -> list[int]:
    return [int(bag["rows"].iloc[0]["label"]) for bag in dataset.bags]


def safe_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, num_classes: int) -> dict[str, Any]:
    if labels.size == 0:
        return {"loss": None, "accuracy": None, "balanced_accuracy": None, "f1_macro": None, "auc": None}

    if accuracy_score is None:
        accuracy = float((labels == preds).mean())
        return {
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "f1_macro": None,
            "auc": None,
        }

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": None,
    }
    try:
        if roc_auc_score is not None:
            if num_classes == 2 and len(np.unique(labels)) == 2:
                metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
            elif num_classes > 2 and len(np.unique(labels)) > 1:
                metrics["auc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auc"] = None
    return metrics


def attention_guidance_loss(attention: torch.Tensor, patch_has_cell: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_targets = patch_has_cell.float() * mask.float()
    keep = valid_targets.sum(dim=1) > 0
    if not keep.any():
        return attention.new_tensor(0.0)

    target = valid_targets[keep]
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    pred = attention[keep].clamp_min(1e-8)
    return -(target * pred.log()).sum(dim=1).mean()


def run_epoch(
    model: EndToEndAttentionMIL,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    box_guidance_lambda: float = 0.0,
    progress_desc: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train = optimizer is not None
    model.train(train)

    losses = []
    cls_losses = []
    guide_losses = []
    labels_all = []
    probs_all = []
    preds_all = []
    pred_rows = []

    if tqdm is not None and progress_desc:
        iterator = tqdm(loader, desc=progress_desc, total=len(loader), dynamic_ncols=True, leave=True)
    else:
        iterator = loader

    for batch in iterator:
        images = batch["images"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        patch_has_cell = batch["patch_has_cell"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, attention, _ = model(images, mask)
                cls_loss = criterion(logits, labels)
                guide_loss = attention_guidance_loss(attention, patch_has_cell, mask)
                loss = cls_loss + float(box_guidance_lambda) * guide_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        labels_np = labels.detach().cpu().numpy()

        losses.append(float(loss.detach().cpu()))
        cls_losses.append(float(cls_loss.detach().cpu()))
        guide_losses.append(float(guide_loss.detach().cpu()))

        if tqdm is not None and progress_desc:
            iterator.set_postfix(
                loss=f"{losses[-1]:.4f}",
                cls=f"{cls_losses[-1]:.4f}",
                guide=f"{guide_losses[-1]:.4f}",
            )

        labels_all.extend(labels_np.tolist())
        probs_all.append(probs)
        preds_all.extend(preds.tolist())

        for i, slide_name in enumerate(batch["slide_names"]):
            row = {
                "slide_name": slide_name,
                "bag_key": batch["bag_keys"][i],
                "label": int(labels_np[i]),
                "pred": int(preds[i]),
            }
            for class_idx in range(num_classes):
                row[f"prob_{class_idx}"] = float(probs[i, class_idx])
            pred_rows.append(row)

    labels_arr = np.array(labels_all)
    probs_arr = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, num_classes))
    preds_arr = np.array(preds_all)
    metrics = safe_metrics(labels_arr, probs_arr, preds_arr, num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    metrics["classification_loss"] = float(np.mean(cls_losses)) if cls_losses else None
    metrics["attention_guidance_loss"] = float(np.mean(guide_losses)) if guide_losses else None
    return metrics, pd.DataFrame(pred_rows)


class UltralyticsDetector:
    def __init__(
        self,
        weights: Path,
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        batch_size: int,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.batch_size = max(1, int(batch_size))

    def predict_paths(self, paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scores = []
        detections = []
        if not paths:
            return scores, detections

        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            results = self.model.predict(
                source=batch_paths,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                batch=self.batch_size,
                stream=True,
                save=False,
                verbose=False,
            )

            for path, result in zip(batch_paths, results):
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    scores.append({"detector_score_max": 0.0, "detector_score_sum": 0.0, "detector_count": 0})
                    continue

                confs = boxes.conf.detach().cpu().numpy().astype(float)
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
                scores.append(
                    {
                        "detector_score_max": float(confs.max()) if confs.size else 0.0,
                        "detector_score_sum": float(confs.sum()) if confs.size else 0.0,
                        "detector_count": int(confs.size),
                    }
                )
                for det_idx, (box, score, class_id) in enumerate(zip(xyxy, confs, classes)):
                    detections.append(
                        {
                            "patch_path": path,
                            "det_index": det_idx,
                            "class_id": int(class_id),
                            "confidence": float(score),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }
                    )

            del results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return scores, detections


class TorchModuleDetector:
    def __init__(
        self,
        weights: Path,
        device: torch.device,
        imgsz: int,
        conf: float,
        batch_size: int,
        torchscript: bool,
    ) -> None:
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
            ]
        )

        if torchscript:
            self.model = torch.jit.load(str(weights), map_location=device)
        else:
            checkpoint = load_torch(weights, map_location=device)
            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint
            elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
                self.model = checkpoint["model"]
            else:
                raise RuntimeError(
                    "Torch detector checkpoint must save a full nn.Module or a dict with key 'model'. "
                    "If detector.pth is only a state_dict, define its architecture first or use "
                    "--detector-backend ultralytics for YOLO weights."
                )

        self.model.to(device)
        self.model.eval()

    def _read_batch(self, paths: list[str]) -> torch.Tensor:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images).to(self.device)

    def _parse_one_output(self, path: str, output: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rows = []
        if isinstance(output, dict):
            boxes = output.get("boxes")
            scores = output.get("scores")
            labels = output.get("labels")
            if boxes is None or scores is None:
                return {"detector_score_max": 0.0, "detector_score_sum": 0.0, "detector_count": 0}, rows
            boxes = boxes.detach().cpu().numpy().astype(float)
            scores_np = scores.detach().cpu().numpy().astype(float)
            if labels is None:
                labels_np = np.zeros(len(scores_np), dtype=int)
            else:
                labels_np = labels.detach().cpu().numpy().astype(int)
        elif torch.is_tensor(output):
            arr = output.detach().cpu()
            if arr.ndim != 2 or arr.shape[1] < 5:
                return {"detector_score_max": 0.0, "detector_score_sum": 0.0, "detector_count": 0}, rows
            arr = arr.numpy().astype(float)
            boxes = arr[:, :4]
            scores_np = arr[:, 4]
            labels_np = arr[:, 5].astype(int) if arr.shape[1] >= 6 else np.zeros(len(scores_np), dtype=int)
        else:
            return {"detector_score_max": 0.0, "detector_score_sum": 0.0, "detector_count": 0}, rows

        keep = scores_np >= self.conf
        boxes = boxes[keep]
        scores_np = scores_np[keep]
        labels_np = labels_np[keep]

        score_row = {
            "detector_score_max": float(scores_np.max()) if scores_np.size else 0.0,
            "detector_score_sum": float(scores_np.sum()) if scores_np.size else 0.0,
            "detector_count": int(scores_np.size),
        }
        for det_idx, (box, score, class_id) in enumerate(zip(boxes, scores_np, labels_np)):
            rows.append(
                {
                    "patch_path": path,
                    "det_index": det_idx,
                    "class_id": int(class_id),
                    "confidence": float(score),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )
        return score_row, rows

    @torch.inference_mode()
    def predict_paths(self, paths: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scores = []
        detections = []
        for start in range(0, len(paths), self.batch_size):
            chunk_paths = paths[start : start + self.batch_size]
            batch = self._read_batch(chunk_paths)
            outputs = self.model(batch)

            if isinstance(outputs, dict):
                outputs = [outputs]
            elif torch.is_tensor(outputs):
                if outputs.ndim == 3:
                    outputs = [outputs[i] for i in range(outputs.shape[0])]
                elif outputs.ndim == 2 and len(chunk_paths) == 1:
                    outputs = [outputs]
                else:
                    outputs = [outputs[i] for i in range(outputs.shape[0])]

            for path, output in zip(chunk_paths, outputs):
                score_row, det_rows = self._parse_one_output(path, output)
                scores.append(score_row)
                detections.extend(det_rows)
        return scores, detections


def build_detector(args: argparse.Namespace, device: torch.device):
    weights = resolve_detector_weights(args.detector_weights)
    if weights is None:
        return None

    detector_device = args.detector_device
    if detector_device == "auto":
        detector_device = "0" if torch.cuda.is_available() else "cpu"

    if args.detector_backend in {"auto", "ultralytics"}:
        try:
            detector = UltralyticsDetector(
                weights=weights,
                device=detector_device,
                imgsz=args.detector_imgsz,
                conf=args.detector_conf,
                iou=args.detector_iou,
                batch_size=args.detector_batch_size,
            )
            print(f"Loaded detector with ultralytics: {normalize_path(weights)}")
            return detector
        except Exception as exc:
            if args.detector_backend == "ultralytics":
                raise
            print(f"Ultralytics detector load failed, falling back to torch module: {exc}")

    torch_device = choose_device(detector_device if detector_device != "0" else "cuda:0")
    detector = TorchModuleDetector(
        weights=weights,
        device=torch_device,
        imgsz=args.detector_imgsz,
        conf=args.detector_conf,
        batch_size=args.detector_batch_size,
        torchscript=args.detector_backend == "torchscript",
    )
    print(f"Loaded detector with torch module: {normalize_path(weights)}")
    return detector


def export_topk_and_detector(
    model: EndToEndAttentionMIL,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    prefix: str,
    num_classes: int,
    topk: int,
    detector: Any | None,
    release_model_before_detector: bool = False,
) -> dict[str, str]:
    model.eval()
    roi_rows = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits, attention, _ = model(images, mask)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels_np = labels.detach().cpu().numpy()
            attention_np = attention.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy()

            for i, slide_name in enumerate(batch["slide_names"]):
                valid_indices = np.where(mask_np[i])[0]
                if valid_indices.size == 0:
                    continue
                scores = attention_np[i, valid_indices]
                k = min(topk, valid_indices.size)
                top_order = np.argsort(scores)[::-1][:k]
                top_indices = valid_indices[top_order]

                for rank, patch_index in enumerate(top_indices, start=1):
                    coord = batch["coords"][i][int(patch_index)]
                    row = {
                        "slide_name": slide_name,
                        "bag_key": batch["bag_keys"][i],
                        "label": int(labels_np[i]),
                        "pred": int(preds[i]),
                        "rank": int(rank),
                        "patch_index": int(patch_index),
                        "patch_name": batch["patch_names"][i][int(patch_index)],
                        "patch_path": batch["patch_paths"][i][int(patch_index)],
                        "attention": float(attention_np[i, int(patch_index)]),
                    }
                    for class_idx in range(num_classes):
                        row[f"prob_{class_idx}"] = float(probs[i, class_idx])
                    row.update(coord)
                    roi_rows.append(row)

    topk_df = pd.DataFrame(roi_rows)
    topk_path = out_dir / f"{prefix}_topk_rois.csv"
    topk_df.to_csv(topk_path, index=False, encoding="utf-8-sig")
    outputs = {"topk_rois": normalize_path(topk_path)}

    if detector is None or topk_df.empty:
        return outputs

    if release_model_before_detector:
        try:
            del images, mask, labels, logits, attention
        except NameError:
            pass
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    paths = topk_df["patch_path"].astype(str).tolist()
    detector_scores, detections = detector.predict_paths(paths)

    score_df = pd.concat(
        [topk_df.reset_index(drop=True), pd.DataFrame(detector_scores).reset_index(drop=True)],
        axis=1,
    )
    score_path = out_dir / f"{prefix}_detector_roi_scores.csv"
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")
    outputs["detector_roi_scores"] = normalize_path(score_path)

    if detections:
        det_df = pd.DataFrame(detections)
        meta_cols = [
            "slide_name",
            "bag_key",
            "label",
            "pred",
            "rank",
            "patch_index",
            "patch_name",
            "patch_path",
            "attention",
        ]
        det_df = det_df.merge(topk_df[meta_cols], on="patch_path", how="left")
    else:
        det_df = pd.DataFrame(
            columns=[
                "slide_name",
                "bag_key",
                "label",
                "pred",
                "rank",
                "patch_index",
                "patch_name",
                "patch_path",
                "attention",
                "det_index",
                "class_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
    det_path = out_dir / f"{prefix}_detector_detections.csv"
    det_df.to_csv(det_path, index=False, encoding="utf-8-sig")
    outputs["detector_detections"] = normalize_path(det_path)
    return outputs


def make_loader(
    dataset: FinalDataPatchBagDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_patch_bags,
    )


def build_args_from_config() -> argparse.Namespace:
    return argparse.Namespace(**RUN_CONFIG)


def main() -> None:
    args = build_args_from_config()
    set_seed(args.seed)

    device = choose_device(args.device)
    out_dir = project_path(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Project root: {normalize_path(PROJECT_ROOT)}")
    print(f"Output dir: {normalize_path(out_dir)}")
    print(f"Device: {device}")

    encoder_weights = project_path(args.encoder_weights)
    print("Loading UniCAS encoder...", flush=True)
    encoder = build_unicas_encoder(encoder_weights, device=device)
    mil_head = AttentionMIL(args.embed_dim, args.hidden_dim, args.num_classes, args.dropout)
    model = EndToEndAttentionMIL(
        encoder=encoder,
        mil_head=mil_head,
        encoder_batch_size=args.encoder_batch_size,
        freeze_encoder=not args.train_encoder,
    ).to(device)

    metadata_csv = project_path(args.metadata_csv)
    yolo_label_dir = project_path(args.yolo_label_dir)
    split_csv = project_path(args.split_csv)

    dataset_kwargs = {
        "patch_root": project_path(args.patch_root),
        "metadata_csv": metadata_csv,
        "yolo_label_dir": yolo_label_dir,
        "split_csv": split_csv,
        "max_patches": args.max_patches,
        "encoder_input_size": args.encoder_input_size,
        "patch_input_mode": args.patch_input_mode,
        "subpatch_grid_size": args.subpatch_grid_size,
        "subpatch_tile_size": args.subpatch_tile_size,
        "center_crop_size": args.center_crop_size,
        "expected_patch_size": args.expected_patch_size,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "sample_mode": args.sample_mode,
    }
    train_ds = FinalDataPatchBagDataset(split="train", **dataset_kwargs)
    val_ds = FinalDataPatchBagDataset(split="val", **dataset_kwargs)
    train_loader = make_loader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train bags: {len(train_ds)}")
    print(f"Val bags: {len(val_ds)}")
    print(f"Train patch rows: {train_ds.num_patch_rows}")
    print(f"Val patch rows: {val_ds.num_patch_rows}")

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(bag_labels(train_ds), args.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    param_groups = [{"params": [p for p in model.mil_head.parameters() if p.requires_grad], "lr": args.lr}]
    if args.train_encoder:
        param_groups.append({"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    should_train = bool(args.run_training and args.epochs > 0)
    detector = build_detector(args, device) if should_train and args.detector_weights and args.detector_every > 0 else None

    best_score = -float("inf")
    best_preds = None
    history = []
    start_time = time.time()

    if should_train:
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}: training...", flush=True)
            train_metrics, _ = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                box_guidance_lambda=args.box_guidance_lambda,
                progress_desc=f"epoch {epoch}/{args.epochs} train",
            )
            print(f"Epoch {epoch}/{args.epochs}: validating...", flush=True)
            val_metrics, val_preds = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                box_guidance_lambda=args.box_guidance_lambda,
                progress_desc=f"epoch {epoch}/{args.epochs} val",
            )

            score = val_metrics.get("auc")
            if score is None:
                score = val_metrics.get("balanced_accuracy")
            if score is None:
                score = -float("inf")

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "score": float(score),
                "elapsed_sec": round(time.time() - start_time, 2),
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False))
            (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            val_preds.to_csv(out_dir / "val_predictions_last.csv", index=False, encoding="utf-8-sig")

            if float(score) > best_score:
                best_score = float(score)
                best_preds = val_preds
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "score": best_score,
                        "class_weights": class_weights.detach().cpu() if class_weights is not None else None,
                    },
                    out_dir / "best.pt",
                )

            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "last.pt")

            if detector is not None and args.detector_every > 0 and epoch % args.detector_every == 0:
                outputs = export_topk_and_detector(
                    model=model,
                    loader=val_loader,
                    device=device,
                    out_dir=out_dir,
                    prefix=f"epoch_{epoch:03d}_val",
                    num_classes=args.num_classes,
                    topk=args.topk,
                    detector=detector,
                )
                print(json.dumps({"epoch": epoch, "detector_outputs": outputs}, ensure_ascii=False))
    else:
        print("Skipping classification training; running final Top-K ROI + detector export only.", flush=True)

    if best_preds is not None:
        best_preds.to_csv(out_dir / "val_predictions_best.csv", index=False, encoding="utf-8-sig")

    checkpoint_path = project_path(args.classification_checkpoint) if args.classification_checkpoint else out_dir / "best.pt"
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Classification checkpoint not found: {checkpoint_path}. "
            "Set classification_checkpoint or run training first to create out_dir/best.pt."
        )
    checkpoint = load_torch(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded best checkpoint from {normalize_path(checkpoint_path)} epoch={checkpoint.get('epoch')} score={checkpoint.get('score')}")

    final_detector = build_detector(args, device) if args.detector_weights else None
    final_outputs = export_topk_and_detector(
        model=model,
        loader=val_loader,
        device=device,
        out_dir=out_dir,
        prefix="best_val",
        num_classes=args.num_classes,
        topk=args.topk,
        detector=final_detector,
        release_model_before_detector=final_detector is not None,
    )
    (out_dir / "final_outputs.json").write_text(
        json.dumps(final_outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"final_outputs": final_outputs}, ensure_ascii=False))
    print(f"Done. Outputs written to {normalize_path(out_dir)}")


if __name__ == "__main__":
    main()
