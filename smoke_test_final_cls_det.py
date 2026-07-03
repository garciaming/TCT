from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import AttentionMIL, FinalDataPatchBagDataset, collate_patch_bags


def load_training_module() -> Any:
    module_path = PROJECT_ROOT / "runs" / "train_final_cls_det.py"
    spec = importlib.util.spec_from_file_location("train_final_cls_det_smoke_import", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING = load_training_module()


def project_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny end-to-end smoke test for UniCAS 4x4 subpatch extraction + Attention-MIL training."
    )
    parser.add_argument("--patch-root", default="database/datasets/final_data/1024_patch")
    parser.add_argument(
        "--metadata-csv",
        default="",
        help="Defaults to <patch-root>/patch_1024_metadata.csv.",
    )
    parser.add_argument("--encoder-weights", default="weights/pretrained/UniCAS.pth")
    parser.add_argument("--out-dir", default="runs/final_cls_det_smoke")
    parser.add_argument("--tmp-patch-root", default="", help="Defaults to <out-dir>/smoke_1024_patch.")
    parser.add_argument("--train-bags-per-class", type=int, default=1)
    parser.add_argument("--val-bags-per-class", type=int, default=1)
    parser.add_argument("--patches-per-bag", type=int, default=1)
    parser.add_argument("--max-patches", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-batch-size", type=int, default=4)
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--link-mode", choices=["auto", "symlink", "copy"], default="auto")
    parser.add_argument("--keep-tmp", action="store_true", help="Reuse the tiny temp dataset directory if it exists.")
    parser.add_argument("--rebuild-tmp", action="store_true", help="Deprecated; temp data is rebuilt by default.")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--skip-export-topk", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_metadata_rows(
    metadata_csv: Path,
    train_bags_per_class: int,
    val_bags_per_class: int,
    patches_per_bag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed_bags = train_bags_per_class + val_bags_per_class
    wanted_labels = ("negative", "positive")
    collected: dict[str, dict[str, list[dict[str, Any]]]] = {label: {} for label in wanted_labels}

    usecols = [
        "patch_name",
        "slide_name",
        "slide_file",
        "label",
        "row_1024",
        "col_1024",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "patch_size",
        "level",
        "cell_pixel_ratio",
        "save_path",
    ]
    for chunk in pd.read_csv(metadata_csv, usecols=lambda col: col in usecols, chunksize=100_000):
        for row in chunk.to_dict("records"):
            label = str(row.get("label", "")).strip().lower()
            if label not in collected:
                continue
            slide_name = str(row.get("slide_name", "")).strip()
            patch_name = str(row.get("patch_name", "")).strip()
            if not slide_name or not patch_name:
                continue
            label_bags = collected[label]
            if slide_name not in label_bags and len(label_bags) >= needed_bags:
                continue
            rows = label_bags.setdefault(slide_name, [])
            if len(rows) < patches_per_bag:
                rows.append(row)

        if all(len(collected[label]) >= needed_bags for label in wanted_labels):
            break

    split_rows: list[dict[str, str]] = []
    selected_rows: list[dict[str, Any]] = []
    for label in wanted_labels:
        slides = [slide for slide, rows in collected[label].items() if len(rows) >= patches_per_bag]
        if len(slides) < needed_bags:
            raise ValueError(
                f"Not enough {label} bags in metadata. Need {needed_bags}, found {len(slides)}."
            )
        train_slides = slides[:train_bags_per_class]
        val_slides = slides[train_bags_per_class:needed_bags]
        for split, split_slides in [("train", train_slides), ("val", val_slides)]:
            for slide_name in split_slides:
                bag_key = f"{label}/{slide_name}"
                split_rows.append({"split": split, "bag_key": bag_key, "slide_name": slide_name})
                selected_rows.extend(collected[label][slide_name][:patches_per_bag])

    return pd.DataFrame(selected_rows), pd.DataFrame(split_rows)


def resolve_source_path(row: dict[str, Any], patch_root: Path) -> Path:
    label = str(row.get("label", "")).strip().lower()
    slide_name = str(row.get("slide_name", "")).strip()
    patch_name = str(row.get("patch_name", "")).strip()
    candidates = [
        patch_root / label / slide_name / patch_name,
        patch_root / slide_name / patch_name,
        patch_root / patch_name,
    ]
    save_path = str(row.get("save_path", "")).strip()
    if save_path:
        candidates.extend([Path(save_path), patch_root / save_path, patch_root / Path(save_path).name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find image for {label}/{slide_name}/{patch_name}")


def materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode in {"auto", "symlink"}:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            if link_mode == "symlink":
                raise
    shutil.copy2(src, dst)


def build_tiny_patch_root(args: argparse.Namespace) -> tuple[Path, Path, Path, pd.DataFrame]:
    patch_root = project_path(args.patch_root)
    out_dir = project_path(args.out_dir)
    if patch_root is None or not patch_root.is_dir():
        raise FileNotFoundError(f"patch root not found: {patch_root}")
    if out_dir is None:
        raise ValueError("--out-dir is required")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = project_path(args.metadata_csv) if args.metadata_csv else patch_root / "patch_1024_metadata.csv"
    if metadata_csv is None or not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_csv}")

    tmp_patch_root = project_path(args.tmp_patch_root) if args.tmp_patch_root else out_dir / "smoke_1024_patch"
    if tmp_patch_root is None:
        raise ValueError("--tmp-patch-root resolved to None")
    if tmp_patch_root.exists() and (args.rebuild_tmp or not args.keep_tmp):
        shutil.rmtree(tmp_patch_root)
    tmp_patch_root.mkdir(parents=True, exist_ok=True)

    selected, split_table = read_metadata_rows(
        metadata_csv=metadata_csv,
        train_bags_per_class=args.train_bags_per_class,
        val_bags_per_class=args.val_bags_per_class,
        patches_per_bag=args.patches_per_bag,
    )

    copied_rows = []
    for row in selected.to_dict("records"):
        src = resolve_source_path(row, patch_root)
        label = str(row["label"]).strip().lower()
        slide_name = str(row["slide_name"]).strip()
        patch_name = str(row["patch_name"]).strip()
        dst = tmp_patch_root / label / slide_name / patch_name
        materialize_file(src, dst, args.link_mode)
        copied_rows.append(row)

    tiny_metadata = pd.DataFrame(copied_rows)
    tiny_metadata_path = tmp_patch_root / "patch_1024_metadata.csv"
    split_csv = out_dir / "smoke_split.csv"
    tiny_metadata.to_csv(tiny_metadata_path, index=False, encoding="utf-8-sig")
    split_table.to_csv(split_csv, index=False, encoding="utf-8-sig")
    return tmp_patch_root, tiny_metadata_path, split_csv, split_table


def make_loader(dataset: FinalDataPatchBagDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_patch_bags,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    start = time.time()

    out_dir = project_path(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Building tiny smoke-test patch dataset...", flush=True)
    tmp_patch_root, tiny_metadata_path, split_csv, split_table = build_tiny_patch_root(args)
    print(f"Tiny patch root: {normalize_path(tmp_patch_root)}", flush=True)
    print(split_table.to_string(index=False), flush=True)

    dataset_kwargs = {
        "patch_root": tmp_patch_root,
        "metadata_csv": tiny_metadata_path,
        "yolo_label_dir": None,
        "split_csv": split_csv,
        "max_patches": args.max_patches,
        "patch_input_mode": "subpatch_4x4",
        "encoder_input_size": 224,
        "subpatch_grid_size": 4,
        "subpatch_tile_size": 256,
        "center_crop_size": 224,
        "expected_patch_size": 1024,
        "seed": args.seed,
        "val_fraction": 0.0,
        "test_fraction": 0.0,
        "sample_mode": "uniform",
    }
    train_ds = FinalDataPatchBagDataset(split="train", **dataset_kwargs)
    val_ds = FinalDataPatchBagDataset(split="val", **dataset_kwargs)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    first_item = train_ds[0]
    print(f"First train item images shape: {tuple(first_item['images'].shape)}", flush=True)
    print(f"First train item first coord: {first_item['coords'][0]}", flush=True)

    device = TRAINING.choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device}", flush=True)
    print("Loading UniCAS encoder...", flush=True)
    encoder_weights = project_path(args.encoder_weights)
    encoder = TRAINING.build_unicas_encoder(encoder_weights, device=device)
    mil_head = AttentionMIL(1024, args.hidden_dim, args.num_classes, args.dropout)
    model = TRAINING.EndToEndAttentionMIL(
        encoder=encoder,
        mil_head=mil_head,
        encoder_batch_size=args.encoder_batch_size,
        freeze_encoder=not args.train_encoder,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = TRAINING.compute_class_weights(TRAINING.bag_labels(train_ds), args.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    param_groups = [{"params": [p for p in model.mil_head.parameters() if p.requires_grad], "lr": args.lr}]
    if args.train_encoder:
        param_groups.append(
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.encoder_lr}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}: training...", flush=True)
        train_metrics, _ = TRAINING.run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            box_guidance_lambda=0.0,
        )
        print(f"Epoch {epoch}/{args.epochs}: validating...", flush=True)
        val_metrics, val_preds = TRAINING.run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            box_guidance_lambda=0.0,
        )
        score = val_metrics.get("auc")
        if score is None:
            score = val_metrics.get("balanced_accuracy")
        if score is None:
            score = -float("inf")

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": float(score)}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        val_preds.to_csv(out_dir / "val_predictions_last.csv", index=False, encoding="utf-8-sig")

        if float(score) > best_score:
            best_score = float(score)
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

    (out_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    final_outputs = {}
    if not args.skip_export_topk:
        print("Exporting top-k ROIs without detector...", flush=True)
        final_outputs = TRAINING.export_topk_and_detector(
            model=model,
            loader=val_loader,
            device=device,
            out_dir=out_dir,
            prefix="smoke_val",
            num_classes=args.num_classes,
            topk=args.topk,
            detector=None,
        )
        (out_dir / "final_outputs.json").write_text(
            json.dumps(final_outputs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "ok": True,
        "elapsed_sec": round(time.time() - start, 2),
        "tmp_patch_root": normalize_path(tmp_patch_root),
        "train_bags": len(train_ds),
        "val_bags": len(val_ds),
        "first_train_images_shape": list(first_item["images"].shape),
        "best": normalize_path(out_dir / "best.pt"),
        "last": normalize_path(out_dir / "last.pt"),
        "metrics": normalize_path(out_dir / "metrics.json"),
        "final_outputs": final_outputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
