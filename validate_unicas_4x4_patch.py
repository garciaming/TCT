from __future__ import annotations

import argparse
import functools
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PATH_COLUMNS = ("image_path", "patch_path", "path", "file_path", "filename", "save_path")
X_COLUMNS = ("x_min", "x", "global_x", "left", "patch_x")
Y_COLUMNS = ("y_min", "y", "global_y", "top", "patch_y")


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
        description=(
            "Validate the 1024 patch -> 4x4 256 tiles -> center-crop 224 -> UniCAS feature pipeline."
        )
    )
    parser.add_argument("--patch-root", default="database/datasets/final_data/1024_patch")
    parser.add_argument(
        "--metadata-csv",
        default="",
        help="CSV with patch paths/coordinates. Defaults to <patch-root>/patch_1024_metadata.csv if it exists.",
    )
    parser.add_argument("--weights", default="weights/pretrained/UniCAS.pth")
    parser.add_argument("--out-dir", default="runs/validate_unicas_4x4_patch")
    parser.add_argument("--max-patches", type=int, default=4)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save an annotated preview of the first selected 1024 patch.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device_arg} was requested, but torch.cuda.is_available() is False. "
                "Use --device cpu or fix the CUDA environment."
            )
        device_index = int(device_arg)
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {device_arg} was requested, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        return torch.device(f"cuda:{device_index}")
    return torch.device(device_arg)


def load_torch(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_unicas_encoder(weights_path: Path, device: torch.device) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("UniCAS encoder requires timm. Install it on the server: pip install timm") from exc

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

    if not weights_path.is_file():
        raise FileNotFoundError(f"UniCAS weights not found: {normalize_path(weights_path)}")
    state = load_torch(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    load_msg = model.load_state_dict(state, strict=False)
    print(f"Loaded UniCAS weights: {normalize_path(weights_path)}")
    print(load_msg)

    model.to(device)
    model.eval()
    return model


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def as_clean_str(value: object) -> str:
    return str(value).strip()


def is_missing(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def first_existing_column(row: pd.Series, candidates: tuple[str, ...]) -> object | None:
    for column in candidates:
        if column in row and not is_missing(row[column]):
            return row[column]
    return None


def infer_patch_xy(row: pd.Series, patch_size: int) -> tuple[int | None, int | None]:
    x_value = first_existing_column(row, X_COLUMNS)
    y_value = first_existing_column(row, Y_COLUMNS)
    if x_value is not None and y_value is not None:
        return int(float(x_value)), int(float(y_value))

    if "row_1024" in row and "col_1024" in row and not is_missing(row["row_1024"]) and not is_missing(row["col_1024"]):
        return int(float(row["col_1024"]) * patch_size), int(float(row["row_1024"]) * patch_size)

    return None, None


def scan_images(patch_root: Path, max_images: int = 0) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for image_path in patch_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rows.append(
            {
                "image_path": normalize_path(image_path),
                "patch_name": image_path.name,
                "slide_name": image_path.parent.name,
                "label_name": image_path.parent.parent.name if image_path.parent.parent != patch_root else "",
            }
        )
        if max_images > 0 and len(rows) >= max_images:
            break
    if not rows:
        raise ValueError(f"No patch images found under {normalize_path(patch_root)}")
    return pd.DataFrame(rows)


def resolve_image_from_metadata(row: pd.Series, patch_root: Path) -> Path | None:
    path_value = first_existing_column(row, PATH_COLUMNS)
    if path_value is not None:
        raw_path = str(path_value).strip()
        candidates = []
        direct = project_path(raw_path)
        if direct is not None:
            candidates.append(direct)
        candidates.append(patch_root / raw_path)
        candidates.append(patch_root / Path(raw_path).name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    patch_name = as_clean_str(row.get("patch_name", "")) if "patch_name" in row else ""
    slide_name = as_clean_str(row.get("slide_name", "")) if "slide_name" in row else ""
    label_name = as_clean_str(row.get("label", "")) if "label" in row else ""
    if not patch_name:
        return None

    candidates = []
    if label_name and slide_name:
        candidates.append(patch_root / label_name / slide_name / patch_name)
    if slide_name:
        candidates.append(patch_root / slide_name / patch_name)
    candidates.append(patch_root / patch_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_metadata(
    patch_root: Path,
    metadata_csv_arg: str,
    patch_size: int,
    max_patches: int,
    sample_mode: str,
    seed: int,
) -> pd.DataFrame:
    metadata_csv = project_path(metadata_csv_arg) if metadata_csv_arg else patch_root / "patch_1024_metadata.csv"
    if metadata_csv is None or not metadata_csv.is_file():
        scan_limit = max_patches if max_patches > 0 else 0
        images = scan_images(patch_root, max_images=scan_limit)
        images["patch_x"] = None
        images["patch_y"] = None
        return images

    metadata = pd.read_csv(metadata_csv)
    if metadata.empty:
        raise ValueError(f"Metadata CSV is empty: {normalize_path(metadata_csv)}")
    metadata = select_rows(metadata, max_patches, sample_mode, seed)

    resolved_rows: list[dict[str, object]] = []
    for row_index, row in metadata.iterrows():
        image_path = resolve_image_from_metadata(row, patch_root)
        patch_name = as_clean_str(row.get("patch_name", "")) if "patch_name" in row else ""
        slide_name = as_clean_str(row.get("slide_name", "")) if "slide_name" in row else ""

        if image_path is None:
            continue

        patch_x, patch_y = infer_patch_xy(row, patch_size)
        resolved_rows.append(
            {
                "image_path": normalize_path(image_path),
                "patch_name": patch_name or image_path.name,
                "slide_name": slide_name or image_path.parent.name,
                "patch_x": patch_x,
                "patch_y": patch_y,
                "metadata_row": int(row_index),
            }
        )

    if not resolved_rows:
        raise ValueError(
            "No metadata rows could be matched to image files. Check --patch-root and metadata path columns."
        )
    return pd.DataFrame(resolved_rows)


def select_rows(table: pd.DataFrame, max_patches: int, sample_mode: str, seed: int) -> pd.DataFrame:
    if max_patches <= 0 or max_patches >= len(table):
        return table.reset_index(drop=True)
    if sample_mode == "random":
        return table.sample(n=max_patches, random_state=seed).reset_index(drop=True)
    return table.head(max_patches).reset_index(drop=True)


def split_and_crop_patch(
    image: Image.Image,
    patch_row: pd.Series,
    patch_index: int,
    args: argparse.Namespace,
    transform: transforms.Compose,
) -> tuple[list[torch.Tensor], list[dict[str, object]]]:
    expected_size = (args.patch_size, args.patch_size)
    if image.size != expected_size:
        raise ValueError(
            f"Expected {expected_size[0]}x{expected_size[1]} patch, got {image.size} for {patch_row['image_path']}"
        )
    if args.tile_size * args.grid_size != args.patch_size:
        raise ValueError("--tile-size * --grid-size must equal --patch-size")
    if args.crop_size > args.tile_size:
        raise ValueError("--crop-size must be <= --tile-size")

    margin = (args.tile_size - args.crop_size) // 2
    patch_x = None if is_missing(patch_row.get("patch_x")) else int(patch_row["patch_x"])
    patch_y = None if is_missing(patch_row.get("patch_y")) else int(patch_row["patch_y"])

    tensors: list[torch.Tensor] = []
    coord_rows: list[dict[str, object]] = []
    rgb = image.convert("RGB")

    for tile_row in range(args.grid_size):
        for tile_col in range(args.grid_size):
            tile_x = tile_col * args.tile_size
            tile_y = tile_row * args.tile_size
            crop_x = tile_x + margin
            crop_y = tile_y + margin
            crop_box = (crop_x, crop_y, crop_x + args.crop_size, crop_y + args.crop_size)
            crop = rgb.crop(crop_box)
            tensors.append(transform(crop))

            global_crop_x = None if patch_x is None else patch_x + crop_x
            global_crop_y = None if patch_y is None else patch_y + crop_y
            coord_rows.append(
                {
                    "flat_index": patch_index * args.grid_size * args.grid_size + tile_row * args.grid_size + tile_col,
                    "patch_index": patch_index,
                    "patch_name": patch_row.get("patch_name", Path(str(patch_row["image_path"])).name),
                    "slide_name": patch_row.get("slide_name", ""),
                    "image_path": patch_row["image_path"],
                    "tile_row": tile_row,
                    "tile_col": tile_col,
                    "patch_x": patch_x,
                    "patch_y": patch_y,
                    "tile_x_in_patch": tile_x,
                    "tile_y_in_patch": tile_y,
                    "crop_x_in_patch": crop_x,
                    "crop_y_in_patch": crop_y,
                    "crop_size": args.crop_size,
                    "global_crop_x_min": global_crop_x,
                    "global_crop_y_min": global_crop_y,
                    "global_crop_x_max": None if global_crop_x is None else global_crop_x + args.crop_size,
                    "global_crop_y_max": None if global_crop_y is None else global_crop_y + args.crop_size,
                }
            )

    return tensors, coord_rows


def encode_tensors(model: nn.Module, tensors: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    features: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, tensors.shape[0], batch_size):
            batch = tensors[start : start + batch_size].to(device, non_blocking=True)
            pred = model(batch)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            if pred.ndim == 3:
                pred = pred[:, 0]
            features.append(pred.detach().cpu().float())
    return torch.cat(features, dim=0)


def save_preview(image_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    margin = (args.tile_size - args.crop_size) // 2
    for row in range(args.grid_size):
        for col in range(args.grid_size):
            tile_x = col * args.tile_size
            tile_y = row * args.tile_size
            draw.rectangle(
                (tile_x, tile_y, tile_x + args.tile_size - 1, tile_y + args.tile_size - 1),
                outline="red",
                width=3,
            )
            crop_x = tile_x + margin
            crop_y = tile_y + margin
            draw.rectangle(
                (crop_x, crop_y, crop_x + args.crop_size - 1, crop_y + args.crop_size - 1),
                outline="cyan",
                width=3,
            )
    image.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    patch_root = project_path(args.patch_root)
    weights_path = project_path(args.weights)
    out_dir = project_path(args.out_dir)
    if patch_root is None or not patch_root.is_dir():
        raise FileNotFoundError(f"patch root not found: {patch_root}")
    if weights_path is None:
        raise ValueError("--weights is required")
    if out_dir is None:
        raise ValueError("--out-dir is required")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading metadata and selecting sample patches...", flush=True)
    table = read_metadata(
        patch_root,
        args.metadata_csv,
        args.patch_size,
        args.max_patches,
        args.sample_mode,
        args.seed,
    )
    selected = select_rows(table, args.max_patches, args.sample_mode, args.seed)
    print(f"Selected {len(selected)} 1024 patches.", flush=True)

    transform = make_transform()
    all_tensors: list[torch.Tensor] = []
    coord_rows: list[dict[str, object]] = []

    print("Splitting 1024 patches into 4x4 center-cropped 224 inputs...", flush=True)
    for patch_index, row in selected.iterrows():
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            tensors, rows = split_and_crop_patch(image, row, patch_index, args, transform)
        all_tensors.extend(tensors)
        coord_rows.extend(rows)

    input_tensor = torch.stack(all_tensors, dim=0)
    device = choose_device(args.device)
    print(f"Loading UniCAS encoder on {device}...", flush=True)
    model = build_unicas_encoder(weights_path, device=device)
    print(f"Encoding {input_tensor.shape[0]} cropped 224 inputs...", flush=True)
    features_flat = encode_tensors(model, input_tensor, args.batch_size, device=device)

    subpatches_per_patch = args.grid_size * args.grid_size
    features_by_patch = features_flat.reshape(len(selected), subpatches_per_patch, -1)
    features_mean = features_by_patch.mean(dim=1)

    coord_path = out_dir / "subpatch_224_coords.csv"
    selected_path = out_dir / "selected_1024_patches.csv"
    features_flat_path = out_dir / "unicas_features_224_flat.pt"
    features_by_patch_path = out_dir / "unicas_features_224_by_1024_patch.pt"
    features_mean_path = out_dir / "unicas_features_1024_mean_pool.pt"
    summary_path = out_dir / "summary.json"

    pd.DataFrame(coord_rows).to_csv(coord_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    torch.save(features_flat, features_flat_path)
    torch.save(features_by_patch, features_by_patch_path)
    torch.save(features_mean, features_mean_path)

    preview_path = None
    if args.save_preview and not selected.empty:
        preview_path = out_dir / "preview_first_patch_4x4_224.jpg"
        save_preview(Path(selected.iloc[0]["image_path"]), preview_path, args)

    summary = {
        "patch_root": normalize_path(patch_root),
        "weights": normalize_path(weights_path),
        "out_dir": normalize_path(out_dir),
        "selected_1024_patches": int(len(selected)),
        "subpatches_per_1024_patch": int(subpatches_per_patch),
        "total_224_subpatches": int(features_flat.shape[0]),
        "feature_dim": int(features_flat.shape[1]),
        "input_tensor_shape": list(input_tensor.shape),
        "features_flat_shape": list(features_flat.shape),
        "features_by_patch_shape": list(features_by_patch.shape),
        "features_mean_shape": list(features_mean.shape),
        "coord_csv": normalize_path(coord_path),
        "selected_csv": normalize_path(selected_path),
        "features_flat": normalize_path(features_flat_path),
        "features_by_patch": normalize_path(features_by_patch_path),
        "features_mean_pool": normalize_path(features_mean_path),
        "preview": None if preview_path is None else normalize_path(preview_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
