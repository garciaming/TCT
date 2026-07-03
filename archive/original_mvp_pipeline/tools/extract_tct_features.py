import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from tiffslide import TiffSlide
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_unicas_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
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
    state = torch.load(weights_path, map_location="cpu")
    load_msg = model.load_state_dict(state, strict=False)
    print(f"Loaded UniCAS weights from {weights_path}")
    print(load_msg)
    model.eval()
    model.to(device)
    return model


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_transform(encoder_input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((encoder_input_size, encoder_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def make_uniform_grid(width: int, height: int, patch_size: int, stride: int, max_patches: int) -> list[tuple[int, int]]:
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    coords = [(x, y) for y in ys for x in xs]
    if max_patches > 0 and len(coords) > max_patches:
        indices = np.linspace(0, len(coords) - 1, max_patches)
        coords = [coords[int(round(i))] for i in indices]
    return coords


def read_patch(slide: TiffSlide, x: int, y: int, patch_size: int) -> Image.Image:
    patch = slide.read_region((int(x), int(y)), 0, (patch_size, patch_size))
    return patch.convert("RGB")


@torch.inference_mode()
def encode_slide(
    model: torch.nn.Module,
    slide_path: Path,
    out_dir: Path,
    transform: transforms.Compose,
    device: torch.device,
    patch_size: int,
    encoder_input_size: int,
    stride: int,
    max_patches: int,
    batch_size: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "images.pt"
    coords_path = out_dir / "coords.csv"

    with TiffSlide(str(slide_path)) as slide:
        width, height = slide.dimensions
        coords = make_uniform_grid(width, height, patch_size, stride, max_patches)

        features = []
        coord_rows = []
        batch = []
        batch_coords = []
        for x, y in tqdm(coords, desc=slide_path.name, leave=False):
            patch = read_patch(slide, x, y, patch_size)
            batch.append(transform(patch))
            batch_coords.append((x, y))
            if len(batch) == batch_size:
                x_tensor = torch.stack(batch).to(device, non_blocking=True)
                pred = model(x_tensor)
                if pred.ndim == 3:
                    pred = pred[:, 0]
                features.append(pred.detach().cpu())
                coord_rows.extend(batch_coords)
                batch = []
                batch_coords = []

        if batch:
            x_tensor = torch.stack(batch).to(device, non_blocking=True)
            pred = model(x_tensor)
            if pred.ndim == 3:
                pred = pred[:, 0]
            features.append(pred.detach().cpu())
            coord_rows.extend(batch_coords)

    if not features:
        raise RuntimeError(f"No patches encoded for {slide_path}")

    feature_tensor = torch.cat(features, dim=0).float()
    torch.save(feature_tensor, feature_path)
    pd.DataFrame(
        {
            "x": [x for x, _ in coord_rows],
            "y": [y for _, y in coord_rows],
            "level": 0,
            "patch_size": patch_size,
            "stride": stride,
        }
    ).to_csv(coords_path, index=False)

    return {
        "slide_path": normalize_path(slide_path),
        "feature_path": normalize_path(feature_path),
        "coords_path": normalize_path(coords_path),
        "num_patches": int(feature_tensor.shape[0]),
        "embed_dim": int(feature_tensor.shape[1]),
        "patch_size": int(patch_size),
        "encoder_input_size": int(encoder_input_size),
    }


def feature_is_valid(feature_path: Path, coords_path: Path, min_patches: int, patch_size: int, stride: int) -> bool:
    if not feature_path.is_file() or not coords_path.is_file():
        return False
    try:
        tensor = torch.load(feature_path, map_location="cpu")
        coords = pd.read_csv(coords_path)
    except Exception:
        return False
    if tensor.ndim != 2 or tensor.shape[0] < min_patches or torch.isnan(tensor).any():
        return False
    if "patch_size" not in coords.columns or "stride" not in coords.columns:
        return False
    return (
        int(coords["patch_size"].iloc[0]) == int(patch_size)
        and int(coords["stride"].iloc[0]) == int(stride)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract UniCAS patch features from TCT WSI files.")
    parser.add_argument("--manifest", default="data/datasets/tct_mvp/tct_mvp_all.csv")
    parser.add_argument("--weights", default="weights/pretrained/UniCAS.pth")
    parser.add_argument("--feature-root", default="features/Pathology_TCT_MVP_patch_512")
    parser.add_argument("--patch-size", type=int, default=1024, help="WSI crop size represented by one MIL patch.")
    parser.add_argument("--encoder-input-size", type=int, default=224, help="Resize each WSI patch to this size before UniCAS.")
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slides", type=int, default=0, help="0 means all slides.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(project_path(args.manifest), dtype=str, keep_default_na=False)
    if args.max_slides > 0:
        manifest = manifest.head(args.max_slides)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    model = build_unicas_model(project_path(args.weights), device)
    transform = make_transform(args.encoder_input_size)

    feature_root = project_path(args.feature_root)
    logs = []
    for _, row in manifest.iterrows():
        name = row["name"]
        slide_path = project_path(row["wsi_path"])
        out_dir = feature_root / name / "torch"
        feature_path = out_dir / "images.pt"
        coords_path = out_dir / "coords.csv"

        if args.skip_existing and feature_is_valid(
            feature_path,
            coords_path,
            min_patches=max(1, args.max_patches),
            patch_size=args.patch_size,
            stride=args.stride,
        ):
            print(f"skip existing: {name}")
            logs.append(
                {
                    "slide_path": normalize_path(slide_path),
                    "feature_path": normalize_path(feature_path),
                    "coords_path": normalize_path(coords_path),
                    "patch_size": int(args.patch_size),
                    "encoder_input_size": int(args.encoder_input_size),
                    "skipped": True,
                }
            )
            continue

        if not slide_path.is_file():
            raise FileNotFoundError(slide_path)

        print(f"Encoding {name}: {slide_path}")
        log = encode_slide(
            model=model,
            slide_path=slide_path,
            out_dir=out_dir,
            transform=transform,
            device=device,
            patch_size=args.patch_size,
            encoder_input_size=args.encoder_input_size,
            stride=args.stride,
            max_patches=args.max_patches,
            batch_size=args.batch_size,
        )
        log["name"] = name
        logs.append(log)
        print(json.dumps(log, ensure_ascii=False))

    log_path = feature_root / "feature_manifest.json"
    feature_root.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
