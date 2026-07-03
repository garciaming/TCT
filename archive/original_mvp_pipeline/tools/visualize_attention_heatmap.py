import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tiffslide import TiffSlide
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attention_mil import AttentionMIL, choose_device

DEFAULT_PATCH_SIZE = 1024


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def select_indices(num_items: int, max_items: int) -> torch.Tensor:
    if max_items <= 0 or num_items <= max_items:
        return torch.arange(num_items)
    return torch.linspace(0, num_items - 1, max_items).round().long()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[AttentionMIL, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = AttentionMIL(
        embed_dim=int(ckpt_args.get("embed_dim", 1024)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        num_classes=int(ckpt_args.get("num_classes", 2)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, ckpt_args


def heat_color(value: float) -> tuple[int, int, int, int]:
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        t = value / 0.5
        r = int(255 * t)
        g = int(255 * t)
        b = 255 - int(120 * t)
    else:
        t = (value - 0.5) / 0.5
        r = 255
        g = 255 - int(255 * t)
        b = 80 - int(80 * t)
    alpha = int(45 + 160 * value)
    return r, g, b, alpha


@torch.inference_mode()
def infer_attention(model: AttentionMIL, feature_path: Path, coords_path: Path, max_patches: int, device: torch.device):
    features_raw = torch.load(feature_path, map_location="cpu").float()
    coords_raw = pd.read_csv(coords_path)
    indices = select_indices(features_raw.shape[0], max_patches)
    features = features_raw[indices]
    coords = coords_raw.iloc[indices.numpy()].reset_index(drop=True)

    x = features.unsqueeze(0).to(device)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=device)
    logits, attn = model(x, mask)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    return coords, attn[0].detach().cpu().numpy(), probs


def draw_heatmap(row: pd.Series, coords: pd.DataFrame, attn: np.ndarray, probs: torch.Tensor, args: argparse.Namespace, out_dir: Path) -> dict:
    name = row["name"]
    wsi_path = project_path(row["wsi_path"])
    with TiffSlide(str(wsi_path)) as slide:
        width, height = slide.dimensions
        thumb = slide.get_thumbnail((args.thumbnail_size, args.thumbnail_size)).convert("RGB")

    sx = thumb.width / width
    sy = thumb.height / height
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    p_low, p_high = np.percentile(attn, [5, 95])
    denom = max(float(p_high - p_low), 1e-8)
    norm = np.clip((attn - p_low) / denom, 0, 1)
    patch_size = int(coords["patch_size"].iloc[0]) if "patch_size" in coords.columns and len(coords) else DEFAULT_PATCH_SIZE

    for coord, value in zip(coords.itertuples(index=False), norm):
        x0 = int(coord.x * sx)
        y0 = int(coord.y * sy)
        x1 = max(x0 + 1, int((coord.x + patch_size) * sx))
        y1 = max(y0 + 1, int((coord.y + patch_size) * sy))
        draw.rectangle([x0, y0, x1, y1], fill=heat_color(float(value)))

    top_indices = np.argsort(-attn)[: min(args.top_k, len(attn))]
    for rank, idx in enumerate(top_indices, start=1):
        coord = coords.iloc[int(idx)]
        x0 = int(coord.x * sx)
        y0 = int(coord.y * sy)
        x1 = max(x0 + 4, int((coord.x + patch_size) * sx))
        y1 = max(y0 + 4, int((coord.y + patch_size) * sy))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 230), width=2)
        draw.text((x0 + 2, y0 + 2), str(rank), fill=(255, 255, 255, 255))

    blended = Image.alpha_composite(thumb.convert("RGBA"), overlay).convert("RGB")
    heatmap_path = out_dir / "heatmaps" / f"{name}.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(heatmap_path)

    attention_rows = []
    for idx, (coord, score) in enumerate(zip(coords.itertuples(index=False), attn)):
        attention_rows.append(
            {
                "name": name,
                "patch_index": idx,
                "x": int(coord.x),
                "y": int(coord.y),
                "patch_size": patch_size,
                "attention": float(score),
                "attention_norm": float(norm[idx]),
            }
        )
    return {
        "name": name,
        "heatmap_path": normalize_path(heatmap_path),
        "prob_0": float(probs[0].item()),
        "prob_1": float(probs[1].item()) if probs.numel() > 1 else None,
        "pred": int(torch.argmax(probs).item()),
        "attention_rows": attention_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render WSI attention heatmaps from an Attention-MIL checkpoint.")
    parser.add_argument("--csv", default="data/datasets/tct_mvp/tct_mvp_all.csv")
    parser.add_argument("--checkpoint", default="runs/tct_mvp_512/best.pt")
    parser.add_argument("--feature-root", default="features/Pathology_TCT_MVP_patch_512")
    parser.add_argument("--out-dir", default="runs/tct_mvp_512/attention_heatmaps")
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--thumbnail-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, _ = load_model(project_path(args.checkpoint), device)
    data = pd.read_csv(project_path(args.csv), dtype=str, keep_default_na=False)
    feature_root = project_path(args.feature_root)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    all_attention_rows = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="heatmaps"):
        name = row["name"]
        feature_path = feature_root / name / "torch" / "images.pt"
        coords_path = feature_root / name / "torch" / "coords.csv"
        if not feature_path.is_file() or not coords_path.is_file():
            continue
        coords, attn, probs = infer_attention(model, feature_path, coords_path, args.max_patches, device)
        rendered = draw_heatmap(row, coords, attn, probs, args, out_dir)
        all_attention_rows.extend(rendered.pop("attention_rows"))
        summary_rows.append(row.to_dict() | rendered)

    summary = pd.DataFrame(summary_rows)
    attention_scores = pd.DataFrame(all_attention_rows)
    summary_path = out_dir / "heatmap_manifest.csv"
    scores_path = out_dir / "attention_scores.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    attention_scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "heatmap_manifest": normalize_path(summary_path),
                "attention_scores": normalize_path(scores_path),
                "heatmaps": normalize_path(out_dir / "heatmaps"),
                "rows": int(len(summary)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
