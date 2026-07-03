import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tiffslide import TiffSlide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.attention_mil import AttentionMIL, choose_device, uniform_select


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def selected_indices(num_items: int, max_items: int) -> torch.Tensor:
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


def crop_thumbnail(wsi_path: Path, x: int, y: int, size: int, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with TiffSlide(str(wsi_path)) as slide:
        img = slide.read_region((int(x), int(y)), 0, (size, size)).convert("RGB")
    img.save(save_path)


@torch.inference_mode()
def export_topk(args: argparse.Namespace) -> pd.DataFrame:
    device = choose_device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    model, ckpt_args = load_model(checkpoint_path, device)
    feature_root = project_path(args.feature_root or ckpt_args.get("feature_root", "features/Pathology_TCT_MVP_patch_512"))
    max_patches = int(args.max_patches or ckpt_args.get("max_patches", 512))
    label_column = args.label_column or ckpt_args.get("label_column", "cancer")

    csv_path_arg = project_path(args.csv)
    data = pd.read_csv(csv_path_arg, dtype=str, keep_default_na=False)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in data.iterrows():
        name = row["name"]
        feature_path = feature_root / name / "torch" / "images.pt"
        coords_path = feature_root / name / "torch" / "coords.csv"
        if not feature_path.is_file() or not coords_path.is_file():
            continue

        features_raw = torch.load(feature_path, map_location="cpu").float()
        coords_raw = pd.read_csv(coords_path)
        indices = selected_indices(features_raw.shape[0], max_patches)
        features = features_raw[indices]
        coords = coords_raw.iloc[indices.numpy()].reset_index(drop=True)

        x = features.unsqueeze(0).to(device)
        mask = torch.ones(1, features.shape[0], dtype=torch.bool, device=device)
        logits, attn = model(x, mask)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        pred = int(probs.argmax().item())
        label = int(row[label_column])

        top_values, top_indices = torch.topk(attn[0].detach().cpu(), k=min(args.top_k, features.shape[0]))
        for rank, (attn_value, patch_idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            coord = coords.iloc[int(patch_idx)]
            thumb_path = ""
            if args.save_thumbnails:
                thumb_path_obj = out_dir / "thumbnails" / name / f"rank_{rank:02d}_x{int(coord.x)}_y{int(coord.y)}.png"
                crop_thumbnail(project_path(row["wsi_path"]), int(coord.x), int(coord.y), args.roi_size, thumb_path_obj)
                thumb_path = normalize_path(thumb_path_obj.resolve())

            rows.append(
                {
                    "name": name,
                    "code": row.get("code", ""),
                    "diagnosis_label_std": row.get("diagnosis_label_std", ""),
                    "label": label,
                    "pred": pred,
                    "prob_0": float(probs[0].item()),
                    "prob_1": float(probs[1].item()) if probs.numel() > 1 else "",
                    "rank": rank,
                    "attention": float(attn_value),
                    "x": int(coord.x),
                    "y": int(coord.y),
                    "level": int(coord.level),
                    "patch_size": int(coord.patch_size),
                    "roi_size": int(args.roi_size),
                    "thumbnail_path": thumb_path,
                    "wsi_path": row["wsi_path"],
                }
            )

    result = pd.DataFrame(rows)
    csv_path = out_dir / "topk_rois.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    meta = {
        "csv": normalize_path(csv_path_arg.resolve()),
        "checkpoint": normalize_path(checkpoint_path.resolve()),
        "feature_root": normalize_path(feature_root.resolve()),
        "top_k": args.top_k,
        "roi_size": args.roi_size,
        "rows": int(len(result)),
        "output": normalize_path(csv_path.resolve()),
    }
    (out_dir / "topk_rois_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-K patch coordinates from a trained MIL checkpoint.")
    parser.add_argument("--csv", default="data/datasets/tct_mvp/tct_mvp_all.csv")
    parser.add_argument("--checkpoint", default="runs/tct_mvp_512/best.pt")
    parser.add_argument("--feature-root", default="features/Pathology_TCT_MVP_patch_512")
    parser.add_argument("--out-dir", default="runs/tct_mvp_512/topk_rois_all")
    parser.add_argument("--label-column", default="")
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--roi-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-thumbnails", action="store_true")
    return parser.parse_args()


def main() -> None:
    export_topk(parse_args())


if __name__ == "__main__":
    main()
