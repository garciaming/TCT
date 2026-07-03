import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageStat
from tiffslide import TiffSlide
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def image_stats(image) -> dict:
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    stat_rgb = ImageStat.Stat(rgb)
    stat_hsv = ImageStat.Stat(hsv)
    gray = rgb.convert("L")
    stat_gray = ImageStat.Stat(gray)
    gray_mean = float(stat_gray.mean[0])
    gray_std = float(stat_gray.stddev[0])
    saturation_mean = float(stat_hsv.mean[1])
    # Heuristic only: high brightness + very low variation/saturation usually means empty glass/background.
    background_like = gray_mean > 220 and gray_std < 18 and saturation_mean < 28
    tissue_score = float(gray_std + 0.25 * saturation_mean)
    return {
        "rgb_mean_r": float(stat_rgb.mean[0]),
        "rgb_mean_g": float(stat_rgb.mean[1]),
        "rgb_mean_b": float(stat_rgb.mean[2]),
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "saturation_mean": saturation_mean,
        "tissue_score": tissue_score,
        "background_like": background_like,
    }


def qc_regions(rows: pd.DataFrame, size_column: str, out_path: Path, desc: str) -> pd.DataFrame:
    results = []
    grouped = rows.groupby("wsi_path", dropna=False)
    for wsi_path, group in tqdm(grouped, desc=desc):
        with TiffSlide(str(wsi_path)) as slide:
            for _, row in group.iterrows():
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                size = int(float(row[size_column]))
                image = slide.read_region((x, y), 0, (size, size)).convert("RGB")
                stat = image_stats(image)
                results.append(row.to_dict() | stat)

    out = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def build_sampled_patch_rows(manifest_path: Path, feature_manifest_path: Path, feature_root: Path | None = None) -> pd.DataFrame:
    slide_meta = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    meta_by_name = slide_meta.set_index("name").to_dict("index")

    if feature_root is not None:
        feature_rows = []
        for _, meta in slide_meta.iterrows():
            name = meta["name"]
            feature_rows.append(
                {
                    "name": name,
                    "code": meta.get("code", ""),
                    "case_id": meta.get("case_id", ""),
                    "split": meta.get("split", ""),
                    "coords_path": str(feature_root / name / "torch" / "coords.csv"),
                }
            )
        feature_manifest = pd.DataFrame(feature_rows)
    else:
        feature_manifest = pd.read_csv(feature_manifest_path, dtype=str, keep_default_na=False)

    rows = []
    for _, feature_row in feature_manifest.iterrows():
        name = feature_row["name"]
        coords_path = Path(feature_row["coords_path"])
        if not coords_path.is_file() or name not in meta_by_name:
            continue
        meta = meta_by_name[name]
        coords = pd.read_csv(coords_path)
        for patch_index, coord in coords.iterrows():
            rows.append(
                {
                    "name": name,
                    "code": feature_row.get("code", ""),
                    "case_id": feature_row.get("case_id", ""),
                    "split": feature_row.get("split", ""),
                    "diagnosis_label_std": meta.get("diagnosis_label_std", ""),
                    "patch_index": patch_index,
                    "x": int(coord["x"]),
                    "y": int(coord["y"]),
                    "patch_size": int(coord["patch_size"]),
                    "stride": int(coord["stride"]),
                    "wsi_path": meta["wsi_path"],
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("name", "size"),
            background_like_n=("background_like", "sum"),
            background_like_rate=("background_like", "mean"),
            tissue_score_mean=("tissue_score", "mean"),
            tissue_score_median=("tissue_score", "median"),
            gray_mean_mean=("gray_mean", "mean"),
            gray_std_mean=("gray_std", "mean"),
            saturation_mean_mean=("saturation_mean", "mean"),
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute image QC statistics for sampled patches and top-K ROIs.")
    parser.add_argument("--manifest", default="data/datasets/tct_mvp/tct_mvp_all.csv")
    parser.add_argument("--feature-manifest", default="data/manifests/feature_manifest_512.csv")
    parser.add_argument("--feature-root", default="", help="Optional feature root; overrides --feature-manifest for sampled patch QC.")
    parser.add_argument("--topk-csv", default="runs/tct_mvp_512/topk_rois_all/topk_rois.csv")
    parser.add_argument("--out-dir", default="runs/tct_mvp_512/qc")
    parser.add_argument("--skip-sampled-patches", action="store_true")
    parser.add_argument("--skip-topk-rois", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if not args.skip_sampled_patches:
        feature_root = project_path(args.feature_root) if args.feature_root else None
        sampled_rows = build_sampled_patch_rows(project_path(args.manifest), project_path(args.feature_manifest), feature_root)
        sampled_qc = qc_regions(
            sampled_rows,
            size_column="patch_size",
            out_path=out_dir / "sampled_patch_qc.csv",
            desc="sampled patches QC",
        )
        sampled_summary = summarize(sampled_qc, ["split", "diagnosis_label_std"])
        sampled_summary.to_csv(out_dir / "sampled_patch_qc_summary.csv", index=False, encoding="utf-8-sig")
        outputs["sampled_patch_qc"] = normalize_path(out_dir / "sampled_patch_qc.csv")
        outputs["sampled_patch_qc_summary"] = normalize_path(out_dir / "sampled_patch_qc_summary.csv")

    if not args.skip_topk_rois:
        topk = pd.read_csv(project_path(args.topk_csv), dtype=str, keep_default_na=False)
        topk_qc = qc_regions(
            topk,
            size_column="roi_size",
            out_path=out_dir / "topk_roi_qc.csv",
            desc="top-k ROI QC",
        )
        topk_summary = summarize(topk_qc, ["rank"])
        topk_summary.to_csv(out_dir / "topk_roi_qc_summary_by_rank.csv", index=False, encoding="utf-8-sig")
        outputs["topk_roi_qc"] = normalize_path(out_dir / "topk_roi_qc.csv")
        outputs["topk_roi_qc_summary_by_rank"] = normalize_path(out_dir / "topk_roi_qc_summary_by_rank.csv")

    summary_path = out_dir / "qc_outputs.json"
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
