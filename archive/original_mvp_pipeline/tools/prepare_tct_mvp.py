import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ["NILM/Benign", "ASC-US", "LSIL", "ASC-H", "HSIL"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
HIGH_GRADE = {"ASC-H", "HSIL"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def grade_from_label(label: str) -> int:
    if label == "NILM/Benign":
        return 0
    if label in HIGH_GRADE:
        return 2
    return 1


def build_rows(metadata: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in metadata.iterrows():
        label = str(row.get("diagnosis_label_std", "")).strip()
        if label not in LABEL_TO_ID:
            continue

        slide_filename = str(row.get("slide_filename", "")).strip()
        slide_stem = str(row.get("slide_stem", "")).strip()
        if not slide_stem:
            slide_stem = Path(slide_filename).stem

        slide_relative_path = str(row.get("slide_relative_path", "")).strip()
        wsi_path = dataset_root / slide_relative_path

        case_id = str(row.get("case_id", "")).strip() or slide_stem
        split_group_id = str(row.get("split_group_id", "")).strip() or case_id

        is_abnormal = int(label != "NILM/Benign")
        rows.append(
            {
                "name": slide_stem,
                "code": case_id,
                "case_id": case_id,
                "split_group_id": split_group_id,
                "slide_filename": slide_filename,
                "slide_relative_path": slide_relative_path,
                "wsi_path": normalize_path(wsi_path),
                "diagnosis_label_std": label,
                "diagnosis_raw": str(row.get("diagnosis_raw", "")).strip(),
                "tct_label_id": LABEL_TO_ID[label],
                "tct_binary": is_abnormal,
                "cancer": is_abnormal,
                "cancer_grade": grade_from_label(label),
                "task_id": 0,
                "image_quality_score": str(row.get("image_quality_score", "")).strip(),
                "specimen_adequacy": str(row.get("specimen_adequacy", "")).strip(),
                "hpv_text_mentioned": str(row.get("hpv_text_mentioned", "")).strip(),
                "match_status": str(row.get("match_status", "")).strip(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No usable rows after filtering standardized TCT labels")
    out = out.drop_duplicates(subset=["name"]).reset_index(drop=True)
    return out


def split_by_group(
    data: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = data.copy()
    data["split"] = "train"

    groups = (
        data.groupby("split_group_id", dropna=False)
        .agg(tct_binary=("tct_binary", "max"))
        .reset_index()
    )

    split_for_group = {}
    for label_value, group_df in groups.groupby("tct_binary"):
        group_ids = group_df["split_group_id"].to_numpy(copy=True)
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if val_fraction > 0 and n >= 2:
            n_val = max(1, n_val)
        if test_fraction > 0 and n >= 3:
            n_test = max(1, n_test)
        if n_val + n_test >= n:
            n_val = max(0, n - n_test - 1)

        test_groups = set(group_ids[:n_test])
        val_groups = set(group_ids[n_test : n_test + n_val])
        for group_id in group_ids:
            if group_id in test_groups:
                split_for_group[group_id] = "test"
            elif group_id in val_groups:
                split_for_group[group_id] = "val"
            else:
                split_for_group[group_id] = "train"

    data["split"] = data["split_group_id"].map(split_for_group).fillna("train")
    return data


def write_split_files(data: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ["all", "train", "val", "test"]:
        if split == "all":
            subset = data
        else:
            subset = data[data["split"] == split]
        path = out_dir / f"tct_mvp_{split}.csv"
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        paths[split] = normalize_path(path.resolve())
    return paths


def summarize(data: pd.DataFrame, paths: dict) -> dict:
    summary = {
        "rows": int(len(data)),
        "paths": paths,
        "labels": LABEL_TO_ID,
        "split_counts": data["split"].value_counts().to_dict(),
        "binary_counts_by_split": {
            split: subset["tct_binary"].value_counts().to_dict()
            for split, subset in data.groupby("split")
        },
        "label_counts_by_split": {
            split: subset["diagnosis_label_std"].value_counts().to_dict()
            for split, subset in data.groupby("split")
        },
        "missing_wsi_paths": data.loc[
            ~data["wsi_path"].map(lambda p: Path(p).is_file()), "wsi_path"
        ].tolist(),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TCT WSI metadata for the MVP pipeline.")
    parser.add_argument(
        "--metadata",
        default="data/datasets/tct_mvp/tct_mvp_all.csv",
        help="Input metadata CSV.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/datasets",
        help="Root directory used to resolve slide_relative_path.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/datasets/tct_mvp",
        help="Output directory for MVP CSV files.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = project_path(args.metadata)
    dataset_root = project_path(args.dataset_root)
    out_dir = project_path(args.out_dir)

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    data = build_rows(metadata, dataset_root)
    data = split_by_group(data, args.val_fraction, args.test_fraction, args.seed)
    paths = write_split_files(data, out_dir)

    summary = summarize(data, paths)
    summary_path = out_dir / "tct_mvp_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
