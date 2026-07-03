import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def rel_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_catalog(args: argparse.Namespace) -> dict:
    project_root = Path(args.project_root).resolve()
    mvp_root = Path(args.mvp_root).resolve()
    source_csv = (mvp_root / args.source_csv).resolve()
    feature_root = (mvp_root / args.feature_root).resolve()

    all_df = read_csv(source_csv)

    manifests_dir = mvp_root / "data" / "manifests"
    labels_dir = mvp_root / "data" / "labels" / "files"
    docs_dir = mvp_root / "docs"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    wsi_columns = [
        "name",
        "code",
        "case_id",
        "split_group_id",
        "slide_filename",
        "slide_relative_path",
        "wsi_path",
        "split",
        "image_quality_score",
        "specimen_adequacy",
        "match_status",
    ]
    wsi_manifest = all_df[wsi_columns].copy()
    wsi_manifest["wsi_exists"] = wsi_manifest["wsi_path"].map(lambda p: Path(p).is_file())
    wsi_manifest_path = manifests_dir / "wsi_manifest.csv"
    write_csv(wsi_manifest, wsi_manifest_path)

    feature_rows = []
    for _, row in all_df.iterrows():
        name = row["name"]
        feature_path = feature_root / name / "torch" / "images.pt"
        coords_path = feature_root / name / "torch" / "coords.csv"
        feature_rows.append(
            {
                "name": name,
                "code": row["code"],
                "case_id": row["case_id"],
                "split": row["split"],
                "feature_path": normalize_path(feature_path),
                "coords_path": normalize_path(coords_path),
                "feature_exists": feature_path.is_file(),
                "coords_exists": coords_path.is_file(),
            }
        )
    feature_manifest = pd.DataFrame(feature_rows)
    feature_manifest_path = manifests_dir / "feature_manifest_512.csv"
    write_csv(feature_manifest, feature_manifest_path)

    split_manifest = all_df[
        ["name", "code", "case_id", "split_group_id", "split", "diagnosis_label_std", "cancer"]
    ].copy()
    split_manifest_path = manifests_dir / "split_manifest.csv"
    write_csv(split_manifest, split_manifest_path)

    binary_labels = all_df[
        [
            "name",
            "code",
            "case_id",
            "split",
            "diagnosis_label_std",
            "cancer",
            "tct_binary",
            "cancer_grade",
        ]
    ].copy()
    binary_labels["label_name"] = binary_labels["cancer"].map({"0": "normal", "1": "abnormal"})
    binary_labels_path = labels_dir / "tct_binary_labels.csv"
    write_csv(binary_labels, binary_labels_path)

    multiclass_labels = all_df[
        ["name", "code", "case_id", "split", "diagnosis_label_std", "tct_label_id"]
    ].copy()
    multiclass_labels_path = labels_dir / "tct_multiclass_labels.csv"
    write_csv(multiclass_labels, multiclass_labels_path)

    unicas_labels = all_df[
        [
            "name",
            "code",
            "cancer",
            "cancer_grade",
            "task_id",
            "diagnosis_label_std",
            "split",
        ]
    ].copy()
    unicas_labels_path = labels_dir / "unicas_slide_level_labels.csv"
    write_csv(unicas_labels, unicas_labels_path)

    diagnosis_text = all_df[
        ["name", "code", "case_id", "split", "diagnosis_label_std", "diagnosis_raw"]
    ].copy()
    diagnosis_text_path = labels_dir / "diagnosis_text_labels.csv"
    write_csv(diagnosis_text, diagnosis_text_path)

    catalog_rows = [
        {
            "category": "data",
            "name": "WSI original file manifest",
            "path": rel_path(wsi_manifest_path, project_root),
            "description": "Original TCT WSI file paths and case metadata.",
            "rows": len(wsi_manifest),
        },
        {
            "category": "data",
            "name": "UniCAS 512-patch feature manifest",
            "path": rel_path(feature_manifest_path, project_root),
            "description": "Per-slide 512-patch feature tensor and coordinate file paths.",
            "rows": len(feature_manifest),
        },
        {
            "category": "data",
            "name": "Train/val/test split manifest",
            "path": rel_path(split_manifest_path, project_root),
            "description": "Split assignment for each slide.",
            "rows": len(split_manifest),
        },
        {
            "category": "label",
            "name": "TCT binary labels",
            "path": rel_path(binary_labels_path, project_root),
            "description": "0=NILM/Benign, 1=ASC-US/LSIL/ASC-H/HSIL.",
            "rows": len(binary_labels),
        },
        {
            "category": "label",
            "name": "TCT multiclass labels",
            "path": rel_path(multiclass_labels_path, project_root),
            "description": "NILM/Benign, ASC-US, LSIL, ASC-H, HSIL mapped to ids 0-4.",
            "rows": len(multiclass_labels),
        },
        {
            "category": "label",
            "name": "UniCAS-compatible slide-level labels",
            "path": rel_path(unicas_labels_path, project_root),
            "description": "Columns compatible with the UniCAS slide-level demo style.",
            "rows": len(unicas_labels),
        },
        {
            "category": "label",
            "name": "Raw diagnosis text labels",
            "path": rel_path(diagnosis_text_path, project_root),
            "description": "Standardized label plus raw diagnostic text.",
            "rows": len(diagnosis_text),
        },
    ]
    catalog = pd.DataFrame(catalog_rows)
    catalog_path = manifests_dir / "DATA_LABEL_CATALOG.csv"
    write_csv(catalog, catalog_path)

    summary = {
        "total_slides": int(len(all_df)),
        "manifests_dir": rel_path(manifests_dir, project_root),
        "labels_dir": rel_path(labels_dir, project_root),
        "catalog_path": rel_path(catalog_path, project_root),
        "split_counts": all_df["split"].value_counts().to_dict(),
        "binary_label_counts": all_df["cancer"].value_counts().to_dict(),
        "multiclass_label_counts": all_df["diagnosis_label_std"].value_counts().to_dict(),
        "missing_wsi_files": int((~wsi_manifest["wsi_exists"]).sum()),
        "missing_feature_files": int((~feature_manifest["feature_exists"]).sum()),
        "missing_coord_files": int((~feature_manifest["coords_exists"]).sum()),
    }
    summary_path = manifests_dir / "DATA_LABEL_CATALOG.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_path = docs_dir / "DATA_LABEL目录.md"
    readme_path.write_text(build_markdown(catalog, summary), encoding="utf-8")

    return summary


def build_markdown(catalog: pd.DataFrame, summary: dict) -> str:
    rows = []
    for _, row in catalog.iterrows():
        rows.append(
            f"| {row['category']} | {row['name']} | `{row['path']}` | {row['rows']} | {row['description']} |"
        )

    return "\n".join(
        [
            "# MVP 数据文件与标签文件目录",
            "",
            "## 概览",
            "",
            f"- 总 WSI 数量：{summary['total_slides']}",
            f"- 数据目录文件：`{summary['manifests_dir']}`",
            f"- 标签文件目录：`{summary['labels_dir']}`",
            f"- 缺失原始 WSI：{summary['missing_wsi_files']}",
            f"- 缺失特征文件：{summary['missing_feature_files']}",
            f"- 缺失坐标文件：{summary['missing_coord_files']}",
            "",
            "## 文件目录",
            "",
            "| 类型 | 名称 | 路径 | 行数 | 说明 |",
            "| --- | --- | --- | ---: | --- |",
            *rows,
            "",
            "## 标签定义",
            "",
            "二分类标签：",
            "",
            "```text",
            "0 = normal = NILM/Benign",
            "1 = abnormal = ASC-US / LSIL / ASC-H / HSIL",
            "```",
            "",
            "多分类标签：",
            "",
            "```text",
            "0 = NILM/Benign",
            "1 = ASC-US",
            "2 = LSIL",
            "3 = ASC-H",
            "4 = HSIL",
            "```",
            "",
            "## 当前划分",
            "",
            "```json",
            json.dumps(summary["split_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MVP data and label file catalogs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mvp-root", default=".")
    parser.add_argument("--source-csv", default="data/datasets/tct_mvp/tct_mvp_all.csv")
    parser.add_argument("--feature-root", default="features/Pathology_TCT_MVP_patch_512")
    return parser.parse_args()


def main() -> None:
    summary = build_catalog(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
