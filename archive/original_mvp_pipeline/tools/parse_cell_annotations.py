import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from tiffslide import TiffSlide


TILE_RE = re.compile(r"^(?P<case>[A-Za-z]?\d+)-(?P<a>\d+)x(?P<b>\d+)$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def read_full_metadata(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    data["case_upper"] = data["case_id"].str.upper()
    return data


def read_dealed_case_info(excel_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name="dealed_case_info", dtype=str).fillna("")
    headers = list(raw.iloc[0])
    data = raw.iloc[1:].copy()
    data.columns = headers
    data = data[data["Patient_ID"].astype(str).str.strip() != ""].copy()
    data["case_upper"] = data["Patient_ID"].astype(str).str.upper().str.strip()
    return data


def read_cell_annotations(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str).fillna("")
    data = raw[(raw["filename"] != "") & (raw["type of leision"] != "")].copy()
    data = data.rename(
        columns={
            "type of leision": "cell_label",
            "Location": "local_xmin",
            "Unnamed: 6": "local_ymin",
            "Unnamed: 7": "local_xmax",
            "Unnamed: 8": "local_ymax",
        }
    )
    for col in ["local_xmin", "local_ymin", "local_xmax", "local_ymax"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["local_xmin", "local_ymin", "local_xmax", "local_ymax"]).copy()
    return data


def choose_slide_row(case_rows: pd.DataFrame) -> pd.Series | None:
    if case_rows.empty:
        return None
    if "train_use_flag" in case_rows.columns:
        preferred = case_rows[case_rows["train_use_flag"].isin(["yes", "review"])]
        if not preferred.empty:
            return preferred.iloc[0]
    if "match_status" in case_rows.columns:
        matched = case_rows[case_rows["match_status"].isin(["direct_match", "assisted_match"])]
        if not matched.empty:
            return matched.iloc[0]
    return case_rows.iloc[0]


def parse_annotations(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    excel_path = project_path(args.excel)
    metadata = read_full_metadata(project_path(args.full_metadata))
    case_info = read_dealed_case_info(excel_path)
    case_info_by_case = case_info.set_index("case_upper").to_dict("index")
    annotations = read_cell_annotations(excel_path, args.sheet)

    rows = []
    skipped = []
    for _, ann in annotations.iterrows():
        match = TILE_RE.match(str(ann["filename"]).strip())
        if not match:
            skipped.append({"filename": ann["filename"], "reason": "filename_parse_failed"})
            continue
        case_id = match.group("case")
        case_upper = case_id.upper()
        tile_a = int(match.group("a"))
        tile_b = int(match.group("b"))

        slide_row = choose_slide_row(metadata[metadata["case_upper"] == case_upper])
        if slide_row is None:
            skipped.append({"filename": ann["filename"], "reason": "case_not_in_metadata"})
            continue

        wsi_path = project_path(args.dataset_root) / str(slide_row["slide_relative_path"])
        if not wsi_path.is_file():
            skipped.append({"filename": ann["filename"], "reason": "wsi_missing"})
            continue

        with TiffSlide(str(wsi_path)) as slide:
            width, height = slide.dimensions

        info = case_info_by_case.get(case_upper, {})
        data_num = str(info.get("Data_num", "")).strip()
        grid_size = int(round(math.sqrt(int(data_num)))) if data_num.isdigit() else None

        base = args.tile_index_base
        # xy convention: filename case-xTile x yTile.
        xy_x_offset = (tile_a - base) * args.tile_size
        xy_y_offset = (tile_b - base) * args.tile_size
        # yx convention: filename case-yTile x xTile. Kept for audit because source naming is not documented.
        yx_x_offset = (tile_b - base) * args.tile_size
        yx_y_offset = (tile_a - base) * args.tile_size

        local_xmin = int(ann["local_xmin"])
        local_ymin = int(ann["local_ymin"])
        local_xmax = int(ann["local_xmax"])
        local_ymax = int(ann["local_ymax"])

        xy_box = [
            xy_x_offset + local_xmin,
            xy_y_offset + local_ymin,
            xy_x_offset + local_xmax,
            xy_y_offset + local_ymax,
        ]
        yx_box = [
            yx_x_offset + local_xmin,
            yx_y_offset + local_ymin,
            yx_x_offset + local_xmax,
            yx_y_offset + local_ymax,
        ]

        def in_bounds(box: list[int]) -> bool:
            return box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height

        chosen_box = xy_box if args.tile_order == "xy" else yx_box
        rows.append(
            {
                "case_id": case_id,
                "case_upper": case_upper,
                "slide_name": slide_row.get("slide_stem", slide_row.get("name", "")),
                "slide_filename": slide_row["slide_filename"],
                "wsi_path": normalize_path(wsi_path),
                "diagnosis_label_std": slide_row["diagnosis_label_std"],
                "train_use_flag": slide_row.get("train_use_flag", ""),
                "tile_name": ann["filename"],
                "tile_a": tile_a,
                "tile_b": tile_b,
                "tile_size": args.tile_size,
                "tile_index_base": base,
                "tile_order": args.tile_order,
                "grid_size_from_data_num": grid_size,
                "slide_width": width,
                "slide_height": height,
                "cell_label": ann["cell_label"],
                "local_xmin": local_xmin,
                "local_ymin": local_ymin,
                "local_xmax": local_xmax,
                "local_ymax": local_ymax,
                "global_xmin": chosen_box[0],
                "global_ymin": chosen_box[1],
                "global_xmax": chosen_box[2],
                "global_ymax": chosen_box[3],
                "global_cx": (chosen_box[0] + chosen_box[2]) / 2,
                "global_cy": (chosen_box[1] + chosen_box[3]) / 2,
                "xy_global_xmin": xy_box[0],
                "xy_global_ymin": xy_box[1],
                "xy_global_xmax": xy_box[2],
                "xy_global_ymax": xy_box[3],
                "yx_global_xmin": yx_box[0],
                "yx_global_ymin": yx_box[1],
                "yx_global_xmax": yx_box[2],
                "yx_global_ymax": yx_box[3],
                "xy_in_bounds": in_bounds(xy_box),
                "yx_in_bounds": in_bounds(yx_box),
            }
        )

    out = pd.DataFrame(rows)
    summary = {
        "rows": int(len(out)),
        "skipped": skipped,
        "tile_order": args.tile_order,
        "tile_size": args.tile_size,
        "tile_index_base": args.tile_index_base,
        "cell_label_counts": out["cell_label"].value_counts().to_dict() if not out.empty else {},
        "case_counts": out["case_upper"].value_counts().to_dict() if not out.empty else {},
        "xy_in_bounds_rate": float(out["xy_in_bounds"].mean()) if not out.empty else None,
        "yx_in_bounds_rate": float(out["yx_in_bounds"].mean()) if not out.empty else None,
    }
    return out, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse DATA_INFO_trail cell boxes into WSI global coordinates.")
    parser.add_argument("--excel", default="data/datasets/TCT_Slides/DATA_INFO_trail.xlsx")
    parser.add_argument("--sheet", default="cell_analysi-1 (2)")
    parser.add_argument("--full-metadata", default="data/datasets/tct_mvp/tct_mvp_all.csv")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--out-dir", default="annotations")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--tile-index-base", type=int, default=1)
    parser.add_argument("--tile-order", choices=["xy", "yx"], default="xy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes, summary = parse_annotations(args)
    boxes_path = out_dir / "cell_boxes_global.csv"
    summary_path = out_dir / "cell_boxes_global_summary.json"
    boxes.to_csv(boxes_path, index=False, encoding="utf-8-sig")
    summary["boxes_path"] = normalize_path(boxes_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
