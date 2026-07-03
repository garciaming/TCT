import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def center_inside(center_x: float, center_y: float, roi: np.ndarray) -> bool:
    return roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3]


def evaluate_hits(topk: pd.DataFrame, boxes: pd.DataFrame, thresholds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    topk = topk.copy()
    boxes = boxes.copy()
    topk["rank"] = topk["rank"].astype(int)
    for col in ["x", "y", "roi_size"]:
        topk[col] = pd.to_numeric(topk[col])
    for col in ["global_xmin", "global_ymin", "global_xmax", "global_ymax", "global_cx", "global_cy"]:
        boxes[col] = pd.to_numeric(boxes[col])

    box_rows = []
    slide_rows = []
    topk_names = set(topk["name"])
    candidate_boxes = boxes[boxes["slide_name"].isin(topk_names)].copy()

    for (slide_name, threshold), _ in [
        ((slide_name, threshold), None)
        for slide_name in sorted(candidate_boxes["slide_name"].unique())
        for threshold in thresholds
    ]:
        slide_topk = topk[(topk["name"] == slide_name) & (topk["rank"] <= threshold)].copy()
        slide_boxes = candidate_boxes[candidate_boxes["slide_name"] == slide_name].copy()
        rois = []
        for _, roi in slide_topk.iterrows():
            x0 = float(roi["x"])
            y0 = float(roi["y"])
            size = float(roi["roi_size"])
            rois.append((int(roi["rank"]), np.array([x0, y0, x0 + size, y0 + size], dtype=float)))

        hit_count = 0
        for box_index, box in slide_boxes.iterrows():
            bbox = np.array(
                [box["global_xmin"], box["global_ymin"], box["global_xmax"], box["global_ymax"]],
                dtype=float,
            )
            cx = float(box["global_cx"])
            cy = float(box["global_cy"])
            best_iou = 0.0
            best_rank = None
            center_hit = False
            for rank, roi_box in rois:
                iou = box_iou(bbox, roi_box)
                if iou > best_iou:
                    best_iou = iou
                    best_rank = rank
                if center_inside(cx, cy, roi_box):
                    center_hit = True
                    if best_rank is None:
                        best_rank = rank

            hit = center_hit or best_iou > 0
            hit_count += int(hit)
            box_rows.append(
                {
                    "slide_name": slide_name,
                    "threshold": threshold,
                    "box_index": int(box_index),
                    "case_id": box["case_id"],
                    "tile_name": box["tile_name"],
                    "cell_label": box["cell_label"],
                    "hit": hit,
                    "center_hit": center_hit,
                    "best_iou": best_iou,
                    "best_rank": best_rank,
                }
            )

        n_boxes = len(slide_boxes)
        slide_rows.append(
            {
                "slide_name": slide_name,
                "threshold": threshold,
                "num_boxes": n_boxes,
                "num_rois": len(rois),
                "hit_boxes": hit_count,
                "hit_rate": hit_count / n_boxes if n_boxes else 0.0,
                "hit_any": hit_count > 0,
            }
        )

    return pd.DataFrame(box_rows), pd.DataFrame(slide_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether attention Top-K ROIs hit parsed cell boxes.")
    parser.add_argument("--topk-csv", default="runs/tct_mvp_512/topk_rois_all/topk_rois.csv")
    parser.add_argument("--boxes-csv", default="annotations/cell_boxes_global.csv")
    parser.add_argument("--out-dir", default="runs/tct_mvp_512/roi_hit_eval")
    parser.add_argument("--thresholds", default="10,20,50")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    topk = pd.read_csv(project_path(args.topk_csv), dtype=str, keep_default_na=False)
    boxes = pd.read_csv(project_path(args.boxes_csv), dtype=str, keep_default_na=False)
    box_hits, slide_hits = evaluate_hits(topk, boxes, thresholds)

    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    box_path = out_dir / "box_hit_details.csv"
    slide_path = out_dir / "slide_hit_summary.csv"
    box_hits.to_csv(box_path, index=False, encoding="utf-8-sig")
    slide_hits.to_csv(slide_path, index=False, encoding="utf-8-sig")

    overall = (
        slide_hits.groupby("threshold")
        .agg(
            slides=("slide_name", "nunique"),
            boxes=("num_boxes", "sum"),
            hit_boxes=("hit_boxes", "sum"),
            slide_hit_any_rate=("hit_any", "mean"),
        )
        .reset_index()
    )
    overall["box_hit_rate"] = overall["hit_boxes"] / overall["boxes"].replace(0, np.nan)
    overall_path = out_dir / "overall_hit_summary.csv"
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    result = {
        "box_hit_details": normalize_path(box_path),
        "slide_hit_summary": normalize_path(slide_path),
        "overall_hit_summary": normalize_path(overall_path),
        "thresholds": thresholds,
    }
    (out_dir / "roi_hit_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
