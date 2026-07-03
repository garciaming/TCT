import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tiffslide import TiffSlide
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return device_arg


def crop_roi(row: pd.Series, roi_size: int) -> Image.Image:
    thumbnail_path = str(row.get("thumbnail_path", "")).strip()
    if thumbnail_path:
        thumb = Path(thumbnail_path)
        if thumb.is_file():
            image = Image.open(thumb).convert("RGB")
            if image.size == (roi_size, roi_size):
                return image

    wsi_path = project_path(row["wsi_path"])
    x = int(float(row["x"]))
    y = int(float(row["y"]))
    with TiffSlide(str(wsi_path)) as slide:
        return slide.read_region((x, y), 0, (roi_size, roi_size)).convert("RGB")


def class_name(model: YOLO, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def maybe_save_annotated(result, save_path: Path) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot()
    Image.fromarray(plotted[..., ::-1]).save(save_path)
    return normalize_path(save_path)


def run_detector(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = choose_device(args.device)
    model = YOLO(str(project_path(args.checkpoint)))
    topk = pd.read_csv(project_path(args.topk_csv), dtype=str, keep_default_na=False)
    if args.max_rois > 0:
        topk = topk.head(args.max_rois).copy()

    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_rows = []
    score_rows = []
    for _, row in topk.iterrows():
        image = crop_roi(row, args.roi_size)
        results = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        roi_x = int(float(row["x"]))
        roi_y = int(float(row["y"]))
        scores = []

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for det_idx, (box, score, cls_id_float) in enumerate(zip(xyxy, confs, classes), start=1):
                cls_id = int(cls_id_float)
                x1, y1, x2, y2 = [float(v) for v in box]
                scores.append(float(score))
                detection_rows.append(
                    {
                        "name": row["name"],
                        "roi_rank": int(row["rank"]),
                        "det_index": det_idx,
                        "class_id": cls_id,
                        "label_id": cls_id + 1,
                        "class_name": class_name(model, cls_id),
                        "score": float(score),
                        "roi_xmin": x1,
                        "roi_ymin": y1,
                        "roi_xmax": x2,
                        "roi_ymax": y2,
                        "global_xmin": roi_x + x1,
                        "global_ymin": roi_y + y1,
                        "global_xmax": roi_x + x2,
                        "global_ymax": roi_y + y2,
                        "roi_x": roi_x,
                        "roi_y": roi_y,
                        "wsi_path": row["wsi_path"],
                    }
                )

        annotated_path = ""
        if args.save_annotated:
            annotated_path = maybe_save_annotated(
                result,
                out_dir / "annotated" / row["name"] / f"rank_{int(row['rank']):02d}_x{roi_x}_y{roi_y}.jpg",
            )

        score_rows.append(
            {
                "name": row["name"],
                "roi_rank": int(row["rank"]),
                "attention": float(row.get("attention", 0.0) or 0.0),
                "detector_score_max": max(scores) if scores else 0.0,
                "detector_score_sum": float(sum(scores)),
                "detector_count": len(scores),
                "x": roi_x,
                "y": roi_y,
                "roi_size": args.roi_size,
                "thumbnail_path": row.get("thumbnail_path", ""),
                "annotated_path": annotated_path,
                "wsi_path": row["wsi_path"],
            }
        )

    detections = pd.DataFrame(detection_rows)
    roi_scores = pd.DataFrame(score_rows)
    detections_path = out_dir / "roi_detections.csv"
    scores_path = out_dir / "roi_scores.csv"
    detections.to_csv(detections_path, index=False, encoding="utf-8-sig")
    roi_scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    meta = {
        "checkpoint": normalize_path(project_path(args.checkpoint)),
        "topk_csv": normalize_path(project_path(args.topk_csv)),
        "detections": normalize_path(detections_path),
        "roi_scores": normalize_path(scores_path),
        "detection_rows": int(len(detections)),
        "roi_rows": int(len(roi_scores)),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "roi_size": args.roi_size,
        "device": device,
    }
    (out_dir / "roi_detections_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return detections, roi_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained YOLO11 detector on exported Top-K WSI ROIs.")
    parser.add_argument("--checkpoint", default="runs/yolo11_cell_detector/yolo11m_binary_1024_guidance/weights/best.pt")
    parser.add_argument("--topk-csv", default="runs/tct_mvp_512/topk_rois_all/topk_rois.csv")
    parser.add_argument("--out-dir", default="runs/tct_mvp_512/yolo11_binary_roi_guidance")
    parser.add_argument("--roi-size", type=int, default=1024)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-rois", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-annotated", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_detector(parse_args())


if __name__ == "__main__":
    main()
