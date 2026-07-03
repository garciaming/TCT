from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path | None) -> Path | None:
    if path is None or str(path) == "":
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE).strip("_")
    return text or "unknown"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT = load_font(22)
FONT_SMALL = load_font(18)
FONT_TINY = load_font(15)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def patch_detections(detections: pd.DataFrame, patch_path: str) -> pd.DataFrame:
    if detections.empty or "patch_path" not in detections.columns:
        return detections.iloc[0:0].copy()
    return detections[detections["patch_path"].astype(str) == str(patch_path)].copy()


def draw_roi(row: pd.Series, det_rows: pd.DataFrame, out_path: Path) -> dict[str, Any]:
    patch_path = Path(str(row["patch_path"]))
    if not patch_path.is_file():
        return {
            "status": "missing_patch",
            "patch_path": str(patch_path),
            "out_path": normalize_path(out_path),
        }

    image = Image.open(patch_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size

    det_count = 0
    for _, det in det_rows.iterrows():
        x1 = to_float(det.get("x1"))
        y1 = to_float(det.get("y1"))
        x2 = to_float(det.get("x2"))
        y2 = to_float(det.get("y2"))
        conf = to_float(det.get("confidence"))

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        det_count += 1
        draw.rectangle([x1, y1, x2, y2], outline=(255, 90, 42, 255), width=4)
        draw_label(draw, (int(x1) + 4, int(max(2, y1 - 28))), f"{conf:.2f}", FONT_TINY)

    rank = to_int(row.get("rank"))
    attention = to_float(row.get("attention"))
    label = to_int(row.get("label"))
    pred = to_int(row.get("pred"))
    prob_pos = to_float(row.get("prob_1"))
    score_max = to_float(row.get("detector_score_max"))
    score_sum = to_float(row.get("detector_score_sum"))
    score_count = to_int(row.get("detector_count"), default=det_count)

    header = (
        f"slide={row.get('slide_name')}  rank={rank}  "
        f"label={label} pred={pred} prob_pos={prob_pos:.3f}"
    )
    subheader = (
        f"attention={attention:.6f}  boxes={score_count}  "
        f"score_max={score_max:.3f} score_sum={score_sum:.3f}"
    )
    draw.rectangle([0, 0, width, 72], fill=(255, 255, 255, 220))
    draw.text((12, 10), header, fill=(0, 0, 0), font=FONT_SMALL)
    draw.text((12, 40), subheader, fill=(255, 90, 42), font=FONT_SMALL)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=92)
    return {
        "status": "ok",
        "slide_name": row.get("slide_name"),
        "rank": rank,
        "patch_path": str(patch_path),
        "out_path": normalize_path(out_path),
        "detections": int(det_count),
    }


def make_contact_sheet(items: list[Path], title: str, out_path: Path, thumb_size: int) -> None:
    if not items:
        return

    cols = min(4, len(items))
    rows = int(math.ceil(len(items) / cols))
    title_h = 56
    gap = 12
    sheet_w = cols * thumb_size + (cols + 1) * gap
    sheet_h = title_h + rows * thumb_size + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 16), title, fill=(0, 0, 0), font=FONT)

    for idx, item in enumerate(items):
        image = Image.open(item).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size), (235, 235, 235))
        x = (thumb_size - image.width) // 2
        y = (thumb_size - image.height) // 2
        canvas.paste(image, (x, y))

        row = idx // cols
        col = idx % cols
        px = gap + col * (thumb_size + gap)
        py = title_h + gap + row * (thumb_size + gap)
        sheet.paste(canvas, (px, py))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def select_slides(topk: pd.DataFrame, mode: str, max_slides: int) -> list[str]:
    first_rows = topk.sort_values(["slide_name", "rank"]).groupby("slide_name", as_index=False).first()
    if mode == "wrong":
        first_rows = first_rows[first_rows["label"].astype(str) != first_rows["pred"].astype(str)]
    elif mode == "positive":
        first_rows = first_rows[first_rows["pred"].astype(str) == "1"]
    elif mode == "negative":
        first_rows = first_rows[first_rows["pred"].astype(str) == "0"]
    elif mode == "high_detector" and "detector_count" in first_rows.columns:
        first_rows["detector_count_num"] = first_rows["detector_count"].map(to_int)
        first_rows = first_rows.sort_values("detector_count_num", ascending=False)
    else:
        first_rows = first_rows.sort_values("slide_name")

    slides = first_rows["slide_name"].astype(str).tolist()
    if max_slides > 0:
        slides = slides[:max_slides]
    return slides


def visualize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = project_path(args.run_dir)
    assert run_dir is not None
    prefix = args.prefix
    explicit_topk_csv = project_path(args.topk_csv)
    evidence_csv = run_dir / f"{prefix}_evidence_rois.csv"
    if explicit_topk_csv is not None:
        topk_csv = explicit_topk_csv
    elif not args.use_raw_topk and evidence_csv.is_file():
        topk_csv = evidence_csv
    else:
        topk_csv = run_dir / f"{prefix}_topk_rois.csv"
    scores_csv = project_path(args.scores_csv) or run_dir / f"{prefix}_detector_roi_scores.csv"
    detections_csv = project_path(args.detections_csv) or run_dir / f"{prefix}_detector_detections.csv"
    out_dir = project_path(args.out_dir) or run_dir / "visualizations" / prefix
    assert out_dir is not None

    topk = read_csv(topk_csv)
    if scores_csv.is_file():
        scores = read_csv(scores_csv)
        score_cols = ["patch_path", "detector_score_max", "detector_score_sum", "detector_count"]
        score_cols = [col for col in score_cols if col in scores.columns and col not in topk.columns]
        if score_cols:
            topk = topk.merge(scores[["patch_path", *score_cols]], on="patch_path", how="left")

    if args.min_detector_count > 0 and "detector_count" in topk.columns:
        topk = topk[topk["detector_count"].map(to_int) >= args.min_detector_count].copy()

    if not args.keep_duplicates and {"slide_name", "patch_path"}.issubset(topk.columns):
        sort_cols = [col for col in ["slide_name", "evidence_rank", "rank"] if col in topk.columns]
        if sort_cols:
            topk = topk.sort_values(sort_cols)
        topk = topk.drop_duplicates(["slide_name", "patch_path"], keep="first").reset_index(drop=True)

    detections = read_csv(detections_csv) if detections_csv.is_file() else pd.DataFrame()
    rank_col = "evidence_rank" if "evidence_rank" in topk.columns else "rank"
    if rank_col in topk.columns:
        topk["rank_num"] = topk[rank_col].map(to_int)
        topk = topk.sort_values(["slide_name", "rank_num"])

    slides = select_slides(topk, args.mode, args.max_slides)
    rows = []
    sheets = []

    for slide_name in slides:
        slide_rows = topk[topk["slide_name"].astype(str) == slide_name].copy()
        if "rank_num" in slide_rows.columns:
            slide_rows = slide_rows.sort_values("rank_num")
        if args.per_slide > 0:
            slide_rows = slide_rows.head(args.per_slide)

        rendered = []
        slide_dir = out_dir / safe_name(slide_name)
        for _, row in slide_rows.iterrows():
            rank = to_int(row.get("rank"))
            patch_name = safe_name(row.get("patch_name", f"rank_{rank:02d}"))
            out_path = slide_dir / f"rank_{rank:02d}_{patch_name}.jpg"
            det_rows = patch_detections(detections, str(row["patch_path"]))
            info = draw_roi(row, det_rows, out_path)
            rows.append(info)
            if info["status"] == "ok":
                rendered.append(out_path)

        sheet_path = out_dir / f"{safe_name(slide_name)}_summary.jpg"
        make_contact_sheet(
            rendered,
            title=f"{slide_name} | Top-{len(rendered)} ROI with detector boxes",
            out_path=sheet_path,
            thumb_size=args.thumb_size,
        )
        if rendered:
            sheets.append(sheet_path)

    index = pd.DataFrame(rows)
    index_path = out_dir / "visualization_index.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False, encoding="utf-8-sig")

    meta = {
        "topk_csv": normalize_path(topk_csv),
        "scores_csv": normalize_path(scores_csv),
        "detections_csv": normalize_path(detections_csv),
        "out_dir": normalize_path(out_dir),
        "slides": len(slides),
        "roi_images": int((index["status"] == "ok").sum()) if not index.empty else 0,
        "summary_images": len(sheets),
        "index": normalize_path(index_path),
    }
    (out_dir / "visualization_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Top-K ROI and detector outputs.")
    parser.add_argument("--run-dir", default="runs/final_cls_det_stage1_m64")
    parser.add_argument("--prefix", default="best_val")
    parser.add_argument("--topk-csv", default="")
    parser.add_argument("--scores-csv", default="")
    parser.add_argument("--detections-csv", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--mode", choices=["all", "wrong", "positive", "negative", "high_detector"], default="all")
    parser.add_argument("--max-slides", type=int, default=20, help="0 means all slides.")
    parser.add_argument("--per-slide", type=int, default=8, help="0 means all Top-K ROI per slide.")
    parser.add_argument("--thumb-size", type=int, default=320)
    parser.add_argument("--min-detector-count", type=int, default=0)
    parser.add_argument("--use-raw-topk", action="store_true", help="Ignore <prefix>_evidence_rois.csv and use raw Top-K.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep repeated 1024 patch paths in visualization.")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(visualize(parse_args()), ensure_ascii=False, indent=2))
