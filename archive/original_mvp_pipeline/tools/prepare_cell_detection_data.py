import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def stable_split(key: str, val_fraction: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_fraction else "train"


def normalize_label(label: str) -> str | None:
    label = label.strip().upper()
    if "ASC-US" in label:
        return "ASC-US"
    if "ASC-H" in label:
        return "ASC-H"
    if "LSIL" in label:
        return "LSIL"
    if "HSIL" in label:
        return "HSIL"
    if "SCC" in label:
        return "SCC"
    return None


def parse_voc_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or f"{xml_path.stem}.jpg").strip()
    width = int(float(root.findtext("size/width") or 0))
    height = int(float(root.findtext("size/height") or 0))
    boxes = []
    skipped_labels = Counter()
    for obj in root.findall("object"):
        raw_label = obj.findtext("name") or ""
        label = normalize_label(raw_label)
        if label is None:
            skipped_labels[raw_label.strip()] += 1
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": label,
                "label_id": CLASS_TO_ID[label],
            }
        )
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "boxes": boxes,
        "skipped_labels": skipped_labels,
    }


def build_image_index(dirs: list[Path]) -> dict[str, Path]:
    index = {}
    for image_dir in dirs:
        if not image_dir.is_dir():
            continue
        for path in image_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
    return index


def find_image(filename: str, image_index: dict[str, Path]) -> Path | None:
    direct = image_index.get(filename.lower())
    if direct is not None:
        return direct
    return image_index.get(Path(filename).stem.lower())


def tile_origins(width: int, height: int, tile_size: int) -> list[tuple[int, int]]:
    if width < tile_size or height < tile_size:
        return []
    xs = list(range(0, width - tile_size + 1, tile_size))
    ys = list(range(0, height - tile_size + 1, tile_size))
    if xs[-1] != width - tile_size:
        xs.append(width - tile_size)
    if ys[-1] != height - tile_size:
        ys.append(height - tile_size)
    return [(x, y) for y in ys for x in xs]


def boxes_for_tile(
    boxes: list[dict],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_box_area: float,
) -> list[dict]:
    tile_boxes = []
    x2 = tile_x + tile_size
    y2 = tile_y + tile_size
    for box in boxes:
        ix1 = max(box["xmin"], tile_x)
        iy1 = max(box["ymin"], tile_y)
        ix2 = min(box["xmax"], x2)
        iy2 = min(box["ymax"], y2)
        area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if area < min_box_area:
            continue
        tile_boxes.append(
            {
                "xmin": round(ix1 - tile_x, 2),
                "ymin": round(iy1 - tile_y, 2),
                "xmax": round(ix2 - tile_x, 2),
                "ymax": round(iy2 - tile_y, 2),
                "label": box["label"],
                "label_id": box["label_id"],
            }
        )
    return tile_boxes


def collect_voc_source(
    source_name: str,
    annotation_dir: Path,
    image_dirs: list[Path],
    tile_size: int,
    min_box_area: float,
    include_empty_tiles: bool,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], dict]:
    image_index = build_image_index(image_dirs)
    rows = []
    summary = {
        "source": source_name,
        "xml_files": 0,
        "missing_images": 0,
        "skipped_small_images": 0,
        "skipped_parse_errors": 0,
        "tiles": 0,
        "positive_tiles": 0,
        "objects": 0,
        "labels": Counter(),
        "skipped_labels": Counter(),
    }

    for xml_path in sorted(annotation_dir.glob("*.xml")):
        summary["xml_files"] += 1
        try:
            parsed = parse_voc_xml(xml_path)
        except Exception:
            summary["skipped_parse_errors"] += 1
            continue

        summary["skipped_labels"].update(parsed["skipped_labels"])
        image_path = find_image(parsed["filename"], image_index)
        if image_path is None:
            summary["missing_images"] += 1
            continue

        origins = tile_origins(parsed["width"], parsed["height"], tile_size)
        if not origins:
            summary["skipped_small_images"] += 1
            continue

        split = stable_split(f"{source_name}:{xml_path.stem}", val_fraction, seed)
        for tile_x, tile_y in origins:
            tile_boxes = boxes_for_tile(parsed["boxes"], tile_x, tile_y, tile_size, min_box_area)
            if not tile_boxes and not include_empty_tiles:
                continue
            labels = Counter(box["label"] for box in tile_boxes)
            summary["tiles"] += 1
            summary["positive_tiles"] += int(bool(tile_boxes))
            summary["objects"] += len(tile_boxes)
            summary["labels"].update(labels)
            rows.append(
                {
                    "source": source_name,
                    "split": split,
                    "image_path": normalize_path(image_path),
                    "xml_path": normalize_path(xml_path),
                    "image_stem": image_path.stem,
                    "tile_id": f"{image_path.stem}_x{tile_x}_y{tile_y}",
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_width": tile_size,
                    "tile_height": tile_size,
                    "source_width": parsed["width"],
                    "source_height": parsed["height"],
                    "num_boxes": len(tile_boxes),
                    "labels": ",".join(sorted(labels)),
                    "boxes_json": json.dumps(tile_boxes, ensure_ascii=False),
                }
            )
    return rows, summary


def count_tct1_unlabeled(tct1_dir: Path) -> dict:
    if not tct1_dir.is_dir():
        return {"source": "TCT1", "images": 0, "note": "directory_missing"}
    images = [
        path
        for path in tct1_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    return {
        "source": "TCT1",
        "images": len(images),
        "note": "no XML/box annotations found; excluded from supervised detector training",
    }


def counter_to_dict(value):
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: counter_to_dict(item) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1024x1024 abnormal-cell detection manifest.")
    parser.add_argument("--tct3055-root", default="data/datasets/TCT-3055")
    parser.add_argument("--tianchi-root", default="data/datasets/TCTdata_tianchi")
    parser.add_argument("--tct1-root", default="data/datasets/TCT1")
    parser.add_argument("--out-dir", default="data/datasets/cell_detector")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--min-box-area", type=float, default=16.0)
    parser.add_argument("--no-empty-tiles", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_empty_tiles = not args.no_empty_tiles
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tct3055_root = project_path(args.tct3055_root)
    tianchi_root = project_path(args.tianchi_root)
    rows = []
    summaries = []

    source_rows, source_summary = collect_voc_source(
        source_name="TCT-3055",
        annotation_dir=tct3055_root / "Annotations_NINE",
        image_dirs=[tct3055_root / "JPEGImages"],
        tile_size=args.tile_size,
        min_box_area=args.min_box_area,
        include_empty_tiles=include_empty_tiles,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    rows.extend(source_rows)
    summaries.append(source_summary)

    source_rows, source_summary = collect_voc_source(
        source_name="TCTdata_tianchi",
        annotation_dir=tianchi_root / "Annotations",
        image_dirs=[tianchi_root / "JPEGImages", tianchi_root / "images"],
        tile_size=args.tile_size,
        min_box_area=args.min_box_area,
        include_empty_tiles=include_empty_tiles,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    rows.extend(source_rows)
    summaries.append(source_summary)

    tct1_summary = count_tct1_unlabeled(project_path(args.tct1_root))
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No supervised detection tiles were built.")
    manifest_path = out_dir / "detection_tiles.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    summary = {
        "manifest": normalize_path(manifest_path),
        "classes": CLASS_TO_ID,
        "tile_size": args.tile_size,
        "include_empty_tiles": include_empty_tiles,
        "rows": int(len(manifest)),
        "positive_rows": int((manifest["num_boxes"].astype(int) > 0).sum()),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "source_counts": manifest["source"].value_counts().to_dict(),
        "box_counts_by_class": Counter(
            box["label"]
            for boxes_json in manifest["boxes_json"]
            for box in json.loads(boxes_json)
        ),
        "sources": summaries,
        "unlabeled_sources": [tct1_summary],
    }
    summary = counter_to_dict(summary)
    summary_path = out_dir / "detection_tiles_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
