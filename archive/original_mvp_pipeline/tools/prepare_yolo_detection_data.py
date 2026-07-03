import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["ASC-US", "LSIL", "ASC-H", "HSIL", "SCC"]
BINARY_CLASS_NAMES = ["abnormal_cell"]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def yolo_label_lines(boxes_json: str, width: int, height: int, binary: bool) -> list[str]:
    lines = []
    for box in json.loads(boxes_json):
        class_id = 0 if binary else int(box["label_id"]) - 1
        xmin = max(0.0, min(float(box["xmin"]), width - 1.0))
        ymin = max(0.0, min(float(box["ymin"]), height - 1.0))
        xmax = max(0.0, min(float(box["xmax"]), width * 1.0))
        ymax = max(0.0, min(float(box["ymax"]), height * 1.0))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return lines


def write_tile(row: pd.Series, image_out: Path, label_out: Path, binary: bool) -> int:
    tile_width = int(float(row["tile_width"]))
    tile_height = int(float(row["tile_height"]))
    tile_x = int(float(row["tile_x"]))
    tile_y = int(float(row["tile_y"]))
    source_path = Path(row["image_path"])

    if tile_x == 0 and tile_y == 0:
        try:
            with Image.open(source_path) as image:
                if image.size == (tile_width, tile_height):
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, image_out)
                else:
                    crop = image.convert("RGB").crop((tile_x, tile_y, tile_x + tile_width, tile_y + tile_height))
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(image_out, quality=95)
        except Exception:
            with Image.open(source_path) as image:
                crop = image.convert("RGB").crop((tile_x, tile_y, tile_x + tile_width, tile_y + tile_height))
                image_out.parent.mkdir(parents=True, exist_ok=True)
                crop.save(image_out, quality=95)
    else:
        with Image.open(source_path) as image:
            crop = image.convert("RGB").crop((tile_x, tile_y, tile_x + tile_width, tile_y + tile_height))
            image_out.parent.mkdir(parents=True, exist_ok=True)
            crop.save(image_out, quality=95)

    lines = yolo_label_lines(row["boxes_json"], tile_width, tile_height, binary)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 1024 cell detection data to YOLO format.")
    parser.add_argument("--manifest", default="data/datasets/cell_detector/detection_tiles.csv")
    parser.add_argument("--out-dir", default="data/datasets/cell_detector_yolo_binary")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--negative-fraction", type=float, default=0.2)
    class_mode = parser.add_mutually_exclusive_group()
    class_mode.add_argument("--binary", dest="binary", action="store_true", help="Merge all abnormal cell labels into one class.")
    class_mode.add_argument("--multi-class", dest="binary", action="store_false", help="Keep ASC-US/LSIL/ASC-H/HSIL/SCC as separate classes.")
    parser.set_defaults(binary=True)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_rows(data: pd.DataFrame, split: str, max_rows: int, positive_only: bool, negative_fraction: float, seed: int) -> pd.DataFrame:
    subset = data[data["split"] == split].copy()
    subset["num_boxes"] = subset["num_boxes"].astype(int)
    positives = subset[subset["num_boxes"] > 0]
    negatives = subset[subset["num_boxes"] == 0]
    if positive_only:
        selected = positives
    else:
        n_neg = int(round(len(positives) * max(0.0, negative_fraction)))
        n_neg = min(n_neg, len(negatives))
        negative_sample = negatives.sample(n=n_neg, random_state=seed) if n_neg else negatives.head(0)
        selected = pd.concat([positives, negative_sample], ignore_index=True)
    if max_rows > 0 and len(selected) > max_rows:
        selected = selected.sample(n=max_rows, random_state=seed)
    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(project_path(args.manifest), dtype=str, keep_default_na=False)
    class_names = BINARY_CLASS_NAMES if args.binary else CLASS_NAMES
    splits = {
        "train": select_rows(data, "train", args.max_train, args.positive_only, args.negative_fraction, args.seed),
        "val": select_rows(data, "val", args.max_val, False, args.negative_fraction, args.seed),
    }
    summary = {
        "out_dir": normalize_path(out_dir),
        "classes": class_names,
        "binary": bool(args.binary),
        "splits": {},
    }

    for split, rows in splits.items():
        written = 0
        boxes = 0
        image_list = []
        for _, row in rows.iterrows():
            stem = safe_name(f"{row['source']}_{row['tile_id']}")
            image_out = out_dir / "images" / split / f"{stem}.jpg"
            label_out = out_dir / "labels" / split / f"{stem}.txt"
            n_boxes = write_tile(row, image_out, label_out, args.binary)
            written += 1
            boxes += n_boxes
            image_list.append(normalize_path(image_out))
        list_path = out_dir / f"{split}.txt"
        list_path.write_text("\n".join(image_list) + "\n", encoding="utf-8")
        summary["splits"][split] = {
            "images": written,
            "boxes": boxes,
            "list": normalize_path(list_path),
        }

    yaml_text = "\n".join(
        [
            f"path: {normalize_path(out_dir)}",
            "train: train.txt",
            "val: val.txt",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(class_names)],
            "",
        ]
    )
    data_yaml = out_dir / "cell_detector.yaml"
    data_yaml.write_text(yaml_text, encoding="utf-8")
    summary["data_yaml"] = normalize_path(data_yaml)
    summary_path = out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
