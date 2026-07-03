import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABNORMAL_CATEGORY_IDS = {1, 2, 3, 4, 5, 6}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def parse_category_ids(value: str) -> set[int]:
    ids = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not ids:
        raise ValueError("--positive-category-ids cannot be empty")
    return ids


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_image_index(coco: dict) -> dict[str, dict]:
    return {str(image["id"]): image for image in coco["images"]}


def build_annotation_index(coco: dict) -> dict[str, list[dict]]:
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[str(annotation["image_id"])].append(annotation)
    return by_image


def yolo_line(annotation: dict, image: dict, positive_category_ids: set[int]) -> tuple[str | None, str]:
    category_id = int(annotation["category_id"])
    if category_id not in positive_category_ids:
        return None, "ignored_category"

    x, y, width, height = [float(value) for value in annotation["bbox"]]
    if width <= 1 or height <= 1:
        return None, "bad_size"

    image_width = float(image["width"])
    image_height = float(image["height"])
    x1 = max(0.0, min(x, image_width - 1.0))
    y1 = max(0.0, min(y, image_height - 1.0))
    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 1 or clipped_height <= 1:
        return None, "bad_clipped_size"

    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    bw = clipped_width / image_width
    bh = clipped_height / image_height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}", "kept"


def extract_images(zip_path: Path, split: str, out_dir: Path, overwrite: bool) -> dict:
    images_dir = out_dir / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        for idx, member in enumerate(members, start=1):
            member_path = Path(member.filename)
            if member_path.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            dest = images_dir / member_path.name
            if dest.exists() and not overwrite and dest.stat().st_size == member.file_size:
                skipped += 1
                continue
            with archive.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            copied += 1
            if idx % 500 == 0:
                print(f"{split}: extracted {idx}/{len(members)} files")
    return {"copied": copied, "skipped": skipped, "images_dir": normalize_path(images_dir)}


def write_labels(coco: dict, split: str, out_dir: Path, positive_category_ids: set[int]) -> dict:
    labels_dir = out_dir / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images" / split
    image_index = build_image_index(coco)
    annotations_by_image = build_annotation_index(coco)
    counters = Counter()
    class_counter = Counter()
    image_list = []
    positive_images = 0
    empty_images = 0

    for image_id, image in image_index.items():
        file_name = str(image["file_name"])
        image_path = images_dir / Path(file_name).name
        label_path = labels_dir / f"{Path(file_name).stem}.txt"
        lines = []
        for annotation in annotations_by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            if category_id in positive_category_ids:
                class_counter[category_id] += 1
            line, status = yolo_line(annotation, image, positive_category_ids)
            counters[status] += 1
            if line:
                lines.append(line)

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_list.append(normalize_path(image_path))
        if lines:
            positive_images += 1
        else:
            empty_images += 1

    list_path = out_dir / f"{split}.txt"
    list_path.write_text("\n".join(image_list) + "\n", encoding="utf-8")
    return {
        "images": len(image_index),
        "positive_images": positive_images,
        "empty_images": empty_images,
        "labels_dir": normalize_path(labels_dir),
        "list": normalize_path(list_path),
        "annotations": dict(counters),
        "positive_category_annotations": {str(key): value for key, value in sorted(class_counter.items())},
    }


def write_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "cell_detector.yaml"
    yaml_text = "\n".join(
        [
            "path: data/datasets/detector/yolo_binary",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: abnormal_cell",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def prepare(args: argparse.Namespace) -> dict:
    source_dir = project_path(args.source_dir)
    out_dir = project_path(args.out_dir)
    positive_category_ids = parse_category_ids(args.positive_category_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_coco = load_coco(source_dir / "train.json")
    val_coco = load_coco(source_dir / "test.json")
    categories = {int(category["id"]): category["name"] for category in train_coco["categories"]}

    summary = {
        "source_dir": normalize_path(source_dir),
        "out_dir": normalize_path(out_dir),
        "mode": "binary_abnormal_cell",
        "positive_category_ids": sorted(positive_category_ids),
        "positive_categories": {
            str(category_id): categories.get(category_id, "")
            for category_id in sorted(positive_category_ids)
        },
        "ignored_categories": {
            str(category_id): name
            for category_id, name in sorted(categories.items())
            if category_id not in positive_category_ids
        },
        "splits": {},
    }

    if not args.no_extract:
        summary["splits"]["train_extract"] = extract_images(
            source_dir / "train.zip",
            "train",
            out_dir,
            overwrite=args.overwrite_images,
        )
        summary["splits"]["val_extract"] = extract_images(
            source_dir / "test.zip",
            "val",
            out_dir,
            overwrite=args.overwrite_images,
        )

    summary["splits"]["train"] = write_labels(train_coco, "train", out_dir, positive_category_ids)
    summary["splits"]["val"] = write_labels(val_coco, "val", out_dir, positive_category_ids)
    yaml_path = write_yaml(out_dir)
    summary["data_yaml"] = normalize_path(yaml_path)
    summary_path = out_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ComparisonDetectorDataset COCO data as binary YOLO labels.")
    parser.add_argument(
        "--source-dir",
        default="data/datasets/detector/raw/ComparisonDetectorDataset",
        help="Directory containing train.json, test.json, train.zip and test.zip.",
    )
    parser.add_argument("--out-dir", default="data/datasets/detector/yolo_binary")
    parser.add_argument(
        "--positive-category-ids",
        default="1,2,3,4,5,6",
        help="COCO category ids merged into class 0 abnormal_cell.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Only rewrite labels/yaml, do not extract images.")
    parser.add_argument("--overwrite-images", action="store_true", help="Rewrite extracted image files.")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
