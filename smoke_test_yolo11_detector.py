import argparse
import json
import os
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny YOLO11 detector training smoke test on the detector dataset."
    )
    parser.add_argument("--data-root", default="database/datasets/detector/yolo_binary")
    parser.add_argument("--out-dir", default="runs/yolo11_cell_detector_smoke/dataset")
    parser.add_argument("--project", default="runs/yolo11_cell_detector_smoke")
    parser.add_argument("--name", default="smoke_yolo11n_320")
    parser.add_argument(
        "--model",
        default="yolo11n.yaml",
        help="YOLO model config (*.yaml) for scratch training, or checkpoint (*.pt) for fine-tuning/resume.",
    )
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--empty-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the tiny train/val lists and YAML; do not start YOLO training.",
    )
    return parser.parse_args()


def resolve_model_arg(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.is_absolute() or model_path.parent != Path("."):
        return normalize_path(project_path(model_path))
    return model_arg


def load_yolo_class():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it first, for example: pip install ultralytics"
        ) from exc
    return YOLO


def validate_label_file(label_path: Path) -> bool:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return False

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no} must have 5 YOLO columns, got {len(parts)}")
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no} has a non-numeric YOLO value") from exc
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_no} expected class id 0, got {class_id}")
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{label_path}:{line_no} has a coordinate outside [0, 1]")
    return True


def collect_split(data_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {normalize_path(image_dir)}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {normalize_path(label_dir)}")

    positive_images: list[Path] = []
    empty_images: list[Path] = []
    missing_labels: list[Path] = []

    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path)
            continue
        if validate_label_file(label_path):
            positive_images.append(image_path)
        else:
            empty_images.append(image_path)

    if missing_labels:
        preview = ", ".join(normalize_path(path) for path in missing_labels[:5])
        raise FileNotFoundError(
            f"{split} has {len(missing_labels)} images without labels. First missing labels: {preview}"
        )
    if not positive_images:
        raise ValueError(f"{split} has no positive labeled images")
    return positive_images, empty_images


def select_images(
    positive_images: list[Path],
    empty_images: list[Path],
    total: int,
    empty_ratio: float,
    rng: random.Random,
) -> list[Path]:
    if total <= 0:
        raise ValueError("Sample count must be positive")
    if empty_ratio < 0.0 or empty_ratio > 1.0:
        raise ValueError("--empty-ratio must be in [0, 1]")

    empty_count = min(len(empty_images), round(total * empty_ratio))
    positive_count = min(len(positive_images), total - empty_count)
    empty_count = min(len(empty_images), total - positive_count)

    selected = rng.sample(positive_images, positive_count)
    if empty_count:
        selected.extend(rng.sample(empty_images, empty_count))
    rng.shuffle(selected)

    if not selected:
        raise ValueError("No images selected for the smoke test")
    return selected


def write_list(path: Path, images: list[Path]) -> None:
    path.write_text(
        "\n".join(normalize_path(image_path) for image_path in images) + "\n",
        encoding="utf-8",
    )


def prepare_smoke_dataset(args: argparse.Namespace) -> dict[str, str | int]:
    rng = random.Random(args.seed)
    data_root = project_path(args.data_root)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_positive, train_empty = collect_split(data_root, "train")
    val_positive, val_empty = collect_split(data_root, "val")

    train_images = select_images(
        train_positive, train_empty, args.train_samples, args.empty_ratio, rng
    )
    val_images = select_images(val_positive, val_empty, args.val_samples, args.empty_ratio, rng)

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    data_yaml = out_dir / "cell_detector_smoke.yaml"

    write_list(train_txt, train_images)
    write_list(val_txt, val_images)
    data_yaml.write_text(
        "\n".join(
            [
                f"train: {normalize_path(train_txt)}",
                f"val: {normalize_path(val_txt)}",
                "names:",
                "  0: abnormal_cell",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "data_root": normalize_path(data_root),
        "out_dir": normalize_path(out_dir),
        "data_yaml": normalize_path(data_yaml),
        "train_txt": normalize_path(train_txt),
        "val_txt": normalize_path(val_txt),
        "train_samples": len(train_images),
        "val_samples": len(val_images),
        "train_positive_pool": len(train_positive),
        "train_empty_pool": len(train_empty),
        "val_positive_pool": len(val_positive),
        "val_empty_pool": len(val_empty),
    }


def main() -> None:
    args = parse_args()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)

    summary = prepare_smoke_dataset(args)
    if args.prepare_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    YOLO = load_yolo_class()
    model = YOLO(resolve_model_arg(args.model))
    project = project_path(args.project)
    results = model.train(
        data=summary["data_yaml"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        cache=args.cache,
        exist_ok=True,
        plots=False,
    )

    save_dir = Path(getattr(results, "save_dir", project / args.name))
    summary.update(
        {
            "model": resolve_model_arg(args.model),
            "save_dir": normalize_path(save_dir),
            "best": normalize_path(save_dir / "weights" / "best.pt"),
            "last": normalize_path(save_dir / "weights" / "last.pt"),
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
