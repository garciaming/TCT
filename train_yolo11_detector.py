import argparse
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


# Default detector pretraining config for the 96G-GPU server.
# Run directly with:
#   python train_yolo11_detector.py
RUN_CONFIG = {
    "data": "database/datasets/detector/yolo_binary/cell_detector.yaml",
    "model": "yolo11m.yaml",
    "project": "runs/yolo11_cell_detector",
    "name": "yolo11m_binary_1024_from_scratch",
    "epochs": 100,
    "imgsz": 1024,
    "batch": 32,
    "device": "0",
    "workers": 16,
    "patience": 25,
    "cache": "False",
    "amp": False,
}


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def normalize_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO11 abnormal-cell detector.")
    parser.add_argument("--data", default=RUN_CONFIG["data"])
    parser.add_argument(
        "--model",
        default=RUN_CONFIG["model"],
        help="YOLO model config (*.yaml) for scratch training, or checkpoint (*.pt) for fine-tuning/resume.",
    )
    parser.add_argument("--project", default=RUN_CONFIG["project"])
    parser.add_argument("--name", default=RUN_CONFIG["name"])
    parser.add_argument("--epochs", type=int, default=RUN_CONFIG["epochs"])
    parser.add_argument("--imgsz", type=int, default=RUN_CONFIG["imgsz"])
    parser.add_argument("--batch", type=int, default=RUN_CONFIG["batch"])
    parser.add_argument("--device", default=RUN_CONFIG["device"])
    parser.add_argument("--workers", type=int, default=RUN_CONFIG["workers"])
    parser.add_argument("--patience", type=int, default=RUN_CONFIG["patience"])
    parser.add_argument("--cache", default=RUN_CONFIG["cache"])
    parser.add_argument(
        "--amp",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=RUN_CONFIG["amp"],
        help="Enable AMP. Default is false to avoid Ultralytics AMP check downloads on offline servers.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_model_arg(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.is_absolute() or model_path.parent != Path("."):
        return str(project_path(model_path))
    return model_arg


def load_yolo_class():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it before detector pretraining, "
            "for example: pip install ultralytics"
        ) from exc
    return YOLO


def make_resolved_data_yaml(data_path: Path, project_dir: Path) -> Path:
    data_root = data_path.parent
    required_dirs = [
        data_root / "images" / "train",
        data_root / "images" / "val",
        data_root / "labels" / "train",
        data_root / "labels" / "val",
    ]
    missing = [path for path in required_dirs if not path.is_dir()]
    if missing:
        missing_text = "\n".join(f"  - {normalize_path(path)}" for path in missing)
        raise FileNotFoundError(f"YOLO dataset directories not found:\n{missing_text}")

    project_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = project_dir / "_resolved_cell_detector.yaml"
    resolved_path.write_text(
        "\n".join(
            [
                f'path: "{normalize_path(data_root)}"',
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: abnormal_cell",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return resolved_path


def main() -> None:
    args = parse_args()
    # Avoid a stale proxy breaking YOLO weight downloads.
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)

    data_path = project_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO data config not found: {normalize_path(data_path)}")

    project_path_arg = project_path(args.project)
    resolved_data_path = make_resolved_data_yaml(data_path, project_path_arg)
    model_arg = resolve_model_arg(args.model)
    YOLO = load_yolo_class()
    model = YOLO(model_arg)
    results = model.train(
        data=str(resolved_data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path_arg),
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
        resume=args.resume,
        exist_ok=True,
    )
    save_dir = Path(getattr(results, "save_dir", project_path_arg / args.name))
    summary = {
        "source_data": normalize_path(data_path),
        "resolved_data": normalize_path(resolved_data_path),
        "model": model_arg,
        "amp": bool(args.amp),
        "save_dir": normalize_path(save_dir),
        "best": normalize_path(save_dir / "weights" / "best.pt"),
        "last": normalize_path(save_dir / "weights" / "last.pt"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
