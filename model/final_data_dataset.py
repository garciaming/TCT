from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_LABEL_MAP = {"negative": 0, "positive": 1}


def _normalize_key(value: object) -> str:
    return str(value).strip().lower()


def make_patch_transform(encoder_input_size: int = 224, resize: bool = True) -> transforms.Compose:
    steps = []
    if resize:
        steps.append(transforms.Resize((encoder_input_size, encoder_input_size)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def _uniform_indices(num_items: int, max_items: int) -> torch.Tensor:
    if max_items <= 0 or num_items <= max_items:
        return torch.arange(num_items)
    return torch.linspace(0, num_items - 1, max_items).round().long()


def _split_bag_keys(
    bag_table: pd.DataFrame,
    split: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> set[str]:
    if split in ("all", "", None):
        return set(bag_table["bag_key"].astype(str))

    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test/all, got {split!r}")

    generator = torch.Generator().manual_seed(seed)
    selected: set[str] = set()

    for _, class_rows in bag_table.groupby("label"):
        keys = class_rows["bag_key"].astype(str).tolist()
        if not keys:
            continue

        perm = torch.randperm(len(keys), generator=generator).tolist()
        keys = [keys[i] for i in perm]
        n_total = len(keys)
        n_test = int(round(n_total * test_fraction))
        n_val = int(round(n_total * val_fraction))

        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test : n_test + n_val])
        train_keys = set(keys[n_test + n_val :])

        if split == "train":
            selected.update(train_keys)
        elif split == "val":
            selected.update(val_keys)
        else:
            selected.update(test_keys)

    return selected


class FinalDataPatchBagDataset(Dataset):
    """
    Load pre-cut 1024 patch images from database/datasets/final_data as slide-level MIL bags.

    Expected folder layout:

        1024_patch/
          positive/<slide_name>/<patch_name>.jpg
          negative/<slide_name>/<patch_name>.jpg
          patch_1024_metadata.csv

    Each item returns a bag of transformed patch images with shape [N, 3, H, W].
    YOLO txt labels from json_to_1024_output/yolo_labels are optional and are returned
    only as auxiliary targets; they are not required when skipping detector pretraining.
    """

    def __init__(
        self,
        patch_root: str | Path = "database/datasets/final_data/1024_patch",
        metadata_csv: str | Path | None = None,
        yolo_label_dir: str | Path | None = "database/datasets/final_data/json_to_1024_output/yolo_labels",
        split: str = "train",
        split_csv: str | Path | None = None,
        max_patches: int = 512,
        encoder_input_size: int = 224,
        patch_input_mode: str = "resize",
        subpatch_grid_size: int = 4,
        subpatch_tile_size: int = 256,
        center_crop_size: int | None = None,
        expected_patch_size: int = 1024,
        label_map: dict[str, int] | None = None,
        seed: int = 9,
        val_fraction: float = 0.2,
        test_fraction: float = 0.0,
        sample_mode: str = "random",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.patch_root = Path(patch_root)
        self.metadata_csv = Path(metadata_csv) if metadata_csv else self.patch_root / "patch_1024_metadata.csv"
        self.yolo_label_dir = Path(yolo_label_dir) if yolo_label_dir else None
        self.split = split
        self.max_patches = max_patches
        self.seed = seed
        self.sample_mode = sample_mode
        self.encoder_input_size = encoder_input_size
        self.patch_input_mode = patch_input_mode
        self.subpatch_grid_size = subpatch_grid_size
        self.subpatch_tile_size = subpatch_tile_size
        self.center_crop_size = center_crop_size or encoder_input_size
        self.expected_patch_size = expected_patch_size
        self.transform = transform or make_patch_transform(
            encoder_input_size,
            resize=patch_input_mode == "resize",
        )
        self.label_map = {k.lower(): int(v) for k, v in (label_map or DEFAULT_LABEL_MAP).items()}

        if sample_mode not in {"random", "uniform"}:
            raise ValueError(f"sample_mode must be random or uniform, got {sample_mode!r}")
        if patch_input_mode not in {"resize", "subpatch_4x4"}:
            raise ValueError(f"patch_input_mode must be resize or subpatch_4x4, got {patch_input_mode!r}")
        if self.subpatch_tile_size * self.subpatch_grid_size != self.expected_patch_size:
            raise ValueError("subpatch_tile_size * subpatch_grid_size must equal expected_patch_size")
        if self.center_crop_size > self.subpatch_tile_size:
            raise ValueError("center_crop_size must be <= subpatch_tile_size")
        if not self.patch_root.is_dir():
            raise FileNotFoundError(f"patch_root does not exist: {self.patch_root}")

        self.patch_table = self._build_patch_table()
        self.patch_table = self._apply_split(
            self.patch_table,
            split_csv=Path(split_csv) if split_csv else None,
            split=split,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        if self.patch_table.empty:
            raise ValueError(f"No patches found for split={split!r} under {self.patch_root}")

        sort_cols = [c for c in ["bag_key", "row_1024", "col_1024", "patch_name"] if c in self.patch_table.columns]
        self.patch_table = self.patch_table.sort_values(sort_cols).reset_index(drop=True)
        self.num_patch_rows = len(self.patch_table)
        self.bags = [
            {"bag_key": bag_key, "rows": rows.reset_index(drop=True)}
            for bag_key, rows in self.patch_table.groupby("bag_key", sort=False)
        ]
        # Keep per-bag rows only; the full table duplicates many strings and can be large.
        self.patch_table = pd.DataFrame()

    def _scan_image_files(self) -> pd.DataFrame:
        rows = []
        for label_dir in sorted(p for p in self.patch_root.iterdir() if p.is_dir()):
            label_name = label_dir.name.lower()
            if label_name not in self.label_map:
                continue
            for image_path in label_dir.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                slide_name = image_path.parent.name
                patch_name = image_path.name
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label_name": label_name,
                        "label": self.label_map[label_name],
                        "slide_name": slide_name,
                        "patch_name": patch_name,
                        "join_key": f"{label_name}|{_normalize_key(slide_name)}|{_normalize_key(patch_name)}",
                    }
                )

        if not rows:
            raise ValueError(f"No patch images found under {self.patch_root}")
        return pd.DataFrame(rows)

    def _read_metadata(self) -> pd.DataFrame:
        if not self.metadata_csv.is_file():
            return pd.DataFrame()

        usecols = [
            "patch_name",
            "slide_name",
            "slide_file",
            "label",
            "row_1024",
            "col_1024",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "patch_size",
            "level",
            "cell_pixel_ratio",
            "save_path",
        ]
        metadata = pd.read_csv(
            self.metadata_csv,
            usecols=lambda col: col in usecols,
            dtype={
                "patch_name": "string",
                "slide_name": "string",
                "slide_file": "string",
                "label": "string",
                "save_path": "string",
            },
        )
        if metadata.empty:
            return metadata

        metadata["label_name"] = metadata["label"].map(lambda x: _normalize_key(x))
        metadata["join_key"] = metadata.apply(
            lambda row: (
                f"{row['label_name']}|"
                f"{_normalize_key(row['slide_name'])}|"
                f"{_normalize_key(row['patch_name'])}"
            ),
            axis=1,
        )
        return metadata

    def _build_patch_table_from_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        required = {"patch_name", "slide_name", "label_name"}
        if metadata.empty or not required.issubset(metadata.columns):
            return pd.DataFrame()

        table = metadata.copy()
        table = table[table["label_name"].isin(self.label_map)].copy()
        if table.empty:
            return table

        table["label"] = table["label_name"].map(self.label_map).astype(int)
        table["patch_name"] = table["patch_name"].astype(str)
        table["slide_name"] = table["slide_name"].astype(str)
        table["image_path"] = [
            str(self.patch_root / label_name / slide_name / patch_name)
            for label_name, slide_name, patch_name in zip(
                table["label_name"].astype(str),
                table["slide_name"].astype(str),
                table["patch_name"].astype(str),
            )
        ]
        table["bag_key"] = table["label_name"].astype(str) + "/" + table["slide_name"].astype(str)
        keep_cols = [
            "image_path",
            "label_name",
            "label",
            "slide_name",
            "patch_name",
            "bag_key",
            "row_1024",
            "col_1024",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]
        table = table[[col for col in keep_cols if col in table.columns]]
        return table

    def _build_patch_table(self) -> pd.DataFrame:
        metadata = self._read_metadata()
        table = self._build_patch_table_from_metadata(metadata)
        if not table.empty:
            return table

        images = self._scan_image_files()
        if metadata.empty:
            table = images
        else:
            metadata = metadata.drop_duplicates("join_key", keep="first")
            metadata = metadata.drop(columns=["label"], errors="ignore")
            table = images.merge(
                metadata,
                how="left",
                on="join_key",
                suffixes=("", "_meta"),
            )
            for col in ["slide_name", "patch_name", "label_name"]:
                meta_col = f"{col}_meta"
                if meta_col in table.columns:
                    table[col] = table[meta_col].fillna(table[col])
                    table = table.drop(columns=[meta_col])

        table["bag_key"] = table["label_name"].astype(str) + "/" + table["slide_name"].astype(str)
        return table

    def _apply_split(
        self,
        table: pd.DataFrame,
        split_csv: Path | None,
        split: str,
        seed: int,
        val_fraction: float,
        test_fraction: float,
    ) -> pd.DataFrame:
        if split in ("all", "", None):
            return table

        bag_table = table[["bag_key", "slide_name", "label"]].drop_duplicates().reset_index(drop=True)

        if split_csv is not None:
            split_table = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
            if "split" not in split_table.columns:
                raise ValueError(f"split_csv must contain a split column: {split_csv}")
            key_col = "bag_key" if "bag_key" in split_table.columns else "slide_name"
            if key_col not in split_table.columns:
                raise ValueError(f"split_csv must contain bag_key or slide_name: {split_csv}")
            wanted = set(
                split_table.loc[split_table["split"].str.lower() == split, key_col].map(str)
            )
            if key_col == "slide_name":
                return table[table["slide_name"].astype(str).isin(wanted)].reset_index(drop=True)
            return table[table["bag_key"].astype(str).isin(wanted)].reset_index(drop=True)

        selected = _split_bag_keys(bag_table, split, seed, val_fraction, test_fraction)
        return table[table["bag_key"].astype(str).isin(selected)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.bags)

    def _select_rows(self, rows: pd.DataFrame, index: int) -> pd.DataFrame:
        n = len(rows)
        if self.max_patches <= 0 or n <= self.max_patches:
            return rows
        if self.sample_mode == "random" and self.split == "train":
            indices = torch.randperm(n)[: self.max_patches]
            indices = indices.sort().values
        else:
            indices = _uniform_indices(n, self.max_patches)
        return rows.iloc[indices.tolist()].reset_index(drop=True)

    def _base_coord(self, row: pd.Series) -> dict:
        return {
            "row_1024": None if pd.isna(row.get("row_1024")) else int(row.get("row_1024")),
            "col_1024": None if pd.isna(row.get("col_1024")) else int(row.get("col_1024")),
            "x_min": None if pd.isna(row.get("x_min")) else int(row.get("x_min")),
            "y_min": None if pd.isna(row.get("y_min")) else int(row.get("y_min")),
            "x_max": None if pd.isna(row.get("x_max")) else int(row.get("x_max")),
            "y_max": None if pd.isna(row.get("y_max")) else int(row.get("y_max")),
        }

    def _subpatch_hit_count(
        self,
        boxes: torch.Tensor,
        crop_x: int,
        crop_y: int,
        crop_size: int,
    ) -> int:
        if boxes.numel() == 0:
            return 0
        centers_x = boxes[:, 1] * float(self.expected_patch_size)
        centers_y = boxes[:, 2] * float(self.expected_patch_size)
        inside = (
            (centers_x >= crop_x)
            & (centers_x < crop_x + crop_size)
            & (centers_y >= crop_y)
            & (centers_y < crop_y + crop_size)
        )
        return int(inside.sum().item())

    def _read_patch_instances(
        self,
        row: pd.Series,
        source_patch_index: int,
        boxes: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[str], list[str], list[dict], list[int]]:
        image_path = str(row["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.patch_input_mode == "resize":
                return (
                    [self.transform(image)],
                    [str(row["patch_name"])],
                    [image_path],
                    [self._base_coord(row) | {"source_patch_index": source_patch_index, "patch_input_mode": "resize"}],
                    [len(boxes)],
                )

            if image.size != (self.expected_patch_size, self.expected_patch_size):
                raise ValueError(
                    f"Expected {self.expected_patch_size}x{self.expected_patch_size} patch, "
                    f"got {image.size} for {image_path}"
                )

            tensors: list[torch.Tensor] = []
            patch_names: list[str] = []
            patch_paths: list[str] = []
            coords: list[dict] = []
            box_counts: list[int] = []
            margin = (self.subpatch_tile_size - self.center_crop_size) // 2
            base = self._base_coord(row)
            patch_x = base["x_min"]
            patch_y = base["y_min"]
            original_patch = Path(str(row["patch_name"]))
            patch_stem = original_patch.stem
            patch_suffix = original_patch.suffix or ".jpg"

            for tile_row in range(self.subpatch_grid_size):
                for tile_col in range(self.subpatch_grid_size):
                    tile_x = tile_col * self.subpatch_tile_size
                    tile_y = tile_row * self.subpatch_tile_size
                    crop_x = tile_x + margin
                    crop_y = tile_y + margin
                    crop_box = (
                        crop_x,
                        crop_y,
                        crop_x + self.center_crop_size,
                        crop_y + self.center_crop_size,
                    )
                    crop = image.crop(crop_box)
                    tensors.append(self.transform(crop))
                    patch_names.append(f"{patch_stem}__r{tile_row}c{tile_col}{patch_suffix}")
                    patch_paths.append(image_path)

                    global_crop_x = None if patch_x is None else patch_x + crop_x
                    global_crop_y = None if patch_y is None else patch_y + crop_y
                    coords.append(
                        base
                        | {
                            "source_patch_index": source_patch_index,
                            "patch_input_mode": "subpatch_4x4",
                            "original_patch_name": str(row["patch_name"]),
                            "tile_row": tile_row,
                            "tile_col": tile_col,
                            "tile_x_in_patch": tile_x,
                            "tile_y_in_patch": tile_y,
                            "crop_x_in_patch": crop_x,
                            "crop_y_in_patch": crop_y,
                            "crop_size": self.center_crop_size,
                            "global_crop_x_min": global_crop_x,
                            "global_crop_y_min": global_crop_y,
                            "global_crop_x_max": None
                            if global_crop_x is None
                            else global_crop_x + self.center_crop_size,
                            "global_crop_y_max": None
                            if global_crop_y is None
                            else global_crop_y + self.center_crop_size,
                        }
                    )
                    box_counts.append(
                        self._subpatch_hit_count(
                            boxes,
                            crop_x=crop_x,
                            crop_y=crop_y,
                            crop_size=self.center_crop_size,
                        )
                    )

            return tensors, patch_names, patch_paths, coords, box_counts

    def _read_yolo_boxes(self, patch_name: str) -> torch.Tensor:
        if self.yolo_label_dir is None:
            return torch.empty((0, 5), dtype=torch.float32)
        label_path = self.yolo_label_dir / f"{Path(patch_name).stem}.txt"
        if not label_path.is_file():
            return torch.empty((0, 5), dtype=torch.float32)

        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            boxes.append([float(x) for x in parts])
        if not boxes:
            return torch.empty((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        bag = self.bags[index]
        rows = self._select_rows(bag["rows"], index)

        image_items: list[torch.Tensor] = []
        patch_names: list[str] = []
        patch_paths: list[str] = []
        coords: list[dict] = []
        boxes: list[torch.Tensor] = []
        box_count_items: list[int] = []

        for source_patch_index, (_, row) in enumerate(rows.iterrows()):
            patch_boxes = self._read_yolo_boxes(str(row["patch_name"]))
            tensors, names, paths, item_coords, item_box_counts = self._read_patch_instances(
                row=row,
                source_patch_index=source_patch_index,
                boxes=patch_boxes,
            )
            image_items.extend(tensors)
            patch_names.extend(names)
            patch_paths.extend(paths)
            coords.extend(item_coords)
            boxes.extend([patch_boxes] * len(tensors))
            box_count_items.extend(item_box_counts)

        images = torch.stack(image_items)
        box_counts = torch.tensor(box_count_items, dtype=torch.long)

        first = rows.iloc[0]
        return {
            "images": images,
            "label": int(first["label"]),
            "bag_key": str(first["bag_key"]),
            "slide_name": str(first["slide_name"]),
            "label_name": str(first["label_name"]),
            "patch_names": patch_names,
            "patch_paths": patch_paths,
            "coords": coords,
            "boxes": boxes,
            "box_counts": box_counts,
            "patch_has_cell": box_counts > 0,
        }


def collate_patch_bags(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_len = max(item["images"].shape[0] for item in batch)
    channels, height, width = batch[0]["images"].shape[1:]

    images = torch.zeros(batch_size, max_len, channels, height, width, dtype=batch[0]["images"].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    box_counts = torch.zeros(batch_size, max_len, dtype=torch.long)
    patch_has_cell = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["images"].shape[0]
        images[i, :n] = item["images"]
        mask[i, :n] = True
        box_counts[i, :n] = item["box_counts"]
        patch_has_cell[i, :n] = item["patch_has_cell"]

    return {
        "images": images,
        "mask": mask,
        "labels": labels,
        "bag_keys": [item["bag_key"] for item in batch],
        "slide_names": [item["slide_name"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
        "patch_names": [item["patch_names"] for item in batch],
        "patch_paths": [item["patch_paths"] for item in batch],
        "coords": [item["coords"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "box_counts": box_counts,
        "patch_has_cell": patch_has_cell,
    }


def build_final_data_loader(
    patch_root: str | Path = "database/datasets/final_data/1024_patch",
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    dataset = FinalDataPatchBagDataset(patch_root=patch_root, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_patch_bags,
    )


def iter_real_patch_paths(patch_root: str | Path) -> Iterable[Path]:
    patch_root = Path(patch_root)
    for path in patch_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
