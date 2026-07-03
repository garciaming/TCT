from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def uniform_select(x: torch.Tensor, max_patches: int) -> torch.Tensor:
    if max_patches <= 0 or x.shape[0] <= max_patches:
        return x
    indices = torch.linspace(0, x.shape[0] - 1, max_patches).round().long()
    return x[indices]


class FeatureBagDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        feature_root: str,
        label_column: str,
        max_patches: int,
        embed_dim: int,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.feature_root = Path(feature_root)
        self.label_column = label_column
        self.max_patches = max_patches
        self.embed_dim = embed_dim

        data = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        rows = []
        skipped = []
        for _, row in data.iterrows():
            name = row["name"]
            feature_path = self.feature_root / name / "torch" / "images.pt"
            if feature_path.is_file():
                rows.append(row.to_dict() | {"feature_path": str(feature_path)})
            else:
                skipped.append(name)
        if not rows:
            raise ValueError(f"No feature files found for {csv_path} under {feature_root}")

        self.data = pd.DataFrame(rows).reset_index(drop=True)
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        feature = torch.load(row["feature_path"], map_location="cpu").float()
        if feature.ndim != 2:
            raise ValueError(f"Expected [N, D] feature tensor: {row['feature_path']}")
        if feature.shape[1] != self.embed_dim:
            raise ValueError(
                f"Feature dim mismatch for {row['feature_path']}: "
                f"{feature.shape[1]} != {self.embed_dim}"
            )
        feature = uniform_select(feature, self.max_patches)
        label = int(row[self.label_column])
        return {
            "features": feature,
            "label": label,
            "name": row["name"],
            "code": row.get("code", row["name"]),
            "diagnosis_label_std": row.get("diagnosis_label_std", ""),
        }


def collate_bags(batch: list[dict]) -> dict:
    max_len = max(item["features"].shape[0] for item in batch)
    embed_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_len, embed_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        features[i, :n] = item["features"]
        mask[i, :n] = True

    return {
        "features": features,
        "mask": mask,
        "labels": labels,
        "names": [item["name"] for item in batch],
        "codes": [item["code"] for item in batch],
        "diagnosis_label_std": [item["diagnosis_label_std"] for item in batch],
    }


class AttentionMIL(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        attn_logits = self.attention(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, torch.finfo(attn_logits.dtype).min)
        attn = torch.softmax(attn_logits, dim=1)
        bag = torch.sum(h * attn.unsqueeze(-1), dim=1)
        logits = self.classifier(bag)
        return logits, attn
