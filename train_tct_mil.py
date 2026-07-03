import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from model.attention_mil import AttentionMIL, FeatureBagDataset, choose_device, collate_bags


SCRIPT_DIR = Path(__file__).resolve().parent


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def compute_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def safe_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, num_classes: int) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    try:
        if num_classes == 2 and len(np.unique(labels)) == 2:
            metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
        elif num_classes > 2 and len(np.unique(labels)) > 1:
            metrics["auc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
        else:
            metrics["auc"] = None
    except ValueError:
        metrics["auc"] = None
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    num_classes: int = 2,
) -> tuple[float, dict, pd.DataFrame]:
    train = optimizer is not None
    model.train(train)
    losses = []
    labels_all = []
    probs_all = []
    preds_all = []
    pred_rows = []

    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train):
            logits, _ = model(features, mask)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        losses.append(float(loss.detach().cpu()))
        labels_np = labels.detach().cpu().numpy()
        labels_all.extend(labels_np.tolist())
        probs_all.append(probs)
        preds_all.extend(preds.tolist())

        for i, name in enumerate(batch["names"]):
            row = {
                "name": name,
                "code": batch["codes"][i],
                "diagnosis_label_std": batch["diagnosis_label_std"][i],
                "label": int(labels_np[i]),
                "pred": int(preds[i]),
            }
            for cls_idx in range(num_classes):
                row[f"prob_{cls_idx}"] = float(probs[i, cls_idx])
            pred_rows.append(row)

    labels_arr = np.array(labels_all)
    probs_arr = np.concatenate(probs_all, axis=0)
    preds_arr = np.array(preds_all)
    metrics = safe_metrics(labels_arr, probs_arr, preds_arr, num_classes)
    metrics["loss"] = float(np.mean(losses))
    return metrics["loss"], metrics, pd.DataFrame(pred_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-process TCT Attention-MIL MVP.")
    parser.add_argument("--train-csv", default="data/datasets/tct_mvp/tct_mvp_train.csv")
    parser.add_argument("--val-csv", default="data/datasets/tct_mvp/tct_mvp_val.csv")
    parser.add_argument("--feature-root", default="features/Pathology_TCT_MVP_patch_512")
    parser.add_argument("--label-column", default="cancer")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs/tct_mvp_512")
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = choose_device(args.device)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    train_ds = FeatureBagDataset(
        str(project_path(args.train_csv)),
        str(project_path(args.feature_root)),
        args.label_column,
        args.max_patches,
        args.embed_dim,
    )
    val_ds = FeatureBagDataset(
        str(project_path(args.val_csv)),
        str(project_path(args.feature_root)),
        args.label_column,
        args.max_patches,
        args.embed_dim,
    )
    print(f"train rows={len(train_ds)} skipped={len(train_ds.skipped)}")
    print(f"val rows={len(val_ds)} skipped={len(val_ds.skipped)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bags,
    )

    model = AttentionMIL(args.embed_dim, args.hidden_dim, args.num_classes, args.dropout).to(device)
    train_labels = train_ds.data[args.label_column].astype(int).tolist()
    class_weights = compute_class_weights(train_labels, args.num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score = -1.0
    history = []
    best_preds = None
    for epoch in range(args.epochs):
        _, train_metrics, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, args.num_classes
        )
        _, val_metrics, val_preds = run_epoch(
            model, val_loader, criterion, device, None, args.num_classes
        )
        score = val_metrics.get("auc")
        if score is None:
            score = val_metrics["balanced_accuracy"]
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if score > best_score:
            best_score = float(score)
            best_preds = val_preds
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "score": best_score,
                    "class_weights": class_weights.detach().cpu(),
                },
                out_dir / "best.pt",
            )

    torch.save({"model": model.state_dict(), "args": vars(args)}, out_dir / "last.pt")
    (out_dir / "metrics.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if best_preds is not None:
        best_preds.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote outputs to {normalize_path(out_dir.resolve())}")


if __name__ == "__main__":
    main()
