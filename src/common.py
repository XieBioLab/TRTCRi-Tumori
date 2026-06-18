import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Dataset


TCR_COLUMNS = ["TRAV", "CDR3a", "TRAJ", "TRBV", "CDR3b", "TRBJ"]


class TCRDataset(Dataset):
    def __init__(self, sequences, tokenizer, labels=None, max_length=320):
        self.sequences = list(sequences)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.sequences[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError("Input file must be .xlsx, .xls, .csv, or .tsv")


def build_tcr_sequences(df):
    missing = [col for col in TCR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required TCR columns: {missing}")
    return df[TCR_COLUMNS].fillna("").astype(str).agg("_".join, axis=1).tolist()


def require_label_column(df, label_col="Label"):
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    return df[label_col].astype(int).tolist()


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def save_metrics_json(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_curves(y_true, y_prob, output_dir, prefix="validation"):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) != 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_roc_curve.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_pr_curve.pdf")
    plt.close(fig)


def predict_probabilities(model, dataset, device, batch_size=16):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(batch_probs.detach().cpu().numpy().tolist())
    return np.asarray(probs, dtype=float)


def make_output_dir(root, model_name):
    output_dir = Path(root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
