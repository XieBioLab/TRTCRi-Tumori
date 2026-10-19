import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

from training_utils import add_head_arguments, make_trainer, new_model, save_run_config, validate_training_arguments

from common import (
    TCRDataset,
    build_tcr_sequences,
    compute_binary_metrics,
    predict_probabilities,
    read_table,
    require_label_column,
    save_curves,
    save_metrics_json,
    set_seed,
)


def train_fold(fold_id, train_idx, val_idx, sequences, labels, tokenizer, args, run_dir, device):
    fold_dir = run_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed + fold_id)

    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    model = new_model(args).to(device)
    train_dataset = TCRDataset(train_sequences, tokenizer, train_labels, max_length=args.max_length)
    val_dataset = TCRDataset(val_sequences, tokenizer, val_labels, max_length=args.max_length)

    trainer = make_trainer(model, fold_dir, train_dataset, val_dataset, args)
    trainer.train()
    trainer.save_model(fold_dir)
    tokenizer.save_pretrained(fold_dir)

    val_probs = predict_probabilities(trainer.model, val_dataset, device, batch_size=args.predict_batch_size)
    metrics = compute_binary_metrics(val_labels, val_probs, threshold=args.threshold)
    metrics["fold"] = fold_id
    save_metrics_json(metrics, fold_dir / "metrics.json")
    save_curves(val_labels, val_probs, fold_dir)

    pd.DataFrame(
        {
            "row_index": val_idx,
            "sequence": val_sequences,
            "Label": val_labels,
            "TRTCRi_probability": val_probs,
            "TRTCRi_prediction": (val_probs >= args.threshold).astype(int),
        }
    ).to_csv(fold_dir / "validation_predictions.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the TRTCRi five-fold model.")
    parser.add_argument("--train_file", default="datasets/train_TRTCRi.xlsx")
    parser.add_argument("--base_model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output_root", default="modelresult/TRTCRi")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    add_head_arguments(parser)
    args = parser.parse_args()
    validate_training_arguments(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = read_table(args.train_file)
    sequences = build_tcr_sequences(df)
    labels = require_label_column(df, "Label")

    run_name = datetime.datetime.now().strftime("TRTCRi_ESM2_%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    save_run_config(args, run_dir, labels, "stratified sequence-record fivefold")
    fold_metrics = []
    assignments = np.zeros(len(labels), dtype=int)

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(sequences, labels), start=1):
        assignments[val_idx] = fold_id
        print(f"Training fold {fold_id}/{args.folds}")
        fold_metrics.append(
            train_fold(fold_id, train_idx, val_idx, sequences, labels, tokenizer, args, run_dir, device)
        )

    pd.DataFrame({"row_index": np.arange(len(labels)), "Label": labels, "validation_fold": assignments}).to_csv(run_dir / "fold_assignments.csv", index=False)
    summary = pd.DataFrame(fold_metrics)
    summary.to_csv(run_dir / "fold_metrics.csv", index=False)

    numeric_cols = [col for col in summary.columns if col not in {"fold", "confusion_matrix"}]
    averages = summary[numeric_cols].mean(numeric_only=True).to_dict()
    standard_deviations = summary[numeric_cols].std(ddof=1, numeric_only=True).to_dict()
    save_metrics_json({"folds": fold_metrics, "average": averages, "sample_sd": standard_deviations}, run_dir / "summary.json")
    print(f"Training complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
