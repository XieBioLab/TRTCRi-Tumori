import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

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


def build_training_args(output_dir, args):
    kwargs = {
        "output_dir": str(output_dir / "trainer_checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_strategy": "epoch",
        "logging_steps": 100,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1",
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "fp16": torch.cuda.is_available() and args.fp16,
    }
    try:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy="epoch", **kwargs)


def compute_trainer_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    metrics = compute_binary_metrics(labels, probs)
    return {"accuracy": metrics["accuracy"], "f1": metrics["f1"]}


def train_fold(fold_id, train_idx, val_idx, sequences, labels, tokenizer, args, run_dir, device):
    fold_dir = run_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    config.classifier_dropout = args.classifier_dropout

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, config=config).to(device)
    train_dataset = TCRDataset(train_sequences, tokenizer, train_labels, max_length=args.max_length)
    val_dataset = TCRDataset(val_sequences, tokenizer, val_labels, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        args=build_training_args(fold_dir, args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_trainer_metrics,
    )
    trainer.train()
    trainer.save_model(fold_dir)
    tokenizer.save_pretrained(fold_dir)

    val_probs = predict_probabilities(model, val_dataset, device, batch_size=args.predict_batch_size)
    metrics = compute_binary_metrics(val_labels, val_probs, threshold=args.threshold)
    metrics["fold"] = fold_id
    save_metrics_json(metrics, fold_dir / "metrics.json")
    save_curves(val_labels, val_probs, fold_dir)

    pd.DataFrame(
        {
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
    args = parser.parse_args()

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
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(sequences), start=1):
        print(f"Training fold {fold_id}/{args.folds}")
        fold_metrics.append(
            train_fold(fold_id, train_idx, val_idx, sequences, labels, tokenizer, args, run_dir, device)
        )

    summary = pd.DataFrame(fold_metrics)
    summary.to_csv(run_dir / "fold_metrics.csv", index=False)

    numeric_cols = [col for col in summary.columns if col not in {"fold", "confusion_matrix"}]
    averages = summary[numeric_cols].mean(numeric_only=True).to_dict()
    save_metrics_json({"folds": fold_metrics, "average": averages}, run_dir / "summary.json")
    print(f"Training complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
