"""Development grid search on a fixed stratified 80:20 TRTCRi split."""

import argparse
import datetime
from copy import copy
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from common import TCRDataset, build_tcr_sequences, read_table, require_label_column, save_metrics_json, set_seed
from training_utils import make_trainer, new_model, save_run_config, validate_training_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="datasets/train_TRTCRi.xlsx")
    parser.add_argument("--base_model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output_root", default="modelresult/grid_search")
    # Require explicit search values: the manuscript does not enumerate the grid.
    parser.add_argument("--encoder_learning_rates", type=float, nargs="+", required=True)
    parser.add_argument("--head_learning_rates", type=float, nargs="+", required=True)
    parser.add_argument("--classifier_dropouts", type=float, nargs="+", required=True)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    args = parser.parse_args()
    if any(x <= 0 for x in args.encoder_learning_rates + args.head_learning_rates):
        parser.error("Grid learning rates must be positive.")
    if any(not 0 <= x < 1 for x in args.classifier_dropouts):
        parser.error("Grid dropout rates must lie in [0, 1).")
    data = read_table(args.train_file)
    sequences = build_tcr_sequences(data)
    labels = require_label_column(data)
    train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=0.2,
                                         stratify=labels, random_state=args.seed)
    run_dir = Path(args.output_root) / datetime.datetime.now().strftime("grid_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    val_rows = set(val_idx)
    pd.DataFrame({"row_index": range(len(labels)), "Label": labels,
                  "split": ["validation" if i in val_rows else "train" for i in range(len(labels))]}).to_csv(run_dir / "split_assignments.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_set = TCRDataset([sequences[i] for i in train_idx], tokenizer,
                           [labels[i] for i in train_idx], max_length=args.max_length)
    val_set = TCRDataset([sequences[i] for i in val_idx], tokenizer,
                         [labels[i] for i in val_idx], max_length=args.max_length)
    trials = []
    for trial_id, (encoder_lr, head_lr, dropout) in enumerate(product(
            args.encoder_learning_rates, args.head_learning_rates, args.classifier_dropouts), 1):
        trial = copy(args)
        trial.learning_rate = encoder_lr
        trial.head_learning_rate = head_lr
        trial.classifier_dropout = dropout
        trial.dropout_rates = [dropout] * 3
        validate_training_arguments(trial)
        set_seed(args.seed)
        trial_dir = run_dir / f"trial_{trial_id}"
        trial_dir.mkdir()
        save_run_config(trial, trial_dir, labels, "development stratified 80:20")
        model = new_model(trial)
        trainer = make_trainer(model, trial_dir, train_set, val_set, trial)
        trainer.train()
        metrics = trainer.evaluate()
        trials.append({"trial": trial_id, "encoder_learning_rate": encoder_lr,
                       "head_learning_rate": head_lr, "classifier_dropout": dropout,
                       "validation_f1": metrics["eval_f1"],
                       "best_epoch_checkpoint": trainer.state.best_model_checkpoint})
        pd.DataFrame(trials).to_csv(run_dir / "grid_results.csv", index=False)
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(trials, key=lambda result: result["validation_f1"])
    save_metrics_json({"selection_metric": "validation F1", "best": best,
                       "formal_training_initialization": "original ESM-2, not a grid-search checkpoint"},
                      run_dir / "best_hyperparameters.json")
    print(f"Grid search complete: {run_dir}")


if __name__ == "__main__":
    main()
