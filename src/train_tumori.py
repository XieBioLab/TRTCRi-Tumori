import argparse
import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
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


def main():
    parser = argparse.ArgumentParser(description="Train the Tumori model.")
    parser.add_argument("--train_file", default="datasets/train_Tumori.csv")
    parser.add_argument("--base_model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output_root", default="modelresult/Tumori")
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--validation_size", type=float, default=0.2)
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

    train_indices, val_indices = train_test_split(
        list(range(len(labels))),
        test_size=args.validation_size,
        stratify=labels,
        random_state=args.seed,
    )

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]

    run_name = datetime.datetime.now().strftime("Tumori_ESM2_%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(args, run_dir, labels, "stratified sequence-record train/validation")
    validation_rows = set(val_indices)
    pd.DataFrame({"row_index": range(len(labels)), "Label": labels,
                  "split": ["validation" if i in validation_rows else "train" for i in range(len(labels))]}).to_csv(run_dir / "split_assignments.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = new_model(args).to(device)
    train_dataset = TCRDataset(train_sequences, tokenizer, train_labels, max_length=args.max_length)
    val_dataset = TCRDataset(val_sequences, tokenizer, val_labels, max_length=args.max_length)

    trainer = make_trainer(model, run_dir, train_dataset, val_dataset, args)
    trainer.train()
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    val_probs = predict_probabilities(trainer.model, val_dataset, device, batch_size=args.predict_batch_size)
    metrics = compute_binary_metrics(val_labels, val_probs, threshold=args.threshold)
    save_metrics_json(metrics, run_dir / "metrics.json")
    save_curves(val_labels, val_probs, run_dir)

    pd.DataFrame(
        {
            "row_index": val_indices,
            "sequence": val_sequences,
            "Label": val_labels,
            "Tumori_probability": val_probs,
            "Tumori_prediction": (val_probs >= args.threshold).astype(int),
        }
    ).to_csv(run_dir / "validation_predictions.csv", index=False)

    print(f"Training complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
