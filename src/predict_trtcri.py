import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from modeling_tcr import load_prediction_model

from common import TCRDataset, build_tcr_sequences, predict_probabilities, read_table


def find_model_dirs(model_dir):
    root = Path(model_dir)
    fold_dirs = [root / f"fold_{i}" for i in range(1, 6)]
    existing_folds = [path for path in fold_dirs if (path / "config.json").exists()]
    if existing_folds:
        return existing_folds
    if (root / "config.json").exists():
        return [root]
    raise FileNotFoundError(f"No Hugging Face model directory found under: {root}")


def main():
    parser = argparse.ArgumentParser(description="Run TRTCRi inference with a single model or fold ensemble.")
    parser.add_argument("--input_file", default="examples/TRTCRi_test.xlsx")
    parser.add_argument("--model_dir", default="model_weights/TRTCRi")
    parser.add_argument("--output_dir", default="outputs/TRTCRi")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_table(args.input_file)
    sequences = build_tcr_sequences(df)
    model_dirs = find_model_dirs(args.model_dir)

    all_probs = []
    for model_path in model_dirs:
        print(f"Loading TRTCRi model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = load_prediction_model(model_path).to(device)
        dataset = TCRDataset(sequences, tokenizer, labels=None, max_length=args.max_length)
        all_probs.append(predict_probabilities(model, dataset, device, batch_size=args.batch_size))

    prob_matrix = np.vstack(all_probs)
    df["TRTCRi_sequence"] = sequences
    df["TRTCRi_probability"] = prob_matrix.mean(axis=0)
    df["TRTCRi_probability_std"] = prob_matrix.std(axis=0) if len(all_probs) > 1 else 0.0
    df["TRTCRi_prediction"] = (df["TRTCRi_probability"] >= args.threshold).astype(int)
    df["TRTCRi_predicted_class"] = np.where(df["TRTCRi_prediction"] == 1, "Tumor-reactive", "Tumor-nonreactive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(args.input_file).stem}_TRTCRi_predictions.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
