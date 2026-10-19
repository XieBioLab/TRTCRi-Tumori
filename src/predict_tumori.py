import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from modeling_tcr import load_prediction_model

from common import TCRDataset, build_tcr_sequences, predict_probabilities, read_table


def normalize_topk_weighted_score(score, scale=31.92):
    score = np.asarray(score, dtype=float)
    return -np.expm1(-scale * score)


def add_sample_score(df, probability_col, top_k=None, top_fraction=0.05, scale=31.92):
    reads_cols = {"reads_A", "reads_B"}
    if not reads_cols.issubset(df.columns):
        return None

    scored = df.copy()
    scored["Reads_Sum"] = scored["reads_A"].astype(float) + scored["reads_B"].astype(float)
    if not np.isfinite(scored["Reads_Sum"]).all() or (scored[["reads_A", "reads_B"]].astype(float) < 0).any().any():
        raise ValueError("Read support must be finite and nonnegative.")
    total_reads = scored["Reads_Sum"].sum()
    if total_reads <= 0:
        return None

    scored["Frequency"] = scored["Reads_Sum"] / total_reads
    if not 0 < top_fraction <= 1 or scale <= 0:
        raise ValueError("Top fraction must lie in (0, 1]; scale must be positive.")
    selected_k = max(1, int(np.ceil(top_fraction * len(scored)))) if top_k is None else top_k
    if selected_k < 1:
        raise ValueError("Top K must be positive.")
    selected_k = min(selected_k, len(scored))
    # Stable sorting retains input order for tied read support. Do not renormalize.
    top_df = scored.sort_values("Frequency", ascending=False, kind="stable").head(selected_k)
    weighted_score = float((top_df[probability_col] * top_df["Frequency"]).sum())
    normalized_score = float(normalize_topk_weighted_score(weighted_score, scale=scale))
    return scored, {
        "top_k": int(selected_k),
        "top_fraction": top_fraction,
        "scale": scale,
        "weighted_score": weighted_score,
        "normalized_score": normalized_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Tumori inference.")
    parser.add_argument("--input_file", default="examples/melanoma-patients.csv")
    parser.add_argument("--model_dir", default="model_weights/Tumori")
    parser.add_argument("--output_dir", default="outputs/Tumori")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=None, help="Optional fixed K override; default uses Top 5%%.")
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=31.92)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_table(args.input_file)
    sequences = build_tcr_sequences(df)

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = load_prediction_model(model_dir).to(device)
    dataset = TCRDataset(sequences, tokenizer, labels=None, max_length=args.max_length)
    probs = predict_probabilities(model, dataset, device, batch_size=args.batch_size)

    df["Tumori_sequence"] = sequences
    df["Tumori_probability"] = probs
    df["Tumori_prediction"] = (probs >= args.threshold).astype(int)

    sample_score = add_sample_score(df, "Tumori_probability", args.top_k,
                                    top_fraction=args.top_fraction, scale=args.scale)
    summary = {
        "n_sequences": int(len(df)),
        "positive_ratio": float(df["Tumori_prediction"].mean()),
        "mean_probability": float(df["Tumori_probability"].mean()),
    }
    if sample_score is not None:
        df, score_info = sample_score
        summary.update(score_info)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input_file).stem
    df.to_csv(output_dir / f"{stem}_Tumori_predictions.csv", index=False)
    pd.Series(summary).to_json(output_dir / f"{stem}_Tumori_summary.json", indent=2)
    print(f"Saved predictions to: {output_dir}")


if __name__ == "__main__":
    main()
