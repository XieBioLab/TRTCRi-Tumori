import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Custom Dataset for TCR sequences
class TCRDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }

# Scoring normalization function
def normalize_topk_weighted_score(score):
    """
    Normalize the TopK weighted tumor reactivity score.
    Input:
        score: float, original TopK weighted score
    Output:
        y: float, normalized score
    """
    score = np.array(score)
    x = score * 48
    y = -(np.exp(-x) - 1)
    y = np.floor(y * 100) / 100
    return y

def main():
    # Get input/output from command line
    input_file = sys.argv[1] if len(sys.argv) > 1 else "examples/melanoma-patients.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== Device used for inference: {device} =====")

    if input_file.endswith('.xlsx'):
        test_df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        test_df = pd.read_csv(input_file)
    elif input_file.endswith('.tsv'):
        test_df = pd.read_csv(input_file, sep='\t')
    else:
        raise ValueError("Input file must be .xlsx, .csv, or .tsv")

    # Combine TCR sequences
    sequences = test_df[['TRAV', 'CDR3a', 'TRAJ', 'TRBV', 'CDR3b', 'TRBJ']].agg('_'.join, axis=1).tolist()
    print(f"\nNumber of test samples: {len(sequences)}")

    # Load model
    model_path = "Tumori"
    model_name = os.path.basename(os.path.normpath(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

    # Build dataset
    test_dataset = TCRDataset(sequences, tokenizer)

    # Inference
    training_args = TrainingArguments(
        output_dir="./outputs",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args)
    print("\n===== Start predicting =====")
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Add predictions
    test_df['Prediction'] = preds
    test_df['Probability'] = probs
    test_df['Reads_Sum'] = test_df['reads_A'] + test_df['reads_B']
    total_reads = test_df['Reads_Sum'].sum()
    test_df['Frequency'] = test_df['Reads_Sum'] / total_reads

    # Top 25 by frequency
    top_df = test_df.sort_values(by='Frequency', ascending=False).head(25)

    # Weighted tumor score
    weighted_score = (top_df['Probability'] * top_df['Frequency']).sum()
    normalized_score = normalize_topk_weighted_score(weighted_score)

    print(f"\nweighted tumor reactivity score (normalized): {normalized_score:.2f}")

    # Save results
    save_path = os.path.join(output_dir, f"{model_name}_{os.path.basename(input_file)}_inference")
    os.makedirs(save_path, exist_ok=True)

    test_df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)

    with open(os.path.join(save_path, "summary.txt"), "w") as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Number of samples: {len(sequences)}\n")
        f.write(f"Positive prediction ratio: {np.mean(preds):.4f}\n")
        f.write(f"Average probability score: {np.mean(probs):.4f}\n")
        f.write(f"Top 25 weighted score (before normalization): {weighted_score:.4f}\n")
        f.write(f"Final tumor reactivity score (normalized): {normalized_score:.2f}\n")

    print(f"\n===== Results saved to: {os.path.abspath(save_path)} =====")

if __name__ == "__main__":
    main()
