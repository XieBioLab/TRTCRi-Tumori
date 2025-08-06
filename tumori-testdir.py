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
    score = np.array(score)
    x = score * 48
    y = -(np.exp(-x) - 1)
    y = np.floor(y * 100) / 100
    return y

def load_dataframe(filepath):
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.tsv'):
        return pd.read_csv(filepath, sep='\t')
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

def process_file(input_file, output_dir, tokenizer, model, model_name, device):
    print(f"\n>>> Processing file: {input_file}")
    test_df = load_dataframe(input_file)

    # Combine TCR sequence
    sequences = test_df[['TRAV', 'CDR3a', 'TRAJ', 'TRBV', 'CDR3b', 'TRBJ']].agg('_'.join, axis=1).tolist()
    print(f"  Number of test samples: {len(sequences)}")

    test_dataset = TCRDataset(sequences, tokenizer)

    training_args = TrainingArguments(
        output_dir="./temp_test_output",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args)
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

    # Top 25 scoring
    top_df = test_df.sort_values(by='Frequency', ascending=False).head(25)
    weighted_score = (top_df['Probability'] * top_df['Frequency']).sum()
    normalized_score = normalize_topk_weighted_score(weighted_score)

    # Save results
    file_basename = os.path.splitext(os.path.basename(input_file))[0]
    save_path = os.path.join(output_dir, f"{model_name}_{file_basename}_inference")
    os.makedirs(save_path, exist_ok=True)

    test_df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)

    with open(os.path.join(save_path, "summary.txt"), "w") as f:
        f.write(f"Model name: {model_name}\n")
        f.write(f"Number of samples: {len(sequences)}\n")
        f.write(f"Positive prediction ratio: {np.mean(preds):.4f}\n")
        f.write(f"Average probability score: {np.mean(probs):.4f}\n")
        f.write(f"Top 25 weighted score (before normalization): {weighted_score:.4f}\n")
        f.write(f"Final tumor reactivity score (normalized): {normalized_score:.2f}\n")

    print(f"  ✓ Saved to: {os.path.abspath(save_path)}")
    print(f"  ✓ Normalized tumor score: {normalized_score:.2f}")

def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "examples"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== Using device: {device} =====")

    # Load model and tokenizer
    model_path = "Tumori"
    model_name = os.path.basename(os.path.normpath(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)

    # Iterate all valid files in the directory
    valid_exts = (".csv", ".tsv", ".xlsx")
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(valid_exts)]

    if not all_files:
        print(f"No valid files found in {input_dir}. Supported formats: .csv, .tsv, .xlsx")
        return

    for file_path in all_files:
        try:
            process_file(file_path, output_dir, tokenizer, model, model_name, device)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
