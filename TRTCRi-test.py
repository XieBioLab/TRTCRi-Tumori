import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import EsmTokenizer, EsmConfig, EsmForSequenceClassification, AutoModelForSequenceClassification
from safetensors.torch import load_file
from train_Tumori import CustomModel

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def main():
    # Input file path and output directory
    datapath = sys.argv[1] if len(sys.argv) > 1 else "./examples/TRTCRi_test.xlsx"
    result_dir = sys.argv[2] if len(sys.argv) > 2 else "./outputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract clean filename without extension
    dataname = os.path.splitext(os.path.basename(datapath))[0]

    # Read input file (expects 6 TCR columns)
    df = pd.read_excel(datapath)
    sequences = df[['TRAV', 'CDR3a', 'TRAJ', 'TRBV', 'CDR3b', 'TRBJ']].agg('_'.join, axis=1).tolist()
    print(f"Number of input sequences: {len(sequences)}")

    # Load model and tokenizer
    model_path = "./TRTCRi"
    tokenizer = EsmTokenizer.from_pretrained(model_path, local_files_only=True)
    config = EsmConfig.from_pretrained(model_path, local_files_only=True)
    base_model = EsmForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    base_model.config.output_hidden_states = True
    config.problem_type = "single_label_classification"

    # Wrap base model with custom architecture
    model = CustomModel(base_model, hidden_dim=512, num_layers=3, dropout_rate=0.1)
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Inference loop
    probs = []
    for seq in sequences:
        enc = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=320)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids, attention_mask)
            logits = out["logits"]
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)

    # Select sequences predicted as reactive (class 1)
    threshold = 0.5
    preds = (np.array(probs) > threshold).astype(int)
    reactive_tcrs = [seq for seq, pred in zip(sequences, preds) if pred == 1]

    # Prepare output directory and file
    save_dir = os.path.join(result_dir, f"{os.path.basename(model_path)}_{dataname}")
    os.makedirs(save_dir, exist_ok=True)

    # Save reactive TCRs to text file
    output_txt = os.path.join(save_dir, "reactive_tcrs.txt")
    with open(output_txt, "w") as f:
        for tcr in reactive_tcrs:
            f.write(f"{tcr}\n")
    print(f"Saved {len(reactive_tcrs)} reactive TCRs to: {output_txt}")
    # Save probabilities to Excel
    df['TCR_sequence'] = sequences
    df['prob'] = probs
    df['pred'] = preds
    output_excel = os.path.join(save_dir, "prediction_results.xlsx")
    df.to_excel(output_excel, index=False)
    print(f"Saved full prediction results (including probabilities) to: {output_excel}")

if __name__ == "__main__":
    main()
