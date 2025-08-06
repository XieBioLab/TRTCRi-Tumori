import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    auc
)
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoConfig,AutoModelForSequenceClassification
import os
import datetime
import json
import argparse #Added for command-line arguments
from captum.attr import LayerIntegratedGradients  # Dependency: requires captum to be installed
from captum.attr import visualization as viz

# Set global style (using a simple, colorblind-friendly style common in Nature publications)
sns.set(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "font.family": "Arial",       # or Helvetica
    "font.size": 12,
    "figure.dpi": 500,            # High-resolution output
    "savefig.format": "pdf",      # Can be changed to pdf or svg
})

class TCRDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class CustomModel(torch.nn.Module):
    def __init__(self, base_model, hidden_dim, num_layers, dropout_rate):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(base_model.config.hidden_size, hidden_dim, num_layers,
                                  batch_first=True, dropout=dropout_rate, bidirectional=True)
        # Input size is hidden_dim * 2 due to bidirectionality
        self.fc = torch.nn.Linear(hidden_dim * 2, base_model.config.num_labels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get the output hidden states from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # shape: (batch_size, seq_length, hidden_size)

        # Process through LSTM
        lstm_out, _ = self.lstm(last_hidden_state)  # shape: (batch_size, seq_length, hidden_dim * 2)

        # Take the output of the last time step
        logits = self.fc(self.dropout(lstm_out[:, -1, :]))  # shape: (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def integrated_gradients(model, tokenizer, sequence, target_class=1, n_steps=20, max_length=None, save_path=None):
    cpu_model = model.to('cpu').eval()

    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", None) or 512

    encoding = tokenizer(
        sequence, return_tensors="pt", padding="longest", truncation=True, max_length=max_length
    )
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    embed_layer = cpu_model.base_model.get_input_embeddings()

    def forward_from_ids(ids, mask):
        outputs = cpu_model.base_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        lstm_out, _ = cpu_model.lstm(last_hidden)
        logits = cpu_model.fc(cpu_model.dropout(lstm_out[:, -1, :]))
        return logits

    lig = LayerIntegratedGradients(forward_from_ids, embed_layer)
    baseline_ids = torch.zeros_like(input_ids)
    attributions, _ = lig.attribute(
        inputs=input_ids, baselines=baseline_ids, additional_forward_args=(attention_mask,),
        target=target_class, n_steps=n_steps, return_convergence_delta=True
    )

    attr_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # ---------- Figure 1: Token-level Attribution ----------
    tokens = tokens[:512]
    attr_scores = attr_scores[:512]
    fig_width = max(12, len(tokens) * 0.4)
    fig_token, ax_token = plt.subplots(figsize=(fig_width, 4))
    x = list(range(len(tokens)))
    ax_token.bar(x, attr_scores)
    ax_token.set_xticks(x)
    ax_token.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax_token.set_xlabel("Token", fontsize=12, weight="bold")
    ax_token.set_ylabel("Attribution Score", fontsize=12, weight="bold")
    ax_token.set_title(f"Token-level IG (class={target_class})", fontsize=14, weight="bold")
    plt.tight_layout()

    if save_path:
        fig_token.savefig(save_path.replace(".pdf", "_token.pdf"), bbox_inches="tight", dpi=500)
        plt.close(fig_token)

    # ---------- Figure 2: Region-level Attribution ----------
    regions = ["TRAV", "CDR3a", "TRAJ", "TRBV", "CDR3b", "TRBJ"]
    aa_parts = sequence.split("_")
    if len(aa_parts) != 6:
        print(f"[Warning] Sample parsing failed: Expected 6 regions but found {len(aa_parts)}")
        return fig_token, None

    region_tokens = [tokenizer.tokenize(p) for p in aa_parts]
    region_lengths = [len(toks) for toks in region_tokens]
    region_scores = []
    token_idx = 1  # skip [CLS]
    for region_len in region_lengths:
        region_attr = attr_scores[token_idx: token_idx + region_len]
        region_scores.append(np.mean(region_attr))
        token_idx += region_len

    region_names_pretty = ["TRAV", "CDR3α", "TRAJ", "TRBV", "CDR3β", "TRBJ"]
    fig_region, ax_region = plt.subplots(figsize=(8, 4))
    ax_region.bar(region_names_pretty, region_scores, color=sns.color_palette("colorblind", 6))
    ax_region.set_ylabel("Mean Attribution Score", fontsize=12, weight="bold")
    ax_region.set_title(f"Region-level IG (class={target_class})", fontsize=14, weight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig_region.savefig(save_path.replace(".pdf", "_region.pdf"), bbox_inches="tight", dpi=500)
        plt.close(fig_region)

    # ---------- Figure 3: Token Heatmap using actual tokens ----------
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(min(40, len(tokens) * 0.3), 1.5))
    sns.heatmap(np.expand_dims(attr_scores, axis=0), cmap="coolwarm", cbar=True,
                xticklabels=tokens, yticklabels=["IG"], ax=ax_heatmap)
    ax_heatmap.tick_params(axis='x', rotation=90, labelsize=6)
    plt.tight_layout()

    if save_path:
        fig_heatmap.savefig(save_path.replace(".pdf", "_token_heatmap.pdf"), bbox_inches="tight", dpi=500)
        plt.close(fig_heatmap)

    # ---------- Figure 4: Structured Region-wise Heatmap ----------
    try:
        heatmap_data, region_labels = [], []
        token_idx = 1  # Skip [CLS]
        for i, region_len in enumerate(region_lengths):
            region_attr = attr_scores[token_idx: token_idx + region_len]
            heatmap_data.append(region_attr)
            region_labels.append(region_names_pretty[i])
            token_idx += region_len
        # Pad rows to the same length
        max_len = max(len(r) for r in heatmap_data)
        heatmap_data_padded = [np.pad(r, (0, max_len - len(r)), constant_values=np.nan) for r in heatmap_data]
        fig_structured, ax_structured = plt.subplots(figsize=(max(10, max_len * 0.4), 3))
        sns.heatmap(heatmap_data_padded, cmap="coolwarm", cbar=True, yticklabels=region_labels,
                    xticklabels=False, ax=ax_structured, linewidths=0.5, linecolor='gray')
        ax_structured.set_title("Structured Region-wise IG Heatmap", fontsize=14, weight="bold")
        plt.tight_layout()

        if save_path:
            fig_structured.savefig(save_path.replace(".pdf", "_region_heatmap.pdf"), bbox_inches="tight", dpi=500)
            plt.close(fig_structured)
    except Exception as e:
        print(f"[Warning] Could not draw structured region heatmap: {e}")

    return fig_token, fig_region

def visualize_attention(model, tokenizer, sequence, layer=-1, save_path=None):
    cpu_model = model.to('cpu').eval()
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
    with torch.no_grad():
        outputs = cpu_model.base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], output_attentions=True)

    attentions = outputs.attentions[layer].numpy()
    avg_attentions = np.mean(attentions, axis=1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Check if mapping to region labels is needed (if sequence is too long)
    if len(tokens) > 80:
        parts = sequence.split("_")
        regions = ["TRAV", "CDR3a", "TRAJ", "TRBV", "CDR3b", "TRBJ"]
        region_labels = []
        # This logic for region mapping is approximate and may need adjustment based on tokenizer behavior
        current_token_idx = 1 # Skip [CLS]
        for region_name, part in zip(regions, parts):
             part_tokens = tokenizer.tokenize(part)
             region_labels.extend([region_name] * len(part_tokens))
        label_tokens = ["[CLS]"] + region_labels[:len(tokens)-2] + ["[SEP]"] # Ensure length matches
    else:
        label_tokens = tokens

    fig_size = max(10, len(label_tokens) * 0.35)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(avg_attentions, xticklabels=label_tokens, yticklabels=label_tokens, cmap="viridis", square=True)
    plt.title(f"Attention Heatmap (Layer {layer})", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=1000)
        plt.close()

# Plot ROC and PR curves
def plot_metrics(y_true, y_probs, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="#D55E00", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    plt.xlabel("False Positive Rate", fontsize=14, weight="bold")
    plt.ylabel("True Positive Rate", fontsize=14, weight="bold")
    plt.title("ROC Curve", fontsize=16, weight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "roc_curve.pdf"), bbox_inches="tight")
    plt.close()
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color="#0072B2", lw=2, label=f"PR (AUPR = {pr_auc:.2f})")
    plt.xlabel("Recall", fontsize=14, weight="bold")
    plt.ylabel("Precision", fontsize=14, weight="bold")
    plt.title("Precision-Recall Curve", fontsize=16, weight="bold")
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, "pr_curve.pdf"), bbox_inches="tight")
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, output_dir="./results"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
    plt.xlabel("Predicted Labels", fontsize=14, weight="bold")
    plt.ylabel("True Labels", fontsize=14, weight="bold")
    plt.title("Confusion Matrix", fontsize=16, weight="bold")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.pdf"), bbox_inches="tight")
    plt.close()

# Main function
def main():
    #  Handle command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a TCR-peptide binding prediction model.")
    parser.add_argument('--train_file', type=str, default='./datasets/train_Tumori.xlsx',
                        help='Path to the training data file (Excel format).')
    args = parser.parse_args()

    print("\n===== Hardware Configuration Check =====")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data from the specified path
    try:
        df = pd.read_excel(args.train_file)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_file}")
        return

    sequences = df[['TRAV', 'CDR3a', 'TRAJ', 'TRBV', 'CDR3b', 'TRBJ']].agg('_'.join, axis=1).tolist()
    labels = df['Label'].tolist()
    print(f"\nTotal Samples: {len(sequences)}")

    # Split into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

    # Set up paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main_run_dir = os.path.join("modelresult", f"LSTM_{timestamp}")
    save_dir = os.path.join(main_run_dir, "train_val_split")
    os.makedirs(save_dir, exist_ok=True)

    # Load model and tokenizer
    model_path = "models/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.num_labels = 2
    base_model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    base_model.config.output_hidden_states = True
    base_model.config.problem_type = "single_label_classification"
    base_model.config.classifier_dropout = 0.1

    # Wrap with custom LSTM model
    hidden_dim = 128
    num_layers = 2
    dropout_rate = 0.1
    learning_rate = 4.5e-5
    model = CustomModel(base_model, hidden_dim, num_layers, dropout_rate).to(device)

    # Create Datasets
    train_dataset = TCRDataset(train_texts, train_labels, tokenizer)
    val_dataset = TCRDataset(val_texts, val_labels, tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "results"),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(save_dir, "logs"),
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        report_to="none",
        lr_scheduler_type="cosine"
    )

    #Save experiment configuration to config.json
    exp_config = {
        "training_file": args.train_file,
        "model_base_path": model_path,
        "lstm_hidden_dim": hidden_dim,
        "lstm_num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "num_epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "validation_split_size": 0.2,
        "random_state": 42
    }
    with open(os.path.join(main_run_dir, 'config.json'), 'w') as f:
        json.dump(exp_config, f, indent=4)
    print(f"\nExperiment configuration saved to {os.path.join(main_run_dir, 'config.json')}")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Evaluate model
    val_result = trainer.evaluate()
    val_pred = trainer.predict(val_dataset)
    probs = torch.softmax(torch.tensor(val_pred.predictions), dim=1).numpy()[:, 1]
    val_labels_true = val_pred.label_ids
    np.save(os.path.join(save_dir, 'probs.npy'), probs)
    np.save(os.path.join(save_dir, 'labels.npy'), val_labels_true)

    plot_metrics(val_labels_true, probs, output_dir=save_dir)
    preds = np.argmax(val_pred.predictions, axis=1)
    plot_confusion_matrix(val_labels_true, preds, output_dir=save_dir)

    # Feature Visualization
    def extract_features(loader):
        model.eval()
        features, labels_list = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features for t-SNE"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = model.base_model(**inputs)
                lstm_out, _ = model.lstm(outputs.hidden_states[-1])
                features.append(lstm_out[:, -1, :].cpu().numpy())
                labels_list.append(batch['labels'].cpu().numpy())
        return np.concatenate(features), np.concatenate(labels_list)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                            collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) if k!='labels' else torch.tensor([x[k] for x in b]) for k in b[0]})
    features, true_labels = extract_features(val_loader)
    tsne_results = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42).fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=true_labels, palette="viridis", alpha=0.8, s=50, edgecolor="none")
    plt.title("t-SNE Feature Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Class", labels=["Negative", "Positive"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne_visualization.pdf"), dpi=500)
    plt.close()

    # Interpretability Analysis
    sample_seq = "No positive samples in validation set"
    try:
        analysis_dir = os.path.join(save_dir, "interpretability")
        os.makedirs(analysis_dir, exist_ok=True)
        positive_indices = np.where(np.array(val_labels) == 1)[0]
        if len(positive_indices) > 0:
            sample_idx_in_val = positive_indices[0]
            sample_seq = val_texts[sample_idx_in_val]
            cpu_model = model.to('cpu').eval()
            if len(sample_seq.split("_")) < 6:
                print(f"Sample {sample_seq} has insufficient region tags, skipping attribution analysis.")
            else:
                integrated_gradients(cpu_model, tokenizer, sample_seq, target_class=1, n_steps=50,
                                     save_path=os.path.join(analysis_dir, "integrated_gradients.pdf"))
                for layer in [-3, -2, -1]:
                    visualize_attention(cpu_model, tokenizer, sample_seq, layer=layer,
                                        save_path=os.path.join(analysis_dir, f"attention_layer_{layer}.pdf"))
    except Exception as e:
        print(f"Interpretability analysis failed: {e}")

    # Write Report
    roc_auc_val = roc_auc_score(val_labels_true, probs)
    pr_auc_val = average_precision_score(val_labels_true, probs)
    report_content = f"""Training Report
Experiment Time: {datetime.datetime.now()}
Hardware: PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, Device={device}
Training Parameters: Epochs={training_args.num_train_epochs}, LR={training_args.learning_rate}, Batch={training_args.per_device_train_batch_size}
Results: Loss={val_result['eval_loss']:.4f}, Acc={val_result['eval_accuracy']:.4f}, F1={val_result['eval_f1']:.4f}, AUC={roc_auc_val:.4f}, PR-AUC={pr_auc_val:.4f}
Attribution and attention maps generated for sample: {sample_seq}
"""
    with open(os.path.join(save_dir, "training_report.txt"), "w") as f:
        f.write(report_content)

    print(f"\nTraining and analysis complete. Report saved to: {save_dir}")

if __name__ == "__main__":
    main()