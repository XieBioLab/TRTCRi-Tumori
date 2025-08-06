import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
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
from transformers import AutoTokenizer, TrainingArguments, Trainer
import os
import datetime
from transformers import EsmConfig, AutoModelForSequenceClassification,EsmTokenizer,AutoConfig
import json
import argparse # ① Added for command-line arguments
from captum.attr import LayerIntegratedGradients # Dependency: need to install captum
from captum.attr import visualization as viz

# Set global style (using a simple, colorblind-friendly style common in Nature publications)
sns.set(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "font.family": "Arial",       # or Helvetica
    "font.size": 12,
    "figure.dpi": 500,            # high-resolution output
    "savefig.format": "pdf",      # can be changed to pdf or svg
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
        # Input size is hidden_dim * 2 because of bidirectionality
        self.fc = torch.nn.Linear(hidden_dim * 2, base_model.config.num_labels)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, labels=None):
        # Get the hidden states from the base model
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
        sequence,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    embed_layer = cpu_model.base_model.get_input_embeddings()

    def forward_from_ids(ids, mask):
        outputs = cpu_model.base_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        lstm_out, _ = cpu_model.lstm(last_hidden)
        logits = cpu_model.fc(cpu_model.dropout(lstm_out[:, -1, :]))
        return logits

    lig = LayerIntegratedGradients(forward_from_ids, embed_layer)
    baseline_ids = torch.zeros_like(input_ids)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )

    attr_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # ---------- Figure 1: Token-level Attribution ----------
    MAX_TOKENS = 512
    tokens = tokens[:MAX_TOKENS]
    attr_scores = attr_scores[:MAX_TOKENS]

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
    sns.heatmap(np.expand_dims(attr_scores, axis=0),
                cmap="coolwarm",
                cbar=True,
                xticklabels=tokens,  # Use actual tokens
                yticklabels=["IG"],
                ax=ax_heatmap)
    ax_heatmap.tick_params(axis='x', rotation=90, labelsize=6)
    plt.tight_layout()

    if save_path:
        fig_heatmap.savefig(save_path.replace(".pdf", "_token_heatmap.pdf"), bbox_inches="tight", dpi=500)
        plt.close(fig_heatmap)

    # ---------- Figure 4: Structured Region-wise Heatmap (one row per region) ----------
    try:
        heatmap_data = []
        region_labels = []
        token_idx = 1  # Skip [CLS]

        for i, region_len in enumerate(region_lengths):
            region_attr = attr_scores[token_idx: token_idx + region_len]
            heatmap_data.append(region_attr)
            region_labels.append(region_names_pretty[i])
            token_idx += region_len

        # Pad rows to the same length for the heatmap
        max_len = max(len(r) for r in heatmap_data)
        heatmap_data_padded = [np.pad(r, (0, max_len - len(r)), constant_values=np.nan) for r in heatmap_data]

        fig_structured, ax_structured = plt.subplots(figsize=(max(10, max_len * 0.4), 3))
        sns.heatmap(heatmap_data_padded, cmap="coolwarm", cbar=True,
                    yticklabels=region_labels, xticklabels=False, ax=ax_structured,
                    linewidths=0.5, linecolor='gray')
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
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = cpu_model.base_model(input_ids, attention_mask=attention_mask, output_attentions=True)

    attentions = outputs.attentions[layer].numpy()
    avg_attentions = np.mean(attentions, axis=1)[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Determine if labels should be mapped to regions (if sequence is too long)
    MAX_LABELS = 80  # Threshold to merge tokens into region labels
    if len(tokens) > MAX_LABELS:
        parts = sequence.split("_")
        regions = ["TRAV", "CDR3a", "TRAJ", "TRBV", "CDR3b", "TRBJ"]
        region_tokens = [tokenizer.tokenize(p) for p in parts]
        region_lengths = [len(toks) for toks in region_tokens]
        region_labels = []
        idx = 1 # Skip [CLS]
        for region_name, r_len in zip(regions, region_lengths):
            region_labels.extend([region_name] * r_len)
        # Manually construct labels for special tokens and regions
        label_tokens = ["[CLS]"] + region_labels[:len(tokens)-2] + ["[SEP]"]
    else:
        label_tokens = tokens

    fig_size = max(10, len(label_tokens) * 0.35)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(avg_attentions,
                xticklabels=label_tokens,
                yticklabels=label_tokens,
                cmap="viridis",
                square=True)
    plt.title(f"Attention Heatmap (Layer {layer})", fontsize=14)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=1000)
        plt.close()

    return avg_attentions

# Plot ROC and PR curves
def plot_metrics(y_true, y_probs, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)

    # ROC Curve
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

    # PR Curve
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

    return roc_auc, pr_auc

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
    # ② Handle command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate a TCR-peptide binding prediction model.")
    parser.add_argument('--train_file', type=str, default='./datasets/train_TRTCRi.xlsx',
                        help='Path to the training data file (Excel format). Defaults to the original script path.')
    args = parser.parse_args()

    print("\n===== Hardware Configuration Check =====")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data from the path provided via command line
    try:
        df = pd.read_excel(args.train_file)
    except FileNotFoundError:
        print(f"Error: Training file not found at {args.train_file}")
        return

    sequences = df[['TRAV', 'CDR3a', 'TRAJ', 'TRBV', 'CDR3b', 'TRBJ']].agg('_'.join, axis=1).tolist()
    labels = df['Label'].tolist()

    print(f"\nTotal Samples: {len(sequences)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []
    timestamp_all = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    main_save_dir = os.path.join("modelresult", f"LSTM_{timestamp_all}")
    os.makedirs(main_save_dir, exist_ok=True)  # Main folder for this run

    # Define hyperparameters
    model_path = "models/"
    hidden_dim = 512
    num_layers = 3
    dropout_rate = 0.25
    learning_rate = 4.5e-5
    num_epochs = 10
    batch_size = 2
    grad_accum_steps = 4

    # ③ Save experiment configuration to config.json
    exp_config = {
        "model_base_path": model_path,
        "lstm_hidden_dim": hidden_dim,
        "lstm_num_layers": num_layers,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "k_folds": kf.get_n_splits(),
        "random_state": 42
    }
    with open(os.path.join(main_save_dir, 'config.json'), 'w') as f:
        json.dump(exp_config, f, indent=4)
    print(f"\nExperiment configuration saved to {os.path.join(main_save_dir, 'config.json')}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"\n===== Fold {fold + 1} =====")
        train_texts = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 2
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        base_model.config.output_hidden_states = True
        base_model.config.problem_type = "single_label_classification"
        base_model.config.classifier_dropout = 0.1

        model = CustomModel(base_model, hidden_dim, num_layers, dropout_rate).to(device)

        train_dataset = TCRDataset(train_texts, train_labels, tokenizer)
        val_dataset = TCRDataset(val_texts, val_labels, tokenizer)

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold+1}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"./logs/fold_{fold+1}",
            logging_steps=100,
            learning_rate=learning_rate,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            report_to="none",
            lr_scheduler_type="cosine"
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = np.argmax(pred.predictions, axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            return {"accuracy": acc, "f1": f1}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        save_dir = os.path.join(main_save_dir, f"fold_{fold+1}")
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir) # This saves the model's config.json
        tokenizer.save_pretrained(save_dir)

        val_result = trainer.evaluate()
        val_pred = trainer.predict(val_dataset)
        probs = torch.softmax(torch.tensor(val_pred.predictions), dim=1).numpy()[:, 1]
        val_labels_true = val_pred.label_ids

        np.save(os.path.join(save_dir, 'probs.npy'), probs)
        np.save(os.path.join(save_dir, 'labels.npy'), val_labels_true)

        roc_auc_val, pr_auc_val = plot_metrics(val_labels_true, probs, output_dir=save_dir)

        preds = np.argmax(val_pred.predictions, axis=1)
        plot_confusion_matrix(val_labels_true, preds, output_dir=save_dir)

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
                'labels': torch.tensor([x['labels'] for x in batch])
            }
        )

        def extract_features():
            model.eval()
            features, labels_list = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Extracting features for t-SNE"):
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    labels = batch['labels'].to(device)
                    outputs = model.base_model(**inputs)
                    last_hidden = outputs.hidden_states[-1]
                    lstm_out, _ = model.lstm(last_hidden)
                    batch_features = lstm_out[:, -1, :].cpu().numpy()
                    features.append(batch_features)
                    labels_list.append(labels.cpu().numpy())
            return np.concatenate(features), np.concatenate(labels_list)

        features, true_labels = extract_features()
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=1000, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=true_labels,
            palette="viridis",
            alpha=0.8,
            s=50,
            edgecolor="none"
        )
        plt.title("t-SNE Feature Visualization")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Class", labels=["Negative", "Positive"])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "tsne_visualization.pdf"), dpi=500)
        plt.close()

        try:
            analysis_dir = os.path.join(save_dir, "interpretability")
            os.makedirs(analysis_dir, exist_ok=True)
            # Find a positive sample for analysis
            positive_indices = np.where(val_labels_true == 1)[0]
            if len(positive_indices) > 0:
                sample_idx = positive_indices[0]
                sample_seq = val_texts[sample_idx]

                cpu_model = model.to('cpu').eval()
                parts = sample_seq.split("_")
                if( len(parts) < 6):
                    print(f"Sample {sample_seq} has insufficient region tags, skipping attribution analysis.")
                else:
                    integrated_gradients(cpu_model, tokenizer, sample_seq, target_class=1, n_steps=50,
                                         save_path=os.path.join(analysis_dir, "integrated_gradients.pdf"))
                    for layer in [-3, -2, -1]:
                        visualize_attention(cpu_model, tokenizer, sample_seq, layer=layer,
                                            save_path=os.path.join(analysis_dir, f"attention_layer_{layer}.pdf"))
            else:
                sample_seq = "No positive samples in validation set"
                print("Skipping interpretability analysis: No positive samples found in this validation fold.")

        except Exception as e:
            print(f"Interpretability analysis failed: {e}")

        roc_auc_score_val = roc_auc_score(val_labels_true, probs)
        pr_auc_score_val = average_precision_score(val_labels_true, probs)

        report_content = f"""Training Report (Fold {fold + 1})
Experiment Time: {datetime.datetime.now()}
Hardware: PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, Device={device}
Training Parameters: Epochs={training_args.num_train_epochs}, LR={training_args.learning_rate}, Batch={training_args.per_device_train_batch_size}
Results: Loss={val_result['eval_loss']:.4f}, Acc={val_result['eval_accuracy']:.4f}, F1={val_result['eval_f1']:.4f}, AUC={roc_auc_score_val:.4f}, PR-AUC={pr_auc_score_val:.4f}
Attribution and attention maps generated for sample: {sample_seq}
"""
        with open(os.path.join(save_dir, "training_report.txt"), "w") as f:
            f.write(report_content)

        all_results.append({
            "fold": fold + 1,
            "loss": val_result['eval_loss'],
            "acc": val_result['eval_accuracy'],
            "f1": val_result['eval_f1'],
            "roc_auc": roc_auc_score_val,
            "pr_auc": pr_auc_score_val
        })

    print(f"\n===== {kf.get_n_splits()}-Fold Cross-Validation Summary =====")
    for res in all_results:
        print(res)
    avg = lambda k: np.mean([r[k] for r in all_results])
    print(f"Average Loss: {avg('loss'):.4f}, Average Acc: {avg('acc'):.4f}, Average F1: {avg('f1'):.4f}, Average AUC: {avg('roc_auc'):.4f}, Average PR-AUC: {avg('pr_auc'):.4f}")

    # Save the average results to the 'average' subdirectory
    average_dir = os.path.join(main_save_dir, "average")
    os.makedirs(average_dir, exist_ok=True)

    average_report = f"""Cross-Validation Average Results Summary
Time: {datetime.datetime.now()}
Total Folds: {len(all_results)}
Average Loss: {avg('loss'):.4f}
Average Accuracy: {avg('acc'):.4f}
Average F1 Score: {avg('f1'):.4f}
Average ROC AUC: {avg('roc_auc'):.4f}
Average PR AUC: {avg('pr_auc'):.4f}
Learning Rate: {learning_rate}
"""

    # Save as a text file
    with open(os.path.join(average_dir, "average_report.txt"), "w") as f:
        f.write(average_report)

    print("\nAverage results saved to:", os.path.join(average_dir, "average_report.txt"))

if __name__ == "__main__":
    main()