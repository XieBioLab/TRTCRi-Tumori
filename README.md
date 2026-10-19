# TRTCRi and Tumori

This repository contains the training and inference code for TRTCRi and Tumori.

## Contents

- `src/train_trtcri.py`: TRTCRi training with five-fold cross-validation.
- `src/train_tumori.py`: Tumori training with a stratified train/validation split.
- `src/predict_trtcri.py`: TRTCRi inference for a single model directory or a `fold_1` to `fold_5` ensemble.
- `src/predict_tumori.py`: Tumori inference and optional TopK weighted sample scoring when `reads_A` and `reads_B` are present.
- `src/common.py`: shared data loading, sequence construction, metrics, and inference utilities.
- `src/modeling_tcr.py`: ESM-2 with CLS/mean/max pooling, projection and three dropout branches sharing one classifier.
- `src/training_utils.py`: differential encoder/head learning rates, label smoothing and validation-F1 checkpoint selection.
- `src/search_hyperparameters.py`: development grid search on a fixed stratified 80:20 split, using fresh ESM-2 initialization for every trial.
- `datasets/`: updated TRTCRi and Tumori training sets, 69 supplementary validation sample files, and a [dataset guide](datasets/README.md).
- `examples/`: example inputs for inference.

Traditional machine-learning baselines, external model comparisons, intermediate result folders, and local model-weight files are not included in this code folder.

## Model weights

Model weights are not included in this code repository. Empty folders are provided for local model files:

```text
model_weights/
  TRTCRi/
  Tumori/
```

Download the trained weights from Zenodo:

https://zenodo.org/records/17313087

After downloading and extracting the weight files, place them under `model_weights/` with this structure:

```text
model_weights/
  TRTCRi/
    fold_1/
      config.json
      model.safetensors
      special_tokens_map.json
      tokenizer_config.json
      vocab.txt
    fold_2/
      config.json
      model.safetensors
      special_tokens_map.json
      tokenizer_config.json
      vocab.txt
    fold_3/
      config.json
      model.safetensors
      special_tokens_map.json
      tokenizer_config.json
      vocab.txt
    fold_4/
      config.json
      model.safetensors
      special_tokens_map.json
      tokenizer_config.json
      vocab.txt
    fold_5/
      config.json
      model.safetensors
      special_tokens_map.json
      tokenizer_config.json
      vocab.txt
  Tumori/
    config.json
    model.safetensors
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
```

If you use the separately prepared folder `TRTCRi_Tumori_model_files`, copy its contents like this:

```bash
cp -r TRTCRi_Tumori_model_files/TRTCRi/* TRTCRi_Tumori/model_weights/TRTCRi/
cp -r TRTCRi_Tumori_model_files/Tumori/* TRTCRi_Tumori/model_weights/Tumori/
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training scripts default to `facebook/esm2_t33_650M_UR50D` as the base model. If you use a local checkpoint, pass it with `--base_model /path/to/local/model`.

The supported Transformers API is version 4 (`>=4.45,<5`). Both models use a maximum tokenized input length of 320 by default, including boundary tokens. The encoder is fine-tuned end to end. Both training entry points initialize independently from the original ESM-2 encoder/MLM; a fine-tuned TRTCRi/Tumori checkpoint is rejected as `--base_model`.

## Data format

Training and inference tables must contain these TCR columns:

```text
TRAV, CDR3a, TRAJ, TRBV, CDR3b, TRBJ
```

Training tables must also contain:

```text
Label
```

Tumori sample-level scoring is computed only when the input also contains `reads_A` and `reads_B`.

## Train

Train TRTCRi:

```bash
python src/train_trtcri.py \
  --train_file datasets/train_TRTCRi.xlsx \
  --output_root modelresult/TRTCRi \
  --fp16
```

Train Tumori:

```bash
python src/train_tumori.py \
  --train_file datasets/train_Tumori.csv \
  --output_root modelresult/Tumori \
  --fp16
```

Remove `--fp16` when training on CPU or on a GPU without reliable FP16 support.

### Classification head and training settings

The model concatenates CLS, masked mean and masked max representations into a `3H` vector, then applies `Linear(3H,H) -> GELU -> LayerNorm`. Mean/max pooling excludes padding, CLS and EOS. Three independent dropout masks feed the same linear classifier. The three logits are averaged before smoothed cross-entropy and probability calculation. Dropout is disabled for deterministic evaluation; it is not Monte Carlo inference.

`--learning_rate` controls the ESM-2 encoder. `--head_learning_rate` separately controls projection and classification parameters. AdamW uses no weight decay on biases and normalization scales. The cosine schedule applies to both parameter groups. The highest validation F1 selects the checkpoint for both TRTCRi and Tumori, using the same `--threshold` as the reported metrics.

| Setting | TRTCRi default | Tumori default |
| --- | --- | --- |
| Encoder LR | `3e-5` | `4.5e-5` |
| Head LR | `3e-4` | `4.5e-4` |
| Dropout branches | `0.2 0.2 0.2` | `0.1 0.1 0.1` |
| Label smoothing | `0.1` | `0.1` |
| Epochs / batch size / accumulation | `10 / 3 / 4` | `5 / 2 / 4` |

The manuscript specifies the architecture, differential learning rates and label smoothing, but does not provide every numerical head setting. The head-LR multiplier of 10, GELU and smoothing of 0.1 are explicit configurable implementation defaults, not recovered historical hyperparameters. Use the original run settings when reproducing a particular experiment. For example:

```bash
python src/train_trtcri.py \
  --learning_rate 3e-5 --head_learning_rate 3e-4 \
  --dropout_rates 0.2 0.2 0.2 --label_smoothing 0.1 --fp16
```

TRTCRi uses stratified sequence-record fivefold cross-validation; Tumori uses a stratified train/validation split. These defaults do not perform donor/source/similarity-group held-out evaluation. Input class proportions are preserved without automatic resampling. Each run writes `run_config.json`, row-level split assignments, validation predictions and metrics. TRTCRi also writes across-fold means and sample SDs (`ddof=1`); the SD is not a confidence interval. Fold models start afresh from ESM-2 with newly initialized heads.

`--save_total_limit` defaults to 2 retained trainer checkpoints per fold, preserving the selected best checkpoint. Final model files are saved separately. GPU allocations are released between folds to avoid carrying unused models into the next fold.

The optional search entry point requires explicit grids for encoder LR, head LR and dropout. Example values below illustrate use and do not reconstruct the manuscript's original search space:

```bash
python src/search_hyperparameters.py \
  --encoder_learning_rates 1e-5 3e-5 \
  --head_learning_rates 1e-4 3e-4 \
  --classifier_dropouts 0.1 0.2 --fp16
```

It saves the fixed development split, all trial settings, validation F1 and the selected parameter combination. Pass the selected numeric settings to formal training; do not use a search-trial checkpoint as the base model. The development search is separate from fivefold evaluation and is not nested cross-validation.

This code update implements the described head for future training. It does not retrain, replace or certify compatibility of the weights previously deposited on Zenodo, and it does not regenerate the paper's performance numbers. Newly trained checkpoints record `tcr_head_type`, branch probabilities and smoothing in `config.json`; repository inference scripts load that head explicitly. Legacy default-ESM classifiers remain supported when their config has no custom-head marker.

## Predict

Run TRTCRi:

```bash
python src/predict_trtcri.py \
  --input_file examples/TRTCRi_test.xlsx \
  --model_dir model_weights/TRTCRi \
  --output_dir outputs/TRTCRi
```

Run Tumori:

```bash
python src/predict_tumori.py \
  --input_file examples/melanoma-patients.csv \
  --model_dir model_weights/Tumori \
  --output_dir outputs/Tumori
```

All scripts preserve the original input columns and append model probability and prediction columns.

Tumori scoring defaults to `K = max(1, ceil(0.05*N))`, with stable input-order tie breaking, weights `(reads_A + reads_B) / total_reads`, and `score = 1 - exp(-31.92*S)`. The selected weights are not renormalized, and the score is not rounded down. Each input file should already contain the final valid paired clonotypes for a single sample. Optional `--top_k`, `--top_fraction` and `--scale` overrides support explicitly configured analyses.

## Offline verification

```bash
python -m unittest discover -s tests -v
```

Tests use tiny random ESM configurations on CPU and cover masked pooling, shared dropout logits, smoothed-loss gradients, optimizer groups, pretrained initialization, checkpoint save/load, legacy loading and fixed-parameter Top-5% scoring. They do not download the 650M checkpoint or estimate biological performance.
