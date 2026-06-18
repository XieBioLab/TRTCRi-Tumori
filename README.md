# TRTCRi and Tumori

This repository contains the training and inference code for TRTCRi and Tumori.

## Contents

- `src/train_trtcri.py`: TRTCRi training with five-fold cross-validation.
- `src/train_tumori.py`: Tumori training with a stratified train/validation split.
- `src/predict_trtcri.py`: TRTCRi inference for a single model directory or a `fold_1` to `fold_5` ensemble.
- `src/predict_tumori.py`: Tumori inference and optional TopK weighted sample scoring when `reads_A` and `reads_B` are present.
- `src/common.py`: shared data loading, sequence construction, metrics, and inference utilities.
- `datasets/`: TRTCRi and Tumori training sets.
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
  --train_file datasets/train_Tumori.xlsx \
  --output_root modelresult/Tumori \
  --fp16
```

Remove `--fp16` when training on CPU or on a GPU without reliable FP16 support.

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
