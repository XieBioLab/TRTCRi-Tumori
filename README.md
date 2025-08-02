# TRTCRi-Tumori

## overview

**A deep learning framework for antigen-agnostic identification and early detection of tumor-reactive T cell receptors.**

In this project, we propose two complementary deep learning models designed to uncover tumor-immune signatures directly from T cell receptor (TCR) sequences—**TRTCRi** and **Tumori**:

- **TRTCRi** focuses on the *single TCR level*. It predicts whether a given TCR αβ pair is tumor-reactive without requiring prior knowledge of antigen specificity. The model integrates a protein language model (ESM-2) with a custom BiLSTM network to capture both global sequence embeddings and local contextual patterns. TRTCRi achieved strong performance across multiple public datasets (AUC = 0.78, AUPR = 0.81).
- **Tumori** operates at the *repertoire level*, analyzing the entire peripheral blood TCR pool to estimate the proportion of tumor-reactive TCRs. This allows noninvasive early cancer detection and dynamic monitoring of immunotherapy response. Tumori fine-tunes ESM-2 with an attention-based classifier to prioritize informative CDR3 regions, achieving robust results in real-world blood samples from melanoma, colorectal cancer, and Hodgkin lymphoma patients.

Together, TRTCRi and Tumori offer an antigen-agnostic, sequence-based strategy for tumor reactivity analysis and immune surveillance, laying the groundwork for next-generation TCR-centric cancer diagnostics and personalized immunotherapy design.



## install

From Source:

~~~shell
git clone https://github.com/XieBioLab/TRTCRi-Tumori.git
cd TRTCRi-Tumori
~~~

Running the iCanTCR requires python3.6, numpy version 2.3.2, torch version 2.5.1, transformers version 4.47.1, pandas version 2.3.1 and scikit-learn version 1.7.1 to be installed.

If they are not installed on your environment, please first install numpy, pandas, and scikit-learn by running the following command:

~~~shell
 pip install -r requirements.txt
~~~

Next, Please select the most suitable command from the following options to install torch and torchvision based on your terminal’s specific situation:

### Optional: Custom PyTorch Installation

If your system requires a specific CUDA version, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

For example, to install PyTorch 2.5.1 with CUDA 11.8 support:

```bash
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you're using CPU only:

```bash
pip install torch==2.5.1+cpu torchvision==0.17.1+cpu torchaudio==2.5.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Environment Requirements

TRTCRi was trained in the following environment:

```shell
- PyTorch Version: 2.5.1
- Device: CUDA
- GPU: 2 × RTX 4090
```

In theory, our testing code is designed with high compatibility. You can run TRTCRi and Tumori on any modern GPU or even CPU. However, if you plan to train the models with your own data, a more powerful GPU will significantly reduce training time.

---

## Using TRTCRi & Tumori

### TRTCRi

To run **TRTCRi**, you can use the following command. A test file is provided under the `examples` directory. If you wish to use your own data file, ensure it is in `.csv`, `.xlsx`, or `.tsv` format with the following **six columns**:

- `TRAV`, `CDR3a`, `TRAJ`, `TRBV`, `CDR3b`, `TRBJ`

The `V` and `J` fields must be amino acid sequences. Replace `./examples/TRTCRi_test.xlsx` with the path to your own file. If you'd like to specify an output directory, replace `./outputs`. If not specified, default paths will be used.

```bash
python TRTCRi-test.py ./examples/TRTCRi_test.xlsx ./outputs
```

After running the command, TRTCRi will output two files in the specified directory:

```python
Using device: cuda
Number of input sequences: 1021
Saved 268 reactive TCRs to: ./outputs/TRTCRi_test_health/reactive_tcrs.txt
Saved full prediction results (including probabilities) to: ./outputs/TRTCRi_test_health/prediction_results.xlsx
```

- **`reactive_tcrs.txt`**: Contains all TCR sequences predicted to be tumor-reactive (positive).
- **`prediction_results.xlsx`**: Contains prediction probabilities (`probs`) for all input samples.

---

### Tumori

To run **Tumori**, use the command below. A test file is also provided in the `examples` directory. If you wish to use your own data, the file must be in `.csv`, `.xlsx`, or `.tsv` format and contain **eight columns**:

- `TRAV`, `CDR3a`, `TRAJ`, `TRBV`, `CDR3b`, `TRBJ`, `reads_A`, `reads_B`

Here, `V` and `J` must be amino acid sequences. `reads_A` and `reads_B` represent the read counts of the alpha and beta chains during TCR assembly. Depending on your preprocessing tool, the column names may differ, but **they must be renamed** to `reads_A` and `reads_B` before model inference.

Replace `./examples/melanoma-patients.csv` with your own file path. The output directory can also be replaced (`./outputs`), or it will default to the given location.

```bash
python Tumori-test.py ./examples/melanoma-patients.csv ./outputs
```

Once executed, Tumori will output the tumor reactivity score and prediction file in the specified path:

```python
===== Device used for inference: cuda =====

Number of test samples: 215

===== Start predicting =====
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14/14 [00:04<00:00,  3.15it/s]

weighted tumor reactivity score (normalized): 0.82

===== Results saved to: ./outputs/Tumori_melanoma-patients.csv_inference =====
```

#### Batch Testing

If you have **multiple files** to test, we provide a script to process an entire directory. Each file in the folder must follow the same format as required by Tumori.

Replace `./examples` with your folder path and optionally `./outputs` as needed:

```bash
python Tumori-testdir.py ./examples ./outputs
```

---

## Training TRTCRi & Tumori on Your Own Data

To train **TRTCRi** or **Tumori** on your own data, use the following commands:

```bash
# Train TRTCRi (replace with your own training file)
python train_TRTCRi.py --./datasets/train_TRTCRi.xlsx

# Train Tumori (replace with your own training file)
python train_Tumori.py --./datasets/train_Tumori.xlsx
```

Your training data must contain **seven columns**:

- `TRAV`, `CDR3a`, `TRAJ`, `TRBV`, `CDR3b`, `TRBJ`, `Label`

The `Label` column should be `1` for tumor-reactive samples and `0` for negative samples.

