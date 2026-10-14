# Training and supplementary validation data

The files in this directory are the author-supplied training datasets and supplementary peripheral blood validation inputs. Their contents are preserved without additional filtering or deduplication during upload preparation.

## Training datasets

| File | Total records | Positive (`Label = 1`) | Negative (`Label = 0`) | Positive-to-negative ratio |
| --- | ---: | ---: | ---: | --- |
| `train_TRTCRi.xlsx` | 3,616 | 1,808 | 1,808 | 1:1 |
| `train_Tumori.csv` | 20,460 | 1,808 | 18,652 | Approximately 1:10 |

Both files contain the six paired-chain input fields `TRAV`, `CDR3a`, `TRAJ`, `TRBV`, `CDR3b`, and `TRBJ`, plus `Label`. Column order differs between the files; the training loader selects fields by name. `train_Tumori.csv` replaces the previously published `train_Tumori.xlsx`; the README training command and training-script default use the CSV file.

## Supplementary validation inputs

`validation/` contains 69 sample-level CSV files. Each row is a paired TCR input record; each file corresponds to one sample in the supplementary evaluation.

| Source accession | Group | Sample files |
| --- | --- | ---: |
| [GSE270387](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270387) | Prostate-cancer patients | 37 |
| [GSE270387](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270387) | Healthy controls | 16 |
| [GSE161918](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161918) | Healthy controls | 13 |
| [GSE267639](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267639) | Healthy controls | 3 |
| **Total** | **37 tumor samples and 32 healthy controls** | **69** |

These files contain the six paired-chain input fields, the read-support fields `reads_A` and `reads_B`, and the supplied `frequency` and `Tumori_probability` columns. Their filenames identify the source accession, study group, and supplied sample index. The probabilities are stored per TCR record; these files are not a table of patient-level Tumori scores or confidence intervals. The original development-cohort datasets are not part of this supplementary directory.

## File manifest

[`manifest.csv`](manifest.csv) lists all 71 supplied data files, their roles, validation-source groups, record counts, byte sizes, and SHA-256 checksums. `records` counts data rows, excluding the header. Validation group annotations are taken from the author-supplied filenames. SHA-256 values can be used to verify that downloaded file contents match this upload.
