# MetaAI: Predictive Design of Cellular Sequence–Metabolite Landscapes

**MetaAI** is a structure-aware deep learning framework for reconstructing and navigating the **cellular sequence–metabolite landscape**.  
This repository provides **training**, **inference**, **epistasis prediction**, and **plotting** code for two case studies:
- **4CL**
- **CHS**

> The code is organized as two self-contained pipelines under `4CL/` and `CHS/`.

---

## Key Features
- **Structure-aware fitness prediction**: combines protein language model embeddings (ESM2 / ESM++) with geometric features (SPIRED-Fitness embedding).
- **Rank-optimized training objective**: Spearman (soft ranking) loss for robust ranking in screening/design tasks.
- **High-order mutation modeling**: supports multi-mutation fitness prediction and epistasis analysis.

---

## Installation

### 1) Create environment
```bash
conda create -n metaai python=3.10 -y
conda activate metaai
````

### 2) Install dependencies

```bash
pip install torch pandas numpy scipy biopython tqdm click transformers iterative-stratification matplotlib statsmodels openpyxl
```

---

## Repository Structure 

Each of `4CL/` and `CHS/` contains:

* `data/` : datasets (`*.xlsx`)
* `features/wt/` : WT fasta + precomputed embeddings (`*.pt`)
* `01_model_training/` : training code + pretrained checkpoint
* `02_inference/` : candidate generation + batch prediction
* `03_plot/` : epistasis prediction + plotting scripts
* `benchmark/` : correlation metric 

---

## Quick Start

### 0) Choose a target

```bash
cd 4CL
# or
cd CHS
```

### 1) Generate candidate multi-mutation sets (2/3/4 sites)

```bash
cd 02_inference
python generate_unpredicted_muts_csv.py --mut_counts 2
python generate_unpredicted_muts_csv.py --mut_counts 3
python generate_unpredicted_muts_csv.py --mut_counts 4
```

This creates:

* `sorted_mut_counts_2.csv`
* `sorted_mut_counts_3.csv`
* `sorted_mut_counts_4.csv`

> By default, the script enumerates all combinations if feasible, otherwise samples up to **10,000** candidates (edit `max_mutations` in `generate_unpredicted_muts_csv.py` if needed).

### 2) Run inference

```bash
python inference_muts.py --mut_counts 2
python inference_muts.py --mut_counts 3
python inference_muts.py --mut_counts 4
```

Outputs:

* `pred_sorted_mut_counts_{n}.csv` (ranked by predicted fitness)

---

## Generating Embeddings for a New Target 

If you want to adapt MetaAI to a new protein:

1. Replace `features/wt/result.fasta` with your WT sequence.
2. Generate embeddings under `features/wt/` using scripts in `generate_features/`:

   * `generate_esm2_embedding.py` (downloads ESM2 via `torch.hub`)
   * `generate_esmc_embedding.py` (downloads ESM++ via HuggingFace)

---

## License

Apache License 2.0 (see `LICENSE.txt`).

---