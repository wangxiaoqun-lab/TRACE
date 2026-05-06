# TRACE: Transformer-based Reconstruction of Axial Cell Development

TRACE is a hybrid AI framework for reconstructing spatiotemporal developmental trajectories of GABAergic neurons by integrating transformer-based representation learning with optimal transport–based inference. The model explicitly incorporates both spatial and temporal context to infer developmental relationships across embryonic to adult stages.

This repository accompanies our study on linking embryonic spatial patterning to the organization of cortical GABAergic neurons in the adult brain.

This repository contains two connected components:

1. `training_TRACE.py` and `cell_embedding.py` for TRACE model training and cell embedding extraction.
2. `main_mappings.py` and `sbatch_main_mappings.sh` for OT-based cell mapping between developmental stages.

## Installation

### 1. Prepare the code
Ensure the `code_TRACE` directory is available locally:
```bash
cd code_TRACE
```

### 2. Create an environment

Python 3.11 is recommended.

```bash
conda create -n trace python=3.11
conda activate trace
```

### 3. Install dependencies

The repository does not currently provide a pinned `requirements.txt`, so install the main packages manually:

```bash
pip install numpy pandas scipy scikit-learn tqdm anndata scanpy matplotlib pot torch torchtext
```

If you plan to train on GPU, install a CUDA-compatible PyTorch build first, then optionally install FlashAttention if your environment supports it.

Depending on your local setup, you may also need:

```bash
pip install flash-attn
```

Before running the full pipeline, create the output directories expected by the scripts if they do not already exist:

```bash
mkdir -p model embeddings_result output
```

## Data Requirements

TRACE expects spatial transcriptomics data in `AnnData` (`.h5ad`) format.

### Required matrix and metadata

For training and embedding extraction, the scripts expect:

- `adata.X`: gene expression matrix
- `adata.var_names`: gene symbols
- `adata.obs['coord_x']` and `adata.obs['coord_y']`: raw spatial coordinates
- `adata.obs['batch']`: batch or sample name
- `adata.obs['sample_info']`: sample information
- `adata.obs['CellType']`: cell-type annotation
- `adata.obsm['ccf_l']`: aligned spatial coordinates used for niche construction
- `adata.obs['cellid']`: cell identifier used when exporting embeddings

### File naming conventions used in the current scripts

- `training_TRACE.py` reads files named like `{time_point}_xenium.h5ad`
- `cell_embedding.py` reads files named like `{time_point}_xenium_updated.h5ad`

The dataset paths are hard-coded in the current version of the scripts, so you should update them to match your local directory layout before running.

## Usage

### 1. Train TRACE

Edit the dataset path and time-point configuration in `training_TRACE.py`, then run:

```bash
python training_TRACE.py
```

### 2. Generate cell embeddings

Edit the dataset path, time points, and target slice in `cell_embedding.py`, then run:

```bash
python cell_embedding.py
```

### 3. Run OT-based cell mapping

After embeddings have been generated, run:

```bash
python main_mappings.py <alpha> <slice_name> <time_point_idx>
```

Arguments:

- `alpha`: trade-off between embedding similarity and spatial structure
- `slice_name`: `CGE` or `LMGE`
- `time_point_idx`: index of the stage pair to map


Example:

```bash
python main_mappings.py 0.001 CGE 0
```

Expected output in `output/`:

- `LLM_mapping_P<stage_pair>_slice<slice_name>_alpha<alpha>.csv`

Each CSV contains:

- `index1`: source cell ID
- `index2`: mapped target cell ID
- `pi_value`: transport score

### 4. Submit mapping jobs with SLURM
Submit with:

```bash
sbatch sbatch_main_mappings.sh
```

## Citation
