# ImpReSS: Designing and Evaluating a Lightweight Implicit Recommender System in Conversational Support Agents

[![Conference](https://img.shields.io/badge/IUI'26-Accepted-blue)](https://iui.acm.org/2026/)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3742413.3789151-blue)](https://dl.acm.org/doi/10.1145/3742413.3789151)
[![License](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

This repository contains the official implementation and datasets for the paper [**"ImpReSS: Designing and Evaluating a Lightweight Implicit Recommender System in Conversational Support Agents"**](https://dl.acm.org/doi/10.1145/3742413.3789151), accepted at the 31st International Conference on Intelligent User Interfaces (**IUI '26**).

## 📖 Overview

**ImpReSS** (Implicit Recommender System for Support Conversations) is a lightweight recommendation architecture designed for customer support agents. Unlike traditional Conversational Recommender Systems (CRSs) that assume a user wants to buy something, ImpReSS infers **Solution Product Categories (SPCs)** directly from the problem-solving context without explicit purchasing intent.

The system uses a two-stage approach:
1.  **LLM-based Diagnosis:** Distills support conversations into concise diagnostic queries using small, open-source LLMs (e.g., Llama-3.2-1B).
2.  **Multi-View Bi-Encoder Retrieval:** Matches queries against a product catalog using a fine-tuned sentence transformer with multiple views (Intrinsic/Generated features + Extrinsic/Web features) and Reciprocal Rank Fusion (RRF).

## 📂 Repository Structure

```text
impress_repo
 ├── config/               # Configuration files (prompts, hyperparameters)
     ├── config.yaml
     └── prompts.yaml
 ├── data/                 # Dataset folders
 │   ├── ds_ct/            
 │   ├── ds_is/            
 │   └── ds_mu/            
 ├── src/                  # Core source code
 │   ├── data.py
 │   ├── model.py          
 │   ├── llm_handler.py    
 │   ├── evaluation.py     
 │   └── query_generator.py
 ├── scripts/              # Helper scripts
     └── train.py
 ├── main.py               # Main entry point for training/evaluation
 └── results/              # Outputs and metrics
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/omri898/ImpReSS.git
cd ImpReSS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Environment Setup:**
Create a `.env` file in the root directory if you need to set specific API keys (e.g., for Tavily/LangSmith), though the core open-source pipeline relies on local HuggingFace models.

## 📊 Datasets

This repository includes the three datasets introduced in the paper. Each dataset folder contains `train_data.csv`, `validation_data.csv`, `test_data.csv`, and an SPC catalog `spc_data.csv`.

| Code Name | Paper Notation | Domain | Source |
| --- | --- | --- | --- |
| **ds_ct** | $DS^{CT}$ | Cybersecurity | User Study|
| **ds_is** | $DS^{IS}$ | InfoSec | Stack Exchange |
| **ds_mu** | $DS^{MU}$ | Music | Stack Exchange |

## 🚀 Usage

The `main.py` script is the single entry point for all experiments.

### 1. Training

To train the ImpReSS model on a specific dataset:

```bash
python main.py --dataset ds_is --mode train --batch-size 16 --epochs 5
```

* **--dataset**: `ds_ct`, `ds_is`, or `ds_mu`
* **--mode**: `train`

### 2. Diagnosis Generation & Caching

To efficiently test different hyperparameters and training configurations, we suggest generating diagnoses once and saving them to a cache.

**First Run (Generate & Save):**

```bash
python main.py --dataset ds_is --mode train --save-cache
```

**Subsequent Runs (Load from Cache):**

```bash
python main.py --dataset ds_is --mode train --use-cache
```

### 3. Evaluation

To evaluate a trained model on the test set:

```bash
python main.py --dataset ds_is --mode evaluate --save-cache
```

## 🔧 Command-Line Arguments

The `main.py` script supports the following arguments:

### Required Arguments

| Argument | Choices | Description |
| --- | --- | --- |
| `--dataset` | `ds_ct`, `ds_is`, `ds_mu` | Dataset to use: ds_ct (customer support), ds_is (InfoSec SE), ds_mu (Music SE) |
| `--mode` | `train`, `evaluate` | Operation mode: train a new model or evaluate an existing one |

### Path Configuration

| Argument | Default | Description |
| --- | --- | --- |
| `--data-dir` | `data` | Data directory containing dataset folders |
| `--model-dir` | `trained_models` | Model save/load directory |
| `--model-path` | `None` | Specific model path for evaluation (overrides `--model-dir`) |
| `--results-dir` | `results` | Results output directory |
| `--cache-dir` | `cache` | Directory for diagnosis cache files |

### Training Hyperparameters

| Argument | Default | Description |
| --- | --- | --- |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `8` | Training batch size |
| `--learning-rate` | `1e-6` | Learning rate for optimizer |
| `--early-stopping-patience` | `5` | Early stopping patience in epochs |
| `--num-negatives` | `4` | Number of negatives per positive sample |

### Model Configuration

| Argument | Default | Description |
| --- | --- | --- |
| `--embedding-model` | `intfloat/multilingual-e5-large-instruct` | HuggingFace embedding model ID for bi-encoder |
| `--top-k` | `1 3 5` | Top-k values for evaluation metrics (space-separated) |

### LLM Configuration (Diagnosis Generation)

| Argument | Default | Description |
| --- | --- | --- |
| `--llm-model` | `llama1b` | LLM model for diagnosis generation (see Supported Models) |
| `--temperature` | `0.3` | LLM temperature for diagnosis generation |
| `--max-tokens` | `512` | LLM max tokens to generate for diagnosis |

### Caching Options

| Argument | Default | Description |
| --- | --- | --- |
| `--use-cache` | `False` | Load diagnoses from cache if available |
| `--save-cache` | `False` | Save generated diagnoses to cache for future use |

**Note:** Caching is highly recommended to speed up experiments when testing different hyperparameters or training configurations.

### General Options

| Argument | Default | Description |
| --- | --- | --- |
| `--verbose` / `-v` | `False` | Enable verbose logging |
| `--log-file` | `None` | Log file path (if not set, logs to stdout only) |

### Examples with Arguments

```bash
# Train with custom hyperparameters and caching
python main.py --dataset ds_is --mode train \
    --llm-model llama3b --temperature 0.3 \
    --epochs 5 --batch-size 16 --learning-rate 2e-6 \
    --use-cache --save-cache

# Evaluate with specific top-k values
python main.py --dataset ds_ct --mode evaluate \
    --llm-model llama1b --top-k 1 2 3 4 5 \
    --use-cache

# Train with more negatives and verbose logging
python main.py --dataset ds_mu --mode train \
    --num-negatives 8 \
    --verbose --log-file training.log
```

## 🧠 Supported Models

The codebase supports the specific open-source models analyzed in the paper's ablation study. These LLMs generate diagnostic queries from support conversations during **both training and evaluation**. When running experiments, ensure you use the same LLM configuration for both phases to maintain consistency.

| Key | Model | Parameters |
| --- | --- | --- |
| `llama1b` | `meta-llama/Llama-3.2-1B-Instruct` | 1B |
| `llama3b` | `meta-llama/Llama-3.2-3B-Instruct` | 3B |
| `llama8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8B |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` | 7B |
| `qwen1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |

## 📜 Citation

If you use ImpReSS in your research, please cite our IUI '26 paper:

```bibtex
@inproceedings{impress_iui26,
  author = {Omri Haller, Yair Meidan, ...},
  title = {ImpReSS: Designing and Evaluating a Lightweight Implicit Recommender System in Conversational Support Agents},
  booktitle = {31st International Conference on Intelligent User Interfaces (IUI '26)},
  year = {2026},
  location = {Paphos, Cyprus},
  publisher = {ACM},
  doi = {10.1145/3742413.3789151}
}
```

## 📄 License

This dataset and code are released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.
