# Deep Database: Advanced Relational Data Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyG](https://img.shields.io/badge/PyG-2.4+-orange.svg)
[![Linting - flake8](https://img.shields.io/badge/code%20style-flake8-blue)](https://flake8.pycqa.org/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-green)
![Logo](./logo.jpg)
> A comprehensive framework for deep learning on relational databases, featuring database research work: RAM, AIDA, LEVA, ARDA, Qzero and rich baseline implementations.



## Why It Matters

Deep Database helps organizations turn complex relational data into responsible, evidence-based decisions. Municipal teams can surface service gaps earlier, financial analysts can monitor systemic risk without sacrificing compliance, and scientists can fuse heterogeneous datasets to accelerate discovery. By shipping reproducible preprocessing, transparent model baselines, and retrieval-augmented reasoning, the project lowers the barrier for teams whose models must stand up to real-world scrutiny.

## Overview

Deep Database provides a unified platform for applying deep learning and machine learning techniques to relational data analytics. A shared set of dataset abstractions, preprocessing utilities, and task definitions keeps experiments consistent across graph, tabular, and retrieval-augmented approaches.

## Key Features

- **Multiple Modeling Paradigms**: RAM, AIDA, LEVA, ARDA, and additional research prototypes.
- **Consistent Data Abstractions**: Dataset, table, and task classes with self-registration for new workloads.
- **Graph & Tabular Builders**: Convert relational data into heterogeneous graphs or flattened tables.
- **Retrieval Integration**: Document/token pipelines and BM25 retrieval for augmented modeling.
- **Extensive Baselines**: Classic ML (CatBoost, LightGBM), neural tabular models, and graph networks.
- **Reproducible Evaluation**: Scripts and configs for end-to-end benchmarking.

## Setup

This project uses **Conda** for environment and dependency management, configured via [`environment.yml`](./environment.yml).

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Python 3.9+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd embedding_fusion
   ```

2. **Create the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate deepdb
   ```

### Main Dependencies

- **relbench** — relational benchmark datasets
- **torch**, **torch_geometric (PyG)** — deep learning and GNN toolkits
- **torch_frame** — tabular modeling utilities for PyTorch
- **bm25s** — BM25 retrieval implementation
- **pandas**, **numpy** — data processing

## Project Structure

Deep Database is organized so that dataset handling, task logic, and modeling remain loosely coupled:


```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DEEP DATABASE FRAMEWORK                                │
│                     Advanced Relational Data Analytics                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   Raw Data   │
                                    │ (Relational) │
                                    └──────┬───────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                              │
           ┌────────▼─────────┐                          ┌────────▼─────────┐
           │ DATASET REGISTRY │                          │   TASK LAYER     │
           │   (utils/data)   │                          │  (utils/task)    │
           ├──────────────────┤                          ├──────────────────┤
           │ • Database       │◄────────────────────────►│ • Objectives     │
           │ • Dataset        │                          │ • Metrics        │
           │ • Table          │                          │ • Splits         │
           │ • table_data     │                          │ • Classification │
           │   adapters       │                          │ • Regression     │
           └────────┬─────────┘                          └──────────────────┘
                    │
        ┌───────────┴───────────────────┐
        │                               │
┌───────▼──────────┐         ┌──────────▼────────────┐
│ RELATIONAL       │         │   ADAPTIVE            │
│ BUILDERS         │         │   PREPROCESSING       │
│ (utils/builder)  │         │                       │
├──────────────────┤         ├───────────────────────┤
│ • Schema → Graph │         │ • utils/preprocess    │
│ • Homogeneous    │         │   (column analysis)   │
│ • Heterogeneous  │         │ • utils/document      │
│ • Graph Metadata │         │   (row → text)        │
└────────┬─────────┘         │ • utils/tokenize      │
         │                   │   (retrieval prep)    │
         │                   └──────────┬────────────┘
         │                              │
         └──────────┬───────────────────┘
                    │
         ┌──────────▼──────────────────────────────────────┐
         │                                                 │
         │              MODEL LIBRARY (model/)             │
         │                                                 │
         ├─────────────────────────────────────────────────┤
         │                                                 │
         │  ┌─────────────┐  ┌──────────────┐              │
         │  │   SHARED    │  │   TABULAR    │              │
         │  │             │  │              │              │
         │  │ • base      │  │ • MLP        │              │
         │  │ • utils     │  │ • ResNet     │              │
         │  └─────────────┘  │ • TabM       │              │
         │                   │ • FT-Trans   │              │
         │  ┌─────────────┐  │ • ARMNet     │              │
         │  │   GRAPH     │  └──────────────┘              │
         │  │   MODELS    │                                │
         │  │             │  ┌──────────────┐              │
         │  │ • GCN       │  │ CONTRASTIVE  │              │
         │  │ • HGT       │  │              │              │ 
         │  │ • GAT       │  │ • BGRL       │              │
         │  │ • SAGE      │  │ • DGI        │              │
         │  └─────────────┘  │ • GraphCL    │              │
         │                   └──────────────┘              │
         └──────────────────────┬──────────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    │                        │
         ┌──────────▼──────────┐  ┌──────────▼──────────────┐
         │   COMMAND           │  │   ARTEFACT STORES       │
         │   ENTRYPOINTS       │  │      (data/)            │
         │                     │  │                         │
         │ • cmds/             │  │ • Generated Tables      │
         │   - Data Gen        │  │ • TensorFrames          │
         │   - Baselines       │  │ • Cached Features       │
         │   - Diagnostics     │  │ • Model Checkpoints     │
         │                     │  │                         │
         │ • exe/              │  │                         │
         │   - Shell Scripts   │  │                         │
         │   - End-to-End      │  │                         │
         └─────────────────────┘  └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXTENSIONS                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │   RAM    │  │   LEVA   │  │  tabICL  │  │  tabPFN  │  │  Qzero   │           │
│  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤           │
│  │Retrieval │  │ Boosted  │  │ In-Ctx   │  │  Prior   │  │  Neural  │           │
│  │Augmented │  │Relational│  │ Learning │  │ Fitted   │  │  Arch    │           │
│  │ Modeling │  │ Learning │  │          │  │ Network  │  │  Search  │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                                 │
│                              AIDA: Advanced Interface for Data Analytics        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            QUALITY GATES                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  test/   │  │  scripts/    │  │   static/    │  │   webapp/    │             │
│  ├──────────┤  ├──────────────┤  ├──────────────┤  ├──────────────┤             │
│  │   Unit   │  │  Notebooks   │  │   Assets     │  │   Demos      │             │
│  │ Coverage │  │ Exploration  │  │ Figures      │  │ Interactive  │             │
│  └──────────┘  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    DATA FLOW
                                    ─────────
    Raw Relational Data → Dataset Registry → Task Layer
                              ↓
                    Relational Builders + Preprocessing
                              ↓
                    Graph/Tabular Representations
                              ↓
                        Model Library
                              ↓
                    Predictions & Evaluations
                              ↓
                        Artefact Stores

                                KEY FEATURES
                                ────────────
    • Multi-Paradigm: Graph, Tabular, Retrieval-Augmented
    • Consistent Abstractions: Unified dataset/task interfaces
    • Extensive Baselines: Classic ML + Neural Networks
    • Reproducible: End-to-end scripts and configs
    • Extensible: Plugin architecture for new methods
```


### Core Components

1. **Dataset Registry** (`utils/data`)
   - Defines abstractions: Database, Dataset, Table
   - table_data adapters materialize relational sources
   - Self-registration for new workloads

2. **Task Layer** (`utils/task`)
   - Standardizes objectives, metrics, splits
   - Supports classification, regression, future plugins
   - Consistent experiment interface

3. **Relational Builders** (`utils/builder`)
   - Schema → Graph transformations
   - Homogeneous & heterogeneous graph support
   - Reuses registry metadata

4. **Adaptive Preprocessing** (`utils/preprocess`, `utils/document`, `utils/tokenize`)
   - Column type and role inspection
   - Row → text translation for retrieval
   - Language model interface preparation

5. **Model Library** (`model/`)
   - Shared modules: base, utils
   - Graph: GCN, HGT, GAT, SAGE
   - Contrastive: BGRL, DGI, GraphCL
   - Tabular: MLP, ResNet, TabM, FT-Transformer, ARMNet

### Supporting Infrastructure

6. **Command Entrypoints** (`cmds/`, `exe/`)
   - Python frontends for data generation, baselines, diagnostics
   - Shell scripts for end-to-end workflows

7. **Artefact Stores** (`data/`)
   - Persists generated tables and tensorframes
   - Cached features and model checkpoints

### Extensions & Quality

8. **Extensions**
   - RAM: Retrieval-Augmented Modeling
   - LEVA: Boosted Relational Learning
   - tabICL, tabPFN: Tabular Foundation Models
   - Qzero: Neural Architecture Search
   - AIDA: Advanced Interface for Data Analytics

9. **Quality Gates**
   - test/: Unit coverage
   - scripts/: Notebooks for exploration
   - static/, webapp/: Demos and visualizationsts/` and assets in `static/`/`webapp/` support exploration and demos without altering the production workflow.

## Quick Start

```bash
# Generate baseline tabular data
bash exe/generate_table_data.sh


# Launch neural tabular baselines (MLP, ResNet, FT-Transformer)
python cmds/dnn_baseline_table_data.py --help

# Launch classic ML baselines (CatBoost, LightGBM)
python cmds/ml_baseline.py --help
```




## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{deepdatabase2025,
  title={Deep Database: Advanced Relational Data Analytics},
  author={Your Team},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue in the repository.
