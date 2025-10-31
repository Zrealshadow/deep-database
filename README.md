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

- **Dataset Registry** — `utils/data` defines `Database`, `Dataset`, and `Table` classes plus the `table_data` adapters that materialize relational sources into task-aligned tables.
- **Task Layer** — `utils/task` standardizes objectives, metrics, and splits, allowing experiments to pivot between regression, classification, and future task plugins.
- **Relational Builders** — `utils/builder` transforms schemas into homogeneous or heterogeneous graphs while reusing the same registry metadata used for tabular views.
- **Adaptive Preprocessing** — `utils/preprocess` inspects column types and roles; `utils/document` and `utils/tokenize` translate rows into textual artefacts for retrieval and language-model interfaces.
- **Model Library** — `model/` houses shared modules (`model/base`, `model/utils`), graph networks (`model/gcn`, `model/hgt`, `model/gat`), contrastive learners (`model/bgrl`, `model/dgi`, `model/graphcl`), and tabular architectures (`model/tabular`).
- **Command Entrypoints** — `cmds/` provides Python front-ends for data generation, baselines, and diagnostics; `exe/` collects repeatable shell scripts for end-to-end workflows.
- **Artefact Stores** — `data/` persist generated tables and tensorframes.
- **Extensions** — directories like `ram/`, `leva/`, `tabICL/`, `tabPFN/`, and `qzero/` demonstrate how the shared core supports retrieval augmentation, boosted relational learning, tabular foundation models, and neural architecture search.
- **Quality Gates** — `test/` houses unit coverage; notebooks under `scripts/` and assets in `static/`/`webapp/` support exploration and demos without altering the production workflow.

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
