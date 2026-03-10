# MasMffr - Multi-source Data Analysis Framework

This repository contains the implementation of the MasMffr framework for multi-source data analysis and diagnosis tasks.

## Overview

MasMffr is a unified framework that processes multiple data sources (metrics, traces, logs) for system diagnosis including:
- Anomaly Detection (AD)
- Failure Triage (FT) 
- Root Cause Localization (RCL)

## Repository Structure

```
MasMffr/
├── config/              # Configuration files for datasets
│   ├── D1.yaml         # Configuration for dataset D1
│   └── D2.yaml         # Configuration for dataset D2
├── data/               # Data directory (samples, cases, hash_info)
├── models/             # Model implementations
│   ├── diagnosis_tasks/    # Diagnosis task modules
│   └── unified_representation/  # Unified representation learning
├── utils/              # Utility functions
├── res/                # Results directory
└── main.py            # Main execution script
```

## Usage

1. Configure dataset paths and parameters in `config/D1.yaml` or `config/D2.yaml`

2. Run the framework:
```bash
python main.py
```

3. Results will be saved in the `res/{dataset}/` directory:
   - `res.json`: Evaluation results
   - `tmp/`: Temporary results for each task

## Configuration

Key configuration parameters:
- `dataset`: Dataset name (D1 or D2)
- `workflow`: List of diagnosis tasks to run ['AD', 'FT', 'RCL']
- `model_param`: Model hyperparameters
- `downstream_param`: Task-specific parameters

## Requirements

- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- PyYAML
- scipy
- matplotlib

## Citation

If you use this code in your research, please cite the corresponding paper.
