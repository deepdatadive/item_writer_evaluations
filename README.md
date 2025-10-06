# Item Writer Evaluations

A comprehensive repository for evaluating AI-assisted item writing and test development using data analysis and psychometric methods.

## Overview

This project provides tools, scripts, and analyses for evaluating AI-generated test items and content. It includes:

- Data analysis and visualization of test item quality metrics
- Psychometric evaluation of AI-generated items
- Jupyter notebooks showcasing evaluation methodologies
- Reproducible analysis workflows
- Comparative studies between AI-assisted and traditional item writing

## Getting Started

### Prerequisites

- Python 3.8+
- pip and virtualenv

### Installation

1. Clone the repository:
```bash
git clone git@github.com:deepdatadive/item_writer_evaluations.git
cd item_writer_evaluations
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Development

If you want to add new dependencies, edit `requirements.in` and then compile:
```bash
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

## Project Structure

This repository is organized by **experiment**, with each experiment containing its own data, analyses, and results. See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for detailed documentation.

```
item_writer_evaluations/
├── experiments/           # Experimental studies
│   └── exp00X_name/      # Each experiment is self-contained
│       ├── data/         # Raw and processed data
│       ├── notebooks/    # Analysis notebooks
│       ├── results/      # Figures and tables
│       └── README.md     # Experiment documentation
├── shared/               # Shared utilities and templates
│   ├── utils/           # Reusable Python functions
│   └── templates/       # Notebook templates
├── docs/                # Documentation
│   ├── data_schema.md   # Data format specifications  
│   └── analysis_guidelines.md  # Best practices
└── reports/             # Cross-experiment analyses
```

### Quick Start: Creating a New Experiment

```bash
# 1. Create experiment directory structure
mkdir -p experiments/exp00X_experiment_name/{data/{raw,processed},notebooks,results/{figures,tables}}

# 2. Copy templates
cp experiments/exp001_baseline_comparison/README.md experiments/exp00X_experiment_name/
cp experiments/exp001_baseline_comparison/data/metadata.json experiments/exp00X_experiment_name/data/
cp shared/templates/analysis_template.ipynb experiments/exp00X_experiment_name/notebooks/01_analysis.ipynb

# 3. Add your data to data/raw/
# 4. Update metadata.json with experiment details
# 5. Run your analysis notebooks
```

## Key Tools and Libraries

This project uses:

- **Data Analysis**: pandas, numpy, scipy, statsmodels
- **Visualization**: matplotlib, seaborn, plotly
- **Psychometrics**: factor-analyzer, pingouin, scikit-learn
- **Notebooks**: Marimo (reactive Python notebooks), Jupyter
- **Development**: pytest, black, flake8

### Using Marimo (Recommended)

This project now supports [Marimo](https://marimo.io), a modern reactive Python notebook:

```bash
# Install marimo (already in requirements.txt)
pip install marimo

# Run a notebook interactively
cd experiments/basic_vs_ai_assist/notebooks
marimo edit basic_vs_ai_assist.py

# Or run as a web app
marimo run basic_vs_ai_assist.py
```

**Why Marimo?**
- ✅ Reactive execution - cells automatically update when dependencies change
- ✅ No hidden state - reproducibility built-in
- ✅ Better UI with interactive widgets
- ✅ Works as both notebook and deployable app
- ✅ Pure Python files - better for version control

**Converting Jupyter to Marimo:**
```bash
marimo convert notebook.ipynb -o notebook.py
```

## Using Shared Utilities

The `shared/utils/` directory contains reusable functions for common tasks:

```python
import sys
sys.path.append('../../shared')  # Adjust path based on your location
from utils import (
    load_experiment_data,
    validate_long_format,
    calculate_item_stats,
    calculate_reliability,
    setup_plotting_style,
    save_figure
)

# Load experiment data
df, metadata = load_experiment_data('experiments/exp001_baseline_comparison')

# Validate data format
is_valid, issues = validate_long_format(df)

# Calculate psychometric statistics
item_stats = calculate_item_stats(df, group_col='condition')
reliability = calculate_reliability(df)
```

## Data Format

All experimental data should be in **long format** (one row per response):

| participant_id | condition | item_id | response | correct | response_time |
|---------------|-----------|---------|----------|---------|---------------|
| P001 | control | Q001 | A | 1 | 12.5 |
| P001 | control | Q002 | C | 0 | 18.3 |
| P002 | treatment | Q001 | A | 1 | 10.2 |

See [`docs/data_schema.md`](docs/data_schema.md) for complete specifications.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add License Information]

## Contact

[Add Contact Information]
