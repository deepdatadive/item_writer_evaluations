# Project Structure

This document describes the organization of the item_writer_evaluations repository.

## Directory Structure

```
item_writer_evaluations/
│
├── experiments/                 # All experimental data and analyses
│   └── exp001_baseline_comparison/
│       ├── data/
│       │   ├── raw/            # Original, unmodified data files
│       │   ├── processed/      # Cleaned and transformed data
│       │   └── metadata.json   # Experiment metadata and documentation
│       ├── notebooks/          # Jupyter notebooks for this experiment
│       ├── results/
│       │   ├── figures/        # Plots and visualizations
│       │   └── tables/         # Statistical tables and summaries
│       └── README.md           # Experiment-specific documentation
│
├── shared/                      # Shared resources across experiments
│   ├── utils/                  # Reusable Python utilities
│   │   ├── __init__.py
│   │   ├── data_loading.py    # Data loading and validation functions
│   │   ├── psychometrics.py   # Psychometric analysis functions
│   │   └── plotting.py        # Plotting utilities and styles
│   └── templates/              # Template notebooks and files
│       └── analysis_template.ipynb
│
├── reports/                     # Cross-experiment analyses and reports
│   └── meta_analysis/
│
├── docs/                        # Documentation
│   ├── data_schema.md          # Data format specifications
│   └── analysis_guidelines.md  # Analysis best practices
│
├── requirements.in              # Top-level dependencies (unpinned)
├── requirements.txt             # Pinned dependencies (generated)
├── .gitignore                   # Git ignore patterns
├── README.md                    # Project overview
└── PROJECT_STRUCTURE.md         # This file

```

## Workflow for New Experiments

### 1. Create Experiment Directory

```bash
mkdir -p experiments/exp00X_experiment_name/{data/{raw,processed},notebooks,results/{figures,tables}}
```

### 2. Copy Templates

```bash
# Copy README template
cp experiments/exp001_baseline_comparison/README.md experiments/exp00X_experiment_name/

# Copy metadata template
cp experiments/exp001_baseline_comparison/data/metadata.json experiments/exp00X_experiment_name/data/

# Copy analysis template
cp shared/templates/analysis_template.ipynb experiments/exp00X_experiment_name/notebooks/01_analysis.ipynb
```

### 3. Update Metadata

Edit `data/metadata.json` with:
- Experiment ID and name
- Design details
- Sample information
- Variable definitions

### 4. Add Raw Data

Place original data files in `data/raw/`:
- Use naming convention: `responses_YYYYMMDD.csv`
- Keep raw data unmodified
- Document any manual preprocessing

### 5. Conduct Analysis

Create notebooks in order:
- `01_data_exploration.ipynb` - Initial exploration
- `02_analysis.ipynb` - Main statistical analyses
- `03_visualization.ipynb` - Publication figures

Use shared utilities:
```python
import sys
sys.path.append('../../shared')
from utils import load_experiment_data, calculate_item_stats
```

### 6. Save Results

- Figures → `results/figures/`
- Tables → `results/tables/`
- Processed data → `data/processed/`

### 7. Document Findings

Update the experiment's `README.md` with:
- Key findings
- Sample sizes
- Any deviations from plan

## Data Format

All experiments should use **long format** data. See `docs/data_schema.md` for details.

### Required Columns

- `participant_id`: Unique identifier for each participant
- `item_id`: Unique identifier for each item
- `response`: Participant's response

### Recommended Columns

- `condition`: Experimental condition
- `correct`: Whether response was correct (0/1)
- `response_time`: Time taken (seconds)

## Shared Utilities

Located in `shared/utils/`, these functions promote consistency:

### Data Loading (`data_loading.py`)
- `load_experiment_data()`: Load data and metadata
- `validate_long_format()`: Check data structure
- `load_all_experiments()`: Get overview of all experiments

### Psychometrics (`psychometrics.py`)
- `calculate_item_stats()`: Difficulty, discrimination
- `calculate_reliability()`: Cronbach's alpha
- `calculate_item_total_correlation()`: Item discrimination

### Plotting (`plotting.py`)
- `setup_plotting_style()`: Consistent plot styling
- `save_figure()`: Save in multiple formats
- `plot_item_difficulty()`: Common plot types

## Analysis Guidelines

See `docs/analysis_guidelines.md` for:
- Statistical best practices
- Psychometric analysis standards
- Reporting guidelines
- Common pitfalls to avoid

## Version Control

### What to Track

✅ Track:
- All code (Python scripts, notebooks)
- Documentation (README, docs/)
- Requirements files
- Small result files (< 1MB)
- Metadata files

❌ Don't track:
- Virtual environments (`venv/`)
- Large data files (> 10MB) - use data management system
- Temporary files
- Compiled Python files

### Commit Best Practices

- Commit after completing each analysis step
- Clear commit messages: "Add item difficulty analysis for exp001"
- Don't commit broken code
- Keep notebooks clean (clear output before committing)

## Adding New Utility Functions

1. Add function to appropriate file in `shared/utils/`
2. Import in `shared/utils/__init__.py`
3. Add docstring with parameters and examples
4. Update this documentation

## Questions?

See `README.md` or contact the project maintainers.
