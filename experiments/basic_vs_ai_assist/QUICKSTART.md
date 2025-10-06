# Quick Start: Basic vs AI Assist Analysis

## Run the Analysis

### Option 1: Marimo (Recommended - Interactive & Reactive)

```bash
# 1. Navigate to project root
cd /home/chris-foster/PycharmProjects/item_writer_evaluations

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start Marimo
cd experiments/basic_vs_ai_assist/notebooks
marimo edit basic_vs_ai_assist.py
```

Marimo will open in your browser with an interactive, reactive notebook where:
- Cells automatically re-run when dependencies change
- No hidden state issues
- Better UI with built-in widgets
- Can be run as a web app with `marimo run`

### Option 2: Jupyter (Traditional)

```bash
# 1. Navigate to project root
cd /home/chris-foster/PycharmProjects/item_writer_evaluations

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start Jupyter
cd experiments/basic_vs_ai_assist/notebooks
jupyter notebook basic_vs_ai_assist.ipynb
```

## What the Analysis Does

The notebook performs a complete item analysis comparing three forms (100_chat, 500_chat, and 1000_chat):

### 1. Data Loading & Filtering
- Loads tab-delimited data from `data/raw/basic_vs_ai_assist.txt`
- Excludes survey questions:
  - completion code_v2
  - ProlificID_v1
  - history knowledge_v1

### 2. Descriptive Statistics
- Participant counts by form
- Mean scores, standard deviations
- Total correct responses
- Time on test
- Independent t-test comparing forms

### 3. Item Analysis
- **Difficulty**: Proportion of participants answering correctly (0-1)
  - Target: 0.2 - 0.8, ideally ~0.5
- **Discrimination**: Item-total correlation (point-biserial)
  - Target: > 0.2
- **Flags**: Problematic items automatically identified

### 4. Reliability
- Cronbach's alpha for each form
- Target: α > 0.7 (acceptable), α > 0.8 (good)

### 5. Visualizations
Four-panel figure saved to `results/figures/`:
1. Score distribution histograms by form
2. Box plots comparing forms
3. Item difficulty comparison (side-by-side bars)
4. Difficulty vs. discrimination scatter plot

### 6. Export Results
All results saved to:
- `results/tables/item_statistics_by_form.csv`
- `results/tables/form_summary.csv`
- `results/tables/reliability_by_form.csv`
- `data/processed/participant_scores_by_form.csv`

## Interpreting Results

### Good Items
- Difficulty: 0.3 - 0.7 (moderate difficulty)
- Discrimination: > 0.3 (good discrimination)

### Problematic Items
- Too easy: difficulty > 0.9
- Too hard: difficulty < 0.2
- Poor discrimination: < 0.15

### Test Quality
- **Reliability**: α > 0.8 indicates good internal consistency
- **Mean difficulty**: Should be around 0.5 for optimal test information
- **Form equivalence**: Check if t-test p-value > 0.05

## Data Structure

The raw data is in **long format** (one row per response):

| Column | Description |
|--------|-------------|
| delivery_id | Unique participant identifier |
| item_name | Item identifier |
| form_name | Experimental condition (100_chat, 500_chat, or 1000_chat) |
| score | Binary score (0=incorrect, 1=correct) |
| total_seconds | Time spent on item |

## Quick Checks

After running the analysis, review:
1. ✅ Are sample sizes balanced between forms?
2. ✅ Is reliability acceptable (α > 0.7) for both forms?
3. ✅ Are there many problematic items flagged?
4. ✅ Do the forms show equivalent performance (p > 0.05)?
5. ✅ Is the mean difficulty around 0.5?

## Next Steps

Based on results:
- **Problematic items**: Review content, consider revision/removal
- **Form differences**: Investigate systematic differences
- **Low reliability**: May need more items or review item quality
- **Extreme difficulty**: Adjust item difficulty in future versions
