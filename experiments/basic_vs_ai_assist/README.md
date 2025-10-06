# Processing Block Size Experiment

## Overview

This experiment evaluates the impact of different processing block sizes on World War II history test performance. We compare two experimental conditions (forms) that differ in their processing approach to understand whether block size affects test results, item statistics, or overall reliability.

**Research Questions:**
1. Do different processing block sizes (1000 vs 500) result in different test performance?
2. Are item difficulty and discrimination equivalent across forms?
3. Does processing block size affect test reliability?

## Experiment Details

- **Date Started**: 2025-10-02
- **Date Completed**: 2025-10-02
- **Researcher(s)**: Chris Foster
- **Status**: Completed - Analysis in Progress

## Design

### Conditions
1. **1000_chat**: Processing block size of 1000
2. **500_chat**: Processing block size of 500

### Sample
- **Target N**: 100+ participants per condition
- **Actual N**: To be determined from data
- **Inclusion Criteria**: Completed all test items (excluding surveys)
- **Exclusion Criteria**: Survey questions excluded from analysis

### Materials
- **Item Types**: Multiple choice questions on World War II history
- **Number of Items**: ~20 test items per participant
- **Content Areas**: 
  - Military history and battles
  - Political events and decisions
  - War crimes and civilian impacts
  - Post-war consequences 

## Data Files

### Raw Data
- `data/raw/ww2_processing_block_size.txt` - Tab-delimited participant responses
  - **Key Columns**: delivery_id, item_id, item_name, form_name, score, total_seconds
  - **Survey Items Excluded**: Survey - History, Survey - History Knowledge, Survey - General History Knowledge, Survey - Completion Code

### Processed Data
- `data/processed/participant_scores_by_form.csv` - Participant-level aggregated scores by form

## Analysis

### Notebooks
1. `processing_block_size.ipynb` - Complete analysis including:
   - Data loading and filtering
   - Descriptive statistics by form
   - Item analysis (difficulty, discrimination)
   - Reliability analysis (Cronbach's alpha)
   - Visualizations
   - Statistical comparisons

### Key Findings
*To be completed after running analysis*

**Psychometric Criteria:**
- Item difficulty should be 0.2 - 0.8 (ideally ~0.5)
- Item discrimination should be > 0.2
- Test reliability (α) should be > 0.7

## Results

### Figures
- `results/figures/processing_block_analysis.png` - Four-panel figure with:
  1. Score distribution by form
  2. Box plots comparing forms
  3. Item difficulty comparison
  4. Item difficulty vs. discrimination scatter plot

### Tables
- `results/tables/item_statistics_by_form.csv` - Item-level statistics
- `results/tables/form_summary.csv` - Overall performance by form
- `results/tables/reliability_by_form.csv` - Cronbach's alpha by form

## Analysis Instructions

1. Activate virtual environment:
   ```bash
   source ../../../venv/bin/activate
   ```

2. Launch Jupyter:
   ```bash
   cd notebooks
   jupyter notebook processing_block_size.ipynb
   ```

3. Run all cells to generate:
   - Statistical summaries
   - Figures
   - Exported result tables

## Notes

- Survey questions are systematically excluded from test item analysis
- All test items are multiple choice with binary scoring (0 = incorrect, 1 = correct)
- Forms represent different processing block size conditions
- Item-total correlations use corrected totals (excluding target item)
