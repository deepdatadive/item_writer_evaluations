# Data Cleaning Summary: Basic vs AI Assist Analysis

## Overview

This document summarizes the data cleaning and filtering process for the WW2 history test analysis comparing AI-assisted vs. GPT-generated items.

## Target Forms

The analysis focuses on two specific forms from the dataset:

### Form 1: WW2 Form 1 AI Assist questions
- **Purpose**: Items generated with AI assistance
- **Expected items**: 10 test items (plus survey questions)
- **Characteristics**: "ai_assist" and "aiassist" prefixed items

### Form 2: WW2 Form 2 GPT questions  
- **Purpose**: Items generated directly by GPT
- **Expected items**: 22 test items (plus survey questions)
- **Characteristics**: "gpt_question" prefixed items

## Data Filtering Steps

### Step 1: Form Filtering
**Action**: Filter dataset to only include the two target forms
- Original data may contain multiple other forms
- Keep only: `"WW2 Form 1 AI Assist questions"` and `"WW2 Form 2 GPT questions"`

### Step 2: Separate Test Items from Survey Items

**Survey Items** (excluded from psychometric analysis but used for correlations):
- `completion code_v2`
- `completion code_v3`
- `ProlificID_v1`
- `ProlificID_v2`
- `history knowledge_v1`
- `ww2_knowledge_v1`
- `Quiz Feeling Simple_v1`
- `quiz feeling complex_v1`

**Test Items**: All remaining items after excluding surveys

### Step 3: Handle Multiple Item Versions
**Issue**: Some items may have multiple versions (e.g., `item_name_v1`, `item_name_v2`)

**Solution**: Use `item_name` instead of `item_version_name` for analysis
- This consolidates all versions of the same item
- Assumes different versions are minor variants of the same question
- Simpler analysis without version complications

### Step 4: Remove NaN Form Names
**Action**: Filter out any rows where `form_name` is NaN/missing
- Prevents sorting errors
- Ensures clean data for analysis

## Data Structure

### Final Cleaned Datasets

1. **df_filtered**: All data from the two target forms (test + survey items)
2. **df_test**: Only test items (excludes surveys)
3. **df_survey**: Only survey items (for correlation analysis)

### Key Columns

- `delivery_id`: Unique participant identifier
- `form_name`: Form assignment ("WW2 Form 1..." or "WW2 Form 2...")
- `item_name`: Item identifier (without version number)
- `item_version_name`: Item identifier with version (e.g., item_name_v1)
- `score`: Binary score (0=incorrect, 1=correct)
- `total_seconds`: Time spent on item

## Analysis Workflow

### 1. Data Exploration (Sections 2-5)
- Print all unique form names
- Filter to target forms
- List items in each form
- Check for multiple versions

### 2. Summary Statistics (Section 6)
- Participant counts
- Score distributions
- Item difficulty ranges
- Time statistics
- Data completeness checks

### 3. Statistical Comparison (Section 7)
- T-test comparing forms
- Cohen's d effect size
- Performance metrics by form

### 4. Item Analysis (Section 8+)
- Item difficulty (proportion correct)
- Item discrimination (item-total correlation)
- Flag problematic items
- Reliability analysis (Cronbach's alpha)

## Quality Checks

### Participant-Level
- [ ] Each participant has expected number of items
- [ ] No duplicate responses (same participant × item)
- [ ] Score values are valid (0 or 1)

### Item-Level
- [ ] All items have responses from multiple participants
- [ ] Item difficulty between 0.2-0.8 (ideal)
- [ ] Item discrimination > 0.2 (acceptable)
- [ ] No items with all correct or all incorrect responses

### Form-Level
- [ ] Sample sizes adequate (N > 30 per form)
- [ ] Reliability (Cronbach's α) > 0.7
- [ ] Forms are comparable in difficulty

## Expected Outputs

### Tables
- `item_statistics_by_form.xlsx`: Complete item stats
- `quality_summary_by_form.csv`: Summary of item quality
- `survey_correlations.csv`: Survey-performance correlations

### Figures
- Score distribution comparisons
- Item difficulty vs. discrimination plots
- Survey correlation scatterplots

## Notes

### Why Use item_name Instead of item_version_name?
- Simplifies analysis
- Consolidates minor item variations
- Prevents version-related complications
- More stable statistics with larger sample sizes

### Why These Specific Forms?
- Focuses on the core comparison: AI-assisted vs. GPT-generated
- Excludes pilot data or other experimental conditions
- Clean, targeted analysis for research questions

### Survey Items
- NOT included in test scoring or reliability analysis
- USED for correlational analysis with test performance
- Important for validity evidence (e.g., do self-reported knowledge correlate with actual performance?)

## Troubleshooting

### Common Issues

**Issue**: TypeError when sorting forms
- **Cause**: NaN values in form_name
- **Solution**: Filter with `pd.notna(form_name)` before sorting

**Issue**: Inconsistent item counts across participants  
- **Cause**: Missing data or incomplete responses
- **Solution**: Check data completeness report in Section 6

**Issue**: Items flagged as problematic
- **Cause**: Poor psychometric properties (too easy/hard, low discrimination)
- **Solution**: Review item content, consider revision or removal

## Contact

For questions about this data cleaning process, refer to:
- Main analysis notebook: `basic_vs_ai_assist.py`
- Project README: `/item_writer_evaluations/README.md`

