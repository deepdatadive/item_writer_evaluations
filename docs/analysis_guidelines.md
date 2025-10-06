# Analysis Guidelines

## General Principles

1. **Reproducibility**: All analyses should be fully reproducible from raw data
2. **Documentation**: Document all decisions, transformations, and exclusions
3. **Version Control**: Commit data and analysis code to git
4. **Transparency**: Report all analyses conducted, not just significant results

## Workflow

### 1. Data Exploration (Notebook 01)

- Load raw data
- Check data quality
- Identify missing data or anomalies
- Create descriptive statistics
- Visualize distributions

**Outputs**: 
- Data quality report
- Descriptive statistics tables
- Distribution plots

### 2. Data Cleaning (Script or Notebook)

- Handle missing data
- Remove invalid responses
- Apply exclusion criteria
- Transform variables as needed
- Save cleaned data to `data/processed/`

**Document**:
- Number of observations removed and why
- Any data transformations applied
- Decisions about handling edge cases

### 3. Main Analysis (Notebook 02)

- Perform planned statistical tests
- Calculate psychometric properties
- Test hypotheses
- Conduct sensitivity analyses

**Outputs**:
- Statistical test results
- Effect sizes and confidence intervals
- Tables for publication

### 4. Visualization (Notebook 03)

- Create publication-quality figures
- Ensure consistent styling
- Save in multiple formats (PNG, PDF)
- Save to `results/figures/`

**Guidelines**:
- Use colorblind-friendly palettes
- Include error bars where appropriate
- Label axes clearly
- Add legends and annotations

## Statistical Considerations

### Sample Size

- Report planned and actual sample sizes
- Conduct power analysis if relevant
- Document any stopping rules

### Multiple Comparisons

- Adjust for multiple comparisons when appropriate
- Consider family-wise error rate
- Report both adjusted and unadjusted p-values

### Effect Sizes

- Always report effect sizes (Cohen's d, eta-squared, etc.)
- Include confidence intervals
- Interpret practical significance, not just statistical significance

## Psychometric Analysis

### Item Analysis

Standard metrics to report:
- **Item difficulty** (p-value): Proportion of correct responses
- **Item discrimination**: Item-total correlation or point-biserial
- **Distractor analysis**: For multiple choice items

### Reliability

Calculate and report:
- **Cronbach's alpha**: Internal consistency
- **Test-retest reliability**: If applicable
- **Inter-rater reliability**: If applicable

### Validity

Consider:
- **Content validity**: Expert review
- **Construct validity**: Confirmatory factor analysis
- **Criterion validity**: Correlations with external measures

## Comparing AI vs. Human-Generated Items

Key comparisons:
1. **Difficulty**: Are items equally difficult?
2. **Discrimination**: Do items discriminate equally well?
3. **Quality**: Expert ratings of item quality
4. **Efficiency**: Time/cost to generate
5. **Validity**: Do they measure the same construct?

### Recommended Tests

- **t-tests** or **Mann-Whitney U**: Compare means
- **Chi-square**: Compare categorical distributions
- **ANOVA**: Compare multiple conditions
- **Mixed models**: Account for nesting (items within conditions)

## Code Style

- Use meaningful variable names
- Add comments for complex operations
- Follow PEP 8 style guide
- Use the shared utility functions when possible

## Reporting Results

### Tables

- Include descriptive statistics (M, SD, range)
- Report test statistics, p-values, and effect sizes
- Use consistent formatting
- Save as CSV files in `results/tables/`

### Figures

- Use consistent color schemes across experiments
- Label all axes with units
- Include sample sizes in captions
- Save in both PNG (for viewing) and PDF (for publication)

### Text

- Report exact p-values (not just p < .05)
- Include confidence intervals
- Describe practical significance
- Discuss limitations

## Common Pitfalls to Avoid

1. **P-hacking**: Don't run multiple analyses and only report significant ones
2. **Cherry-picking**: Report all conditions/analyses, not just interesting ones
3. **Ignoring assumptions**: Check assumptions of statistical tests
4. **Overinterpreting**: Be cautious about causal claims
5. **Ignoring effect sizes**: Statistical significance ≠ practical importance

## Resources

- [APA Style Guide](https://apastyle.apa.org/)
- [Best Practices in Psychometric Analysis](https://www.apa.org/science/programs/testing/standards)
- Shared utility functions: `shared/utils/`
- Template notebook: `shared/templates/analysis_template.ipynb`
