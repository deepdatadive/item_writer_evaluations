import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # Basic vs AI Assist Analysis

    This notebook analyzes the comparison between basic and AI-assisted item writing for WW2 history test items.

    **Experiment**: Comparing two forms of WW2 history items
    - Form 1: AI Assist questions
    - Form 2: GPT questions
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Setup paths
    NOTEBOOK_DIR = Path.cwd()
    EXPERIMENT_DIR = NOTEBOOK_DIR.parent
    DATA_DIR = EXPERIMENT_DIR / "data"
    RESULTS_DIR = EXPERIMENT_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"

    # Ensure directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Notebook directory: {NOTEBOOK_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    return Path, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---

    ## Generate Report

    Run the cell below to convert this notebook to an HTML report. The report will be saved in the `results/` directory.
    """
    )
    return


@app.cell
def _(Path):
    import subprocess
    from datetime import datetime

    def generate_report(output_format='html', execute=False, hide_code=False):
        """
        Generate a report from this notebook.

        Parameters:
        -----------
        output_format : str
            Format for the report: 'html', 'pdf', 'webpdf', 'markdown'
        execute : bool
            If True, re-execute all cells before converting (like knittr)
        hide_code : bool
            If True, hide code cells and show only outputs
        """
        notebook_path = Path.cwd() / 'processing_block_size.ipynb'
        results_dir = Path.cwd().parent / 'results'  # Get paths
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'processing_block_size_report_{timestamp}'
        output_path = results_dir / f'{output_name}.{output_format}'
        cmd = ['jupyter', 'nbconvert', '--to', output_format, str(notebook_path), '--output', str(output_path)]  # Generate timestamp for filename
        if execute:
            cmd.append('--execute')
        if hide_code:
            cmd.append('--no-input')
        if output_format == 'html':  # Build command
            cmd.extend(['--embed-images', '--template', 'lab'])
        print(f'Generating {output_format.upper()} report...')
        print(f'Command: {' '.join(cmd)}')
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f'\n✅ Report generated successfully!')
            print(f'📄 Location: {output_path}')
            return output_path
        except subprocess.CalledProcessError as e:
            print(f'❌ Error generating report:')
            print(e.stderr)
    # Generate the report
    # Uncomment the options you want:
    # Basic HTML report (code visible)
    # generate_report(output_format='html')
    # HTML report with code hidden (output only)
    # generate_report(output_format='html', hide_code=True)
    # Re-execute everything and generate report (like knittr)
    # generate_report(output_format='html', execute=True)
    # PDF report (requires LaTeX)
    # generate_report(output_format='pdf')
    # PDF via HTML (no LaTeX required, needs chromium)
    # generate_report(output_format='webpdf')
            return None  # Add options for better HTML output  # Embed images in HTML  # Use modern template
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Processing Block Size Experiment Analysis

    **Experiment**: WW2 History Test - Processing Block Size Comparison

    **Date**: 2025-10-06

    **Analyst**: Chris Foster

    ## Overview

    This analysis examines the impact of different processing block sizes on test performance in a World War II history assessment.
    """
    )
    return


@app.cell
def _(pd, plt, sns):
    # Import libraries
    import numpy as np
    from scipy import stats
    plt.style.use('dark_background')
    custom_colors = ['#00FFFF', '#FF1493', '#FF6600']
    muted_grey = '#808080'
    light_grey = '#B0B0B0'
    sns.set_palette(custom_colors)
    # Set up dark theme plotting for presentation
    pd.set_option('display.max_columns', None)
    plt.rcParams['figure.facecolor'] = 'black'
    # Define custom color palette - BRIGHT NEON colors for dark background
    # Cyan, Hot Pink, Neon Orange
    plt.rcParams['axes.facecolor'] = 'black'  # Bright cyan, hot pink, neon orange
    plt.rcParams['axes.edgecolor'] = light_grey
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['xtick.color'] = light_grey
    plt.rcParams['ytick.color'] = light_grey
    plt.rcParams['grid.color'] = muted_grey
    # Configure matplotlib for dark theme
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['legend.facecolor'] = 'black'
    plt.rcParams['legend.edgecolor'] = light_grey
    print('✓ Libraries loaded with NEON dark theme (Cyan/Pink/Orange)')
    return light_grey, muted_grey, np, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Load Data""")
    return


@app.cell
def _(Path, pd):
    EXPERIMENT_PATH = Path('..')
    DATA_FILE = 'basic_vs_ai_assist.txt'

    data_path = EXPERIMENT_PATH / 'data' / 'raw' / DATA_FILE
    df = pd.read_csv(data_path, sep='\t')

    # CRITICAL: Convert numeric columns immediately after loading
    # This prevents string concatenation issues downstream
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['total_seconds'] = pd.to_numeric(df['total_seconds'], errors='coerce')

    print(f"Loaded {len(df)} rows from raw data file")
    print(f"Participants: {df['delivery_id'].nunique()}")
    print(f"All forms in file: {df['form_name'].unique()}")
    print(f"\n✅ Data types after loading:")
    print(f"   score: {df['score'].dtype}")
    print(f"   total_seconds: {df['total_seconds'].dtype}")
    print(f"\n⚠️ NOTE: We will filter to ONLY WW2 forms in the next section")
    print(f"   - WW2 Form 1 AI Assist questions")
    print(f"   - WW2 Form 2 GPT questions")
    df.head()
    return EXPERIMENT_PATH, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Explore Form Names and Data Structure""")
    return


@app.cell
def _(df, pd):
    # Print all unique form names
    print("=" * 80)
    print("ALL UNIQUE FORM NAMES IN THE DATA:")
    print("=" * 80)

    _all_forms = sorted([f for f in df['form_name'].unique() if pd.notna(f)])
    for _i, _form in enumerate(_all_forms, 1):
        _n_participants = df[df['form_name'] == _form]['delivery_id'].nunique()
        _n_responses = len(df[df['form_name'] == _form])
        print(f"{_i:2d}. '{_form}' - {_n_participants} participants, {_n_responses} responses")

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {len(_all_forms)} unique forms")
    print(f"{'=' * 80}")

    # Check for NaN forms
    _nan_count = df['form_name'].isna().sum()
    if _nan_count > 0:
        print(f"\n⚠️ WARNING: {_nan_count} rows have missing (NaN) form_name values")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Clean and Filter Data for Analysis""")
    return


@app.cell
def _(df, pd):
    # Define the two forms we want to analyze
    FORM_1_NAME = "WW2 Form 1 AI Assist questions"
    FORM_2_NAME = "WW2 Form 2 GPT questions"

    # Survey items that should be excluded from test analysis but kept for correlation
    survey_items = [
        'completion code_v2',
        'completion code_v3',
        'ProlificID_v1', 
        'ProlificID_v2',
        'history knowledge_v1',
        'ww2_knowledge_v1',
        'Quiz Feeling Simple_v1',
        'quiz feeling complex_v1'
    ]

    # Show what we're filtering OUT
    _all_forms = [f for f in df['form_name'].unique() if pd.notna(f)]
    _excluded_forms = [f for f in _all_forms if f not in [FORM_1_NAME, FORM_2_NAME]]

    print(f"{'=' * 80}")
    print(f"DATA FILTERING SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nOriginal data: {len(df)} rows, {df['delivery_id'].nunique()} participants")
    print(f"All forms in dataset: {len(_all_forms)}")

    print(f"\n✅ KEEPING these forms:")
    print(f"   1. {FORM_1_NAME}")
    print(f"   2. {FORM_2_NAME}")

    print(f"\n❌ EXCLUDING these {len(_excluded_forms)} forms:")
    for _i, _form in enumerate(_excluded_forms, 1):
        _n_rows = len(df[df['form_name'] == _form])
        print(f"   {_i}. {_form} ({_n_rows} rows)")

    # Filter to only the two forms we want
    df_filtered = df[df['form_name'].isin([FORM_1_NAME, FORM_2_NAME])].copy()

    print(f"\n{'=' * 80}")
    print(f"✅ FILTERED DATA:")
    print(f"{'=' * 80}")
    print(f"Rows: {len(df_filtered)} (was {len(df)}, removed {len(df) - len(df_filtered)})")
    print(f"Participants: {df_filtered['delivery_id'].nunique()} (was {df['delivery_id'].nunique()})")
    print(f"Forms: {df_filtered['form_name'].nunique()} (was {len(_all_forms)})")

    # Separate test items from survey items
    df_test = df_filtered[~df_filtered['item_name'].isin(survey_items)].copy()
    df_survey1 = df_filtered[df_filtered['item_name'].isin(survey_items)].copy()

    # Convert numeric columns to proper types immediately
    df_test['score'] = pd.to_numeric(df_test['score'], errors='coerce')
    df_test['total_seconds'] = pd.to_numeric(df_test['total_seconds'], errors='coerce')

    print(f"\n{'=' * 80}")
    print(f"BREAKDOWN:")
    print(f"{'=' * 80}")
    print(f"Test items: {len(df_test)} rows")
    print(f"Survey items: {len(df_survey1)} rows")
    print(f"\nData types in df_test:")
    print(f"  score: {df_test['score'].dtype}")
    print(f"  total_seconds: {df_test['total_seconds'].dtype}")

    # Summary by form
    print(f"\n{'=' * 80}")
    print(f"BY FORM:")
    print(f"{'=' * 80}")

    for _form in [FORM_1_NAME, FORM_2_NAME]:
        _form_test = df_test[df_test['form_name'] == _form]
        _form_survey = df_survey1[df_survey1['form_name'] == _form]

        print(f"\n{_form}:")
        print(f"  Participants: {_form_test['delivery_id'].nunique()}")
        print(f"  Test items (unique): {_form_test['item_name'].nunique()}")
        print(f"  Test responses: {len(_form_test)}")
        print(f"  Survey items (unique): {_form_survey['item_name'].nunique()}")
        print(f"  Survey responses: {len(_form_survey)}")
    return FORM_1_NAME, FORM_2_NAME, df_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4. Examine Items in Each Form and Extract Version Info""")
    return


@app.cell
def _(df_test, pd):
    # Extract version suffix from item_version_name (e.g., "_v1", "_v2")
    # The version is typically the last 3 characters: _v1, _v2, etc.

    df_test_versioned = df_test.copy()

    # Fix data type issues - convert total_seconds to numeric
    df_test_versioned['total_seconds'] = pd.to_numeric(df_test_versioned['total_seconds'], errors='coerce')
    df_test_versioned['score'] = pd.to_numeric(df_test_versioned['score'], errors='coerce')

    # Extract version suffix (last 3 chars if it matches _vX pattern)
    df_test_versioned['version_suffix'] = df_test_versioned['item_version_name'].str[-3:]

    # For items that don't follow the _vX pattern, mark as 'N/A'
    _is_valid_version = df_test_versioned['version_suffix'].str.match(r'_v\d')
    df_test_versioned.loc[~_is_valid_version, 'version_suffix'] = 'N/A'

    # Create a clean item name by removing the version suffix
    df_test_versioned['item_base'] = df_test_versioned['item_version_name'].str[:-3]
    df_test_versioned.loc[~_is_valid_version, 'item_base'] = df_test_versioned.loc[~_is_valid_version, 'item_version_name']

    print(f"{'=' * 100}")
    print(f"VERSION ANALYSIS")
    print(f"{'=' * 100}")
    print(f"\nSample of version extraction:")
    print(df_test_versioned[['item_name', 'item_version_name', 'item_base', 'version_suffix']].drop_duplicates().head(20))

    print(f"\n{'=' * 100}")
    print(f"VERSION DISTRIBUTION")
    print(f"{'=' * 100}")
    _version_counts = df_test_versioned['version_suffix'].value_counts()
    print(_version_counts)

    # Verify data types
    print(f"\n{'=' * 100}")
    print(f"DATA TYPE VERIFICATION")
    print(f"{'=' * 100}")
    print(f"score dtype: {df_test_versioned['score'].dtype}")
    print(f"total_seconds dtype: {df_test_versioned['total_seconds'].dtype}")
    print(f"\nSample values:")
    print(df_test_versioned[['score', 'total_seconds']].head(10))
    return (df_test_versioned,)


@app.cell
def _(FORM_1_NAME, FORM_2_NAME, df_test_versioned):
    # Show detailed version breakdown by form
    print(f"{'=' * 100}")
    print(f"DETAILED VERSION BREAKDOWN BY FORM")
    print(f"{'=' * 100}")

    for _form_name in [FORM_1_NAME, FORM_2_NAME]:
        _form_data = df_test_versioned[df_test_versioned['form_name'] == _form_name]

        print(f"\n{_form_name}")
        print(f"{'=' * 100}")

        # Group by item_name and show all versions
        for _item_name in sorted(_form_data['item_name'].unique()):
            _item_data = _form_data[_form_data['item_name'] == _item_name]
            _versions = _item_data[['item_version_name', 'version_suffix']].drop_duplicates().sort_values('item_version_name')

            print(f"\nItem: {_item_name}")
            print(f"  Versions: {len(_versions)}")

            if len(_versions) > 1:
                print(f"  Details:")
                for _, _row in _versions.iterrows():
                    _count = len(_item_data[_item_data['item_version_name'] == _row['item_version_name']])
                    print(f"    - {_row['item_version_name']} (suffix: {_row['version_suffix']}, {_count} responses)")
            else:
                _row = _versions.iloc[0]
                _count = len(_item_data)
                print(f"    - {_row['item_version_name']} (suffix: {_row['version_suffix']}, {_count} responses)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Item Matching Strategy: Use Version Suffix""")
    return


@app.cell
def _(df_test_versioned, pd):
    # Check which items have multiple versions
    print(f"{'=' * 80}")
    print(f"ITEMS WITH MULTIPLE VERSIONS:")
    print(f"{'=' * 80}\n")

    _items_with_versions = []

    for _form in df_test_versioned['form_name'].unique():
        if pd.notna(_form):
            _form_data = df_test_versioned[df_test_versioned['form_name'] == _form]

            for _item in _form_data['item_name'].unique():
                _versions = _form_data[_form_data['item_name'] == _item]['item_version_name'].unique()

                if len(_versions) > 1:
                    _items_with_versions.append({
                        'form': _form,
                        'item': _item,
                        'num_versions': len(_versions),
                        'versions': sorted(_versions)
                    })

                    print(f"Form: {_form}")
                    print(f"Item: {_item}")
                    print(f"Versions ({len(_versions)}):")
                    for _v in sorted(_versions):
                        _count = len(_form_data[_form_data['item_version_name'] == _v])
                        print(f"  - {_v} ({_count} responses)")
                    print()

    if not _items_with_versions:
        print("✓ No items with multiple versions found!")
    else:
        print(f"\n⚠️ Found {len(_items_with_versions)} items with multiple versions")
        print(f"\nSTRATEGY: Use item_name (without version) for analysis")
        print(f"This will consolidate all versions of the same item together.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Summary Statistics by Form""")
    return


@app.cell
def _(FORM_1_NAME, FORM_2_NAME, df_test_versioned):
    # Calculate summary statistics for each form
    print(f"{'=' * 80}")
    print(f"SUMMARY STATISTICS BY FORM")
    print(f"{'=' * 80}\n")

    for _form_name in [FORM_1_NAME, FORM_2_NAME]:
        _form_data = df_test_versioned[df_test_versioned['form_name'] == _form_name]

        print(f"\n{_form_name}")
        print(f"{'=' * 80}")

        # Participant stats
        _n_participants = _form_data['delivery_id'].nunique()
        print(f"  Participants: {_n_participants}")

        # Item stats
        _n_items = _form_data['item_name'].nunique()
        print(f"  Unique items: {_n_items}")

        # Calculate participant-level scores
        _participant_scores = _form_data.groupby('delivery_id').agg({
            'score': ['mean', 'sum', 'count'],
            'total_seconds': 'sum'
        }).reset_index()

        _participant_scores.columns = ['delivery_id', 'mean_score', 'total_correct', 'n_items', 'total_time']

        # Score statistics
        print(f"\n  Score Statistics:")
        print(f"    Mean score: {_participant_scores['mean_score'].mean():.3f} ± {_participant_scores['mean_score'].std():.3f}")
        print(f"    Min score: {_participant_scores['mean_score'].min():.3f}")
        print(f"    Max score: {_participant_scores['mean_score'].max():.3f}")
        print(f"    Median score: {_participant_scores['mean_score'].median():.3f}")

        # Item difficulty
        _item_difficulty = _form_data.groupby('item_name')['score'].mean()
        print(f"\n  Item Difficulty:")
        print(f"    Mean difficulty: {_item_difficulty.mean():.3f}")
        print(f"    Range: {_item_difficulty.min():.3f} to {_item_difficulty.max():.3f}")

        # Time statistics
        print(f"\n  Time Statistics:")
        print(f"    Mean total time: {_participant_scores['total_time'].mean():.1f}s ({_participant_scores['total_time'].mean()/60:.1f} min)")
        print(f"    Median total time: {_participant_scores['total_time'].median():.1f}s ({_participant_scores['total_time'].median()/60:.1f} min)")

        # Responses per item
        _responses_per_item = _form_data.groupby('item_name').size()
        print(f"\n  Data Completeness:")
        print(f"    Min responses per item: {_responses_per_item.min()}")
        print(f"    Max responses per item: {_responses_per_item.max()}")
        if _responses_per_item.min() != _responses_per_item.max():
            print(f"    ⚠️ WARNING: Unequal responses across items!")
            print(f"    Items with fewer responses:")
            for _item, _count in _responses_per_item[_responses_per_item < _responses_per_item.max()].items():
                print(f"      - {_item}: {_count} responses")
        else:
            print(f"    ✓ All items have equal responses ({_responses_per_item.min()})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Statistical Comparison Between Forms""")
    return


@app.cell
def _(FORM_1_NAME, FORM_2_NAME, df_test_versioned, pd, stats):
    # Calculate participant-level statistics
    participant_stats = df_test_versioned.groupby(['delivery_id', 'form_name']).agg({
        'score': ['mean', 'sum', 'count'], 
        'total_seconds': 'sum'
    }).reset_index()

    participant_stats.columns = ['delivery_id', 'form_name', 'mean_score', 'total_correct', 'n_items', 'total_time']

    # Ensure numeric types after aggregation
    participant_stats['mean_score'] = pd.to_numeric(participant_stats['mean_score'], errors='coerce')
    participant_stats['total_correct'] = pd.to_numeric(participant_stats['total_correct'], errors='coerce')
    participant_stats['n_items'] = pd.to_numeric(participant_stats['n_items'], errors='coerce')
    participant_stats['total_time'] = pd.to_numeric(participant_stats['total_time'], errors='coerce')

    # Remove any NaN form names
    participant_stats = participant_stats[pd.notna(participant_stats['form_name'])]

    # Summary by form
    form_summary = participant_stats.groupby('form_name').agg({
        'mean_score': ['mean', 'std', 'min', 'max'], 
        'total_correct': ['mean', 'std'], 
        'n_items': 'first', 
        'total_time': ['mean', 'std'], 
        'delivery_id': 'count'
    }).round(3)

    form_summary.columns = ['_'.join(col) for col in form_summary.columns]
    form_summary.rename(columns={'delivery_id_count': 'n_participants'}, inplace=True)

    print(f"{'=' * 80}")
    print('PERFORMANCE BY FORM:')
    print(f"{'=' * 80}\n")
    print(form_summary)

    # Statistical comparisons
    print(f"\n{'=' * 80}")
    print('STATISTICAL COMPARISONS:')
    print(f"{'=' * 80}\n")

    # T-test between the two forms
    form1_scores = participant_stats[participant_stats['form_name'] == FORM_1_NAME]['mean_score']
    form2_scores = participant_stats[participant_stats['form_name'] == FORM_2_NAME]['mean_score']

    t_stat, p_val = stats.ttest_ind(form1_scores, form2_scores)

    print(f"{FORM_1_NAME}:")
    print(f"  N = {len(form1_scores)}")
    print(f"  Mean = {form1_scores.mean():.3f}")
    print(f"  SD = {form1_scores.std():.3f}")

    print(f"\n{FORM_2_NAME}:")
    print(f"  N = {len(form2_scores)}")
    print(f"  Mean = {form2_scores.mean():.3f}")
    print(f"  SD = {form2_scores.std():.3f}")

    print(f"\nIndependent t-test:")
    print(f"  t({len(form1_scores) + len(form2_scores) - 2}) = {t_stat:.3f}")
    print(f"  p = {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

    # Cohen's d effect size
    pooled_std = ((form1_scores.std()**2 + form2_scores.std()**2) / 2) ** 0.5
    cohens_d = (form1_scores.mean() - form2_scores.mean()) / pooled_std
    print(f"  Cohen's d = {cohens_d:.3f}")

    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    print(f"  Effect size: {effect_size}")
    return (participant_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 8. Item Analysis""")
    return


@app.cell
def _(df_test_versioned, pd):
    # Item statistics - use item_name (consolidates versions)
    # Now we have version_suffix and item_base available if needed

    item_stats = df_test_versioned.groupby(['item_name', 'form_name']).agg({
        'score': ['mean', 'std', 'count'], 
        'total_seconds': 'mean'
    }).reset_index()

    item_stats.columns = ['item_name', 'form_name', 'difficulty', 'sd', 'n', 'time']

    # Calculate discrimination (item-total correlation)
    discriminations = []
    for _form in df_test_versioned['form_name'].unique():
        if pd.notna(_form):
            _wide = df_test_versioned[df_test_versioned['form_name'] == _form].pivot_table(
                index='delivery_id', 
                columns='item_name', 
                values='score'
            )

            for _item in _wide.columns:
                # Corrected item-total correlation (exclude item from total)
                _corr = _wide[_item].corr(_wide.sum(axis=1) - _wide[_item])
                discriminations.append({
                    'form_name': _form, 
                    'item_name': _item, 
                    'discrimination': _corr
                })

    disc_df = pd.DataFrame(discriminations)
    item_stats = item_stats.merge(disc_df, on=['form_name', 'item_name'])

    # Flag problematic items
    item_stats['flag'] = ''
    item_stats.loc[item_stats['discrimination'] < 0.15, 'flag'] += 'Low_Disc '
    item_stats.loc[item_stats['difficulty'] < 0.2, 'flag'] += 'Too_Hard '
    item_stats.loc[item_stats['difficulty'] > 0.9, 'flag'] += 'Too_Easy '

    # Round for cleaner display
    item_stats['difficulty'] = item_stats['difficulty'].round(3)
    item_stats['discrimination'] = item_stats['discrimination'].round(3)
    item_stats['time'] = item_stats['time'].round(1)

    print(f"{'=' * 100}")
    print(f'ITEM STATISTICS SUMMARY')
    print(f"{'=' * 100}")
    print(f'Total items analyzed: {len(item_stats)}')
    print(f'Problematic items flagged: {(item_stats['flag'] != '').sum()}')
    print(f'\nNote: Statistics calculated at item_name level (consolidating all versions)')

    print(f"\n{'=' * 100}")
    print('ITEM STATISTICS (sorted by form, then difficulty)')
    print('=' * 100)

    display_cols = ['form_name', 'item_name', 'difficulty', 'discrimination', 'n', 'time', 'flag']
    item_stats_display = item_stats[display_cols].sort_values(['form_name', 'difficulty'])

    # Set pandas display options for full table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.width', None)

    print(item_stats_display.to_string(index=False))

    # Reset display options
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', 50)
    return (item_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4.5 Interactive Searchable Table""")
    return


@app.cell
def _():
    # Interactive searchable table - use Ctrl+F to search in output
    # Style the dataframe with color coding
    def _highlight_issues(row):
        _colors = []
        for _col in row.index:
            if _col == 'difficulty':
                if row[_col] < 0.2:
                    _colors.append('background-color: #ffcccc')  # Red for too hard
                elif row[_col] > 0.9:
                    _colors.append('background-color: #ffffcc')  # Yellow for too easy
                elif 0.3 <= row[_col] <= 0.7:
                    _colors.append('background-color: #ccffcc')  # Green for good
                else:
                    _colors.append('')
            elif _col == 'discrimination':
                if row[_col] < 0.15:
                    _colors.append('background-color: #ffcccc')  # Red for poor
                elif row[_col] >= 0.3:
                    _colors.append('background-color: #ccffcc')  # Green for good
                else:
                    _colors.append('')
            elif _col == 'flag' and row[_col] != '':
                _colors.append('background-color: #ffcccc')  # Red if flagged
            else:
                _colors.append('')
        return _colors

    # Note: This cell requires 'item_short' column which may not exist
    # Commenting out for now - uncomment if you add item_short column
    # _styled_table = item_stats[display_cols].sort_values(['form_name', 'item_name']).style\
    #     .apply(_highlight_issues, axis=1)\
    #     .format({'difficulty': '{:.3f}', 'discrimination': '{:.3f}', 'time': '{:.1f}'})\
    #     .set_caption("Item Statistics by Form (Color coded: Green=Good, Yellow=Easy, Red=Problem)")\
    #     .set_properties(**{'text-align': 'left'})\
    #     .set_table_styles([
    #         {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold')]},
    #         {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]}
    #     ])

    # display(_styled_table)

    print("💡 TIP: Use Ctrl+F (or Cmd+F on Mac) to search for specific items in the output!")
    print("💡 Note: Styled table disabled - enable if you add 'item_short' column")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4.6 Export to Excel for Easy Filtering""")
    return


@app.cell
def _(EXPERIMENT_PATH, item_stats, pd):
    # Export to Excel with formatting for easy filtering and searching
    _excel_path = EXPERIMENT_PATH / 'results' / 'tables' / 'item_statistics_by_form.xlsx'
    _excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for export - using available columns only
    _export_cols = ['form_name', 'item_name', 'difficulty', 'discrimination', 'sd', 'n', 'time', 'flag']
    _export_data = item_stats[_export_cols].copy()
    _export_data = _export_data.sort_values(['form_name', 'item_name'])

    # Write to Excel
    with pd.ExcelWriter(_excel_path, engine='openpyxl') as _writer:
        _export_data.to_excel(_writer, sheet_name='Item Statistics', index=False)

        # Get the worksheet
        _worksheet = _writer.sheets['Item Statistics']

        # Auto-adjust column widths
        for _column in _worksheet.columns:
            _max_length = 0
            _column = [_cell for _cell in _column]
            for _cell in _column:
                try:
                    if len(str(_cell.value)) > _max_length:
                        _max_length = len(str(_cell.value))
                except:
                    pass
            _adjusted_width = min(_max_length + 2, 50)
            _worksheet.column_dimensions[_column[0].column_letter].width = _adjusted_width

    print(f"✓ Exported to Excel: {_excel_path}")
    print(f"\n💡 Open in Excel/Google Sheets for:")
    print(f"   - Column filtering")
    print(f"   - Sorting by any column")
    print(f"   - Advanced search and filtering")
    print(f"   - Pivot tables")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 4.7 Item Quality Summary by Form""")
    return


@app.cell
def _(EXPERIMENT_PATH, item_stats, pd):
    # Create summary of item quality issues by form
    _quality_summary = []
    for _form in sorted(item_stats['form_name'].unique()):
        _form_items = item_stats[item_stats['form_name'] == _form]
        _summary = {'form_name': _form, 'total_items': len(_form_items), 'too_hard': (_form_items['difficulty'] < 0.2).sum(), 'hard': ((_form_items['difficulty'] >= 0.2) & (_form_items['difficulty'] < 0.3)).sum(), 'good_difficulty': ((_form_items['difficulty'] >= 0.3) & (_form_items['difficulty'] <= 0.7)).sum(), 'easy': ((_form_items['difficulty'] > 0.7) & (_form_items['difficulty'] <= 0.9)).sum(), 'too_easy': (_form_items['difficulty'] > 0.9).sum(), 'poor_disc': (_form_items['discrimination'] < 0.15).sum(), 'low_disc': ((_form_items['discrimination'] >= 0.15) & (_form_items['discrimination'] < 0.2)).sum(), 'acceptable_disc': ((_form_items['discrimination'] >= 0.2) & (_form_items['discrimination'] < 0.3)).sum(), 'good_disc': (_form_items['discrimination'] >= 0.3).sum(), 'any_issues': (_form_items['flag'] != '').sum(), 'no_issues': (_form_items['flag'] == '').sum(), 'mean_difficulty': _form_items['difficulty'].mean(), 'mean_discrimination': _form_items['discrimination'].mean()}
        _quality_summary.append(_summary)
    _quality_df = pd.DataFrame(_quality_summary)
    print('=' * 100)
    print('ITEM QUALITY SUMMARY BY FORM')
    print('=' * 100)
    print('\n📊 DIFFICULTY DISTRIBUTION:')  # Difficulty issues
    print(_quality_df[['form_name', 'too_hard', 'hard', 'good_difficulty', 'easy', 'too_easy']].to_string(index=False))
    print('\n📈 DISCRIMINATION DISTRIBUTION:')
    print(_quality_df[['form_name', 'poor_disc', 'low_disc', 'acceptable_disc', 'good_disc']].to_string(index=False))
    print('\n✅ OVERALL QUALITY:')
    print(_quality_df[['form_name', 'total_items', 'no_issues', 'any_issues', 'mean_difficulty', 'mean_discrimination']].round(3).to_string(index=False))
    print('\n📊 PERCENTAGE OF ITEMS WITH ISSUES:')
    for _form in sorted(_quality_df['form_name']):  # Discrimination issues
        _row = _quality_df[_quality_df['form_name'] == _form].iloc[0]
        _pct_issues = _row['any_issues'] / _row['total_items'] * 100
        _pct_poor_disc = _row['poor_disc'] / _row['total_items'] * 100
        _pct_difficulty_issues = (_row['too_hard'] + _row['too_easy']) / _row['total_items'] * 100
        print(f'\n{_form}:')
        print(f'  Items with ANY issues: {_row["any_issues"]}/{_row["total_items"]} ({_pct_issues:.1f}%)')  # Overall quality
        print(f'  Poor discrimination: {_row["poor_disc"]}/{_row["total_items"]} ({_pct_poor_disc:.1f}%)')
        print(f'  Difficulty issues: {_row["too_hard"] + _row["too_easy"]}/{_row["total_items"]} ({_pct_difficulty_issues:.1f}%)')
    _summary_path = EXPERIMENT_PATH / 'results' / 'tables' / 'quality_summary_by_form.csv'
    _quality_df.to_csv(_summary_path, index=False)  # Average stats
    # Display summary
    # Calculate percentages for better comparison
    # Save summary to file
    print(f'\n✓ Summary saved to: {_summary_path}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. UPDATED Visualizations (Dark Theme for Presentation)""")
    return


@app.cell
def _(item_stats, muted_grey, np, participant_stats, plt):
    # DARK THEME VISUALIZATIONS - Use these for your presentation!
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
    colors = {
        'WW2 Form 1 AI Assist questions': '#00FFFF',  # Cyan
        'WW2 Form 2 GPT questions': '#FF1493'          # Hot Pink
    }
    # Color scheme - NEON colors for dark background
    _ax = _axes[0, 0]  # Cyan, Pink
    for _form in sorted(participant_stats['form_name'].unique()):
    # Plot 1: Score distributions
        _data = participant_stats[participant_stats['form_name'] == _form]['mean_score']
        _ax.hist(_data, alpha=0.7, label=_form, bins=10, color=colors[_form], edgecolor='white', linewidth=1)
    _ax.set_xlabel('Mean Score', fontsize=14, fontweight='bold')
    _ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    _ax.set_title('Score Distribution by Form', fontsize=16, fontweight='bold', pad=20)
    _ax.legend(fontsize=12, framealpha=0.9)
    _ax.grid(True, alpha=0.3)
    _ax = _axes[0, 1]
    _form_names = sorted(participant_stats['form_name'].unique())
    _data = [participant_stats[participant_stats['form_name'] == f]['mean_score'].values for f in _form_names]
    # Plot 2: Box plots
    bp = _ax.boxplot(_data, labels=_form_names, patch_artist=True, boxprops=dict(linewidth=2.5), whiskerprops=dict(linewidth=2), capprops=dict(linewidth=2), medianprops=dict(color='#00FF00', linewidth=3))
    for patch, _form in zip(bp['boxes'], _form_names):
        patch.set_facecolor(colors[_form])
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
    _ax.set_ylabel('Mean Score', fontsize=14, fontweight='bold')
    _ax.set_title('Score Comparison by Form', fontsize=16, fontweight='bold', pad=20)
    _ax.grid(True, alpha=0.3, axis='y')
    _ax.tick_params(labelsize=12)
    _ax = _axes[1, 0]
    for _form in sorted(item_stats['form_name'].unique()):
        _d = item_stats[item_stats['form_name'] == _form]
        _ax.scatter(_d['difficulty'], _d['discrimination'], alpha=0.8, label=_form, s=100, color=colors[_form], edgecolors='white', linewidth=1)
    _ax.set_xlabel('Difficulty (Proportion Correct)', fontsize=14, fontweight='bold')
    _ax.set_ylabel('Discrimination (Item-Total r)', fontsize=14, fontweight='bold')
    # Plot 3: Difficulty vs Discrimination
    _ax.axhline(0.2, color='#00FF00', linestyle='--', alpha=0.7, linewidth=3, label='Min acceptable')
    _ax.legend(fontsize=12, framealpha=0.9)
    _ax.set_title('Item Difficulty vs Discrimination', fontsize=16, fontweight='bold', pad=20)
    _ax.grid(True, alpha=0.3)
    _ax.tick_params(labelsize=12)
    _ax = _axes[1, 1]
    difficulty_comparison = item_stats.pivot(index='item_version_name', columns='form_name', values='difficulty')
    x = np.arange(len(difficulty_comparison))
    width = 0.25
    for _i, _form in enumerate(sorted(difficulty_comparison.columns)):
        offset = width * (_i - 1)
        _ax.bar(x + offset, difficulty_comparison[_form], width, label=_form, alpha=0.8, color=colors[_form], edgecolor='white', linewidth=0.8)
    _ax.set_ylabel('Difficulty', fontsize=14, fontweight='bold')
    # Plot 4: Item difficulty by form
    _ax.set_xlabel('Item Index', fontsize=14, fontweight='bold')
    _ax.set_title('Item Difficulty Comparison', fontsize=16, fontweight='bold', pad=20)
    _ax.axhline(0.5, color=muted_grey, linestyle='--', alpha=0.6, linewidth=2)
    _ax.legend(fontsize=12, framealpha=0.9)
    _ax.grid(True, alpha=0.3, axis='y')
    _ax.tick_params(labelsize=11)
    _ax.set_xticks(x[::max(1, len(x) // 10)])
    plt.tight_layout(pad=2)
    plt.savefig('../results/figures/analysis_dark_theme.png', dpi=300, bbox_inches='tight', facecolor='black')
    print('✓ Dark theme figure saved to results/figures/analysis_dark_theme.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Reliability""")
    return


@app.cell
def _(df_test_versioned, np):
    def _cronbach_alpha(_df_wide):
        _n = _df_wide.shape[1]
        if _n < 2:
            return np.nan
        _item_var = _df_wide.var(ddof=1)
        _total_var = _df_wide.sum(axis=1).var(ddof=1)
        return _n / (_n - 1) * (1 - _item_var.sum() / _total_var)
    for _form in df_test_versioned['form_name'].unique():
        _wide = df_test_versioned[df_test_versioned['form_name'] == _form].pivot_table(index='delivery_id', columns='item_name', values='score')
        _alpha = _cronbach_alpha(_wide)
        _n_items = _wide.shape[1]
        _n_participants = _wide.shape[0]
        print(f'{_form}: α = {_alpha:.3f} ({_n_items} items, {_n_participants} participants)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7. Survey Correlations with Test Performance""")
    return


@app.cell
def _(df, participant_stats):
    # Extract survey responses (Likert scale items)
    # These are the actual item_name values from the filtered data
    survey_items_list = [
        'history knowledge',      # General history knowledge
        'ww2',                    # WW2-specific knowledge
        'Quiz Feeling Simple',    # How did the quiz feel (simple version)
        'quiz feeling complex'    # How did the quiz feel (complex version)
    ]

    # Get survey data
    _survey_data = df[df['item_name'].isin(survey_items_list)].copy()

    # Pivot survey data to wide format
    _survey_wide = _survey_data.pivot_table(
        index=['delivery_id', 'form_name'],
        columns='item_name',
        values='score',
        aggfunc='first'
    ).reset_index()

    print(f"Survey responses collected from {len(_survey_wide)} participants")
    print(f"\nSurvey items (Likert scale 0-1, representing 1-5):")
    for _col in _survey_wide.columns:
        if _col not in ['delivery_id', 'form_name']:
            print(f"  - {_col}")

    # Merge with test performance
    survey_with_performance = _survey_wide.merge(
        participant_stats[['delivery_id', 'form_name', 'mean_score', 'total_correct']], 
        on=['delivery_id', 'form_name']
    )

    print(f"\nMerged dataset: {len(survey_with_performance)} participants with both survey and test data")
    survey_with_performance.head()
    return survey_items_list, survey_with_performance


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7.1 Correlation Analysis by Form""")
    return


@app.cell
def _(pd, stats, survey_items_list, survey_with_performance):
    # Calculate correlations for each form
    correlation_results = []
    for _form in sorted(survey_with_performance['form_name'].unique()):
        _form_data = survey_with_performance[survey_with_performance['form_name'] == _form]
        print(f'\n{'=' * 70}')
        print(f'CORRELATIONS FOR {_form}')
        print(f'{'=' * 70}')
        print(f'N = {len(_form_data)} participants\n')
        for _survey_item in survey_items_list:
            if _survey_item in _form_data.columns:
                _valid_data = _form_data[[_survey_item, 'mean_score']].dropna()
                if len(_valid_data) > 2:
                    _corr, p_value = stats.pearsonr(_valid_data[_survey_item], _valid_data['mean_score'])
                    spearman_corr, spearman_p = stats.spearmanr(_valid_data[_survey_item], _valid_data['mean_score'])  # Calculate Pearson correlation
                    correlation_results.append({'form_name': _form, 'survey_item': _survey_item, 'n': len(_valid_data), 'pearson_r': _corr, 'pearson_p': p_value, 'spearman_rho': spearman_corr, 'spearman_p': spearman_p, 'significant': 'Yes' if p_value < 0.05 else 'No'})
                    print(f'{_survey_item}:')
                    print(f'  Pearson r = {_corr:.3f} (p = {p_value:.4f}) {('***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns')}')
                    print(f'  Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.4f})')
                    print(f'  N = {len(_valid_data)}')
                    print()  # Calculate Spearman correlation (for ordinal data)
    corr_df = pd.DataFrame(correlation_results)
    # Create correlation dataframe
    corr_df = corr_df.round({'pearson_r': 3, 'pearson_p': 4, 'spearman_rho': 3, 'spearman_p': 4})
    return (corr_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7.2 Summary Table of Correlations""")
    return


@app.cell
def _(EXPERIMENT_PATH, corr_df):
    # Display comprehensive correlation table
    print("="*100)
    print("CORRELATION SUMMARY: SURVEY RESPONSES vs TEST PERFORMANCE")
    print("="*100)
    print("\nPearson r = Linear correlation")
    print("Spearman ρ = Rank-order correlation (better for ordinal/Likert data)")
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")
    print("\n" + "="*100)

    # Pivot for easier comparison across forms
    print("\n📊 PEARSON CORRELATIONS BY FORM:")
    pearson_pivot = corr_df.pivot(index='survey_item', columns='form_name', values='pearson_r')
    print(pearson_pivot.round(3).to_string())

    print("\n📊 SPEARMAN CORRELATIONS BY FORM:")
    spearman_pivot = corr_df.pivot(index='survey_item', columns='form_name', values='spearman_rho')
    print(spearman_pivot.round(3).to_string())

    print("\n📊 P-VALUES (Pearson) BY FORM:")
    p_value_pivot = corr_df.pivot(index='survey_item', columns='form_name', values='pearson_p')
    print(p_value_pivot.round(4).to_string())

    # Detailed table
    print("\n" + "="*100)
    print("DETAILED CORRELATION TABLE")
    print("="*100)
    display_corr = corr_df[['form_name', 'survey_item', 'n', 'pearson_r', 'pearson_p', 
                             'spearman_rho', 'spearman_p', 'significant']]
    print(display_corr.to_string(index=False))

    # Export correlation results
    corr_path = EXPERIMENT_PATH / 'results' / 'tables' / 'survey_correlations.csv'
    corr_df.to_csv(corr_path, index=False)
    print(f"\n✓ Correlation results saved to: {corr_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 7.3 Visualize Correlations""")
    return


@app.cell
def _(
    EXPERIMENT_PATH,
    light_grey,
    np,
    plt,
    stats,
    survey_items_list,
    survey_with_performance,
):
    # DARK THEME CORRELATION PLOTS
    _fig, _axes = plt.subplots(len(survey_items_list), 2, figsize=(12, 5 * len(survey_items_list)), facecolor='black')
    colors_form = {
        'WW2 Form 1 AI Assist questions': '#00FFFF',  # Cyan
        'WW2 Form 2 GPT questions': '#FF1493'          # Hot Pink
    }
    # Color scheme - NEON colors (matching main plots)
    for _i, _survey_item in enumerate(survey_items_list):  # Cyan, Pink
        for _j, _form in enumerate(sorted(survey_with_performance['form_name'].unique())):
            _ax = _axes[_i, _j] if len(survey_items_list) > 1 else _axes[_j]
            _form_data = survey_with_performance[survey_with_performance['form_name'] == _form]
            _valid_data = _form_data[[_survey_item, 'mean_score']].dropna()
            if len(_valid_data) > 0:
                _ax.scatter(_valid_data[_survey_item], _valid_data['mean_score'], alpha=0.7, s=80, color=colors_form[_form], edgecolors='white', linewidth=1)
                if len(_valid_data) > 2:
                    z = np.polyfit(_valid_data[_survey_item], _valid_data['mean_score'], 1)
                    p = np.poly1d(z)
                    _ax.plot(_valid_data[_survey_item].sort_values(), p(_valid_data[_survey_item].sort_values()), color=colors_form[_form], linestyle='--', alpha=0.9, linewidth=3)  # Scatter plot with dark theme colors
                    _corr, _p_val = stats.pearsonr(_valid_data[_survey_item], _valid_data['mean_score'])
                    sig_marker = '***' if _p_val < 0.001 else '**' if _p_val < 0.01 else '*' if _p_val < 0.05 else 'ns'
                    _ax.text(0.05, 0.95, f'r = {_corr:.3f} {sig_marker}', transform=_ax.transAxes, verticalalignment='top', fontsize=13, fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='black', edgecolor=light_grey, alpha=0.8, linewidth=2))
                _ax.set_xlabel(_survey_item.replace('Survey - ', ''), fontsize=13, fontweight='bold')  # Add trend line
                _ax.set_ylabel('Test Performance (Mean Score)', fontsize=13, fontweight='bold')
                _ax.set_title(f'{_form}', fontsize=15, fontweight='bold', pad=15)
                _ax.grid(True, alpha=0.3)
                _ax.set_ylim(0, 1)
                _ax.tick_params(labelsize=11)
    plt.tight_layout(pad=2)
    corr_fig_path = EXPERIMENT_PATH / 'results' / 'figures' / 'survey_correlations_dark_theme.png'
    plt.savefig(corr_fig_path, dpi=300, bbox_inches='tight', facecolor='black')  # Add correlation coefficient
    print(f'✓ Dark theme correlation plots saved to: {corr_fig_path}')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 6. Visualizations""")
    return


@app.cell
def _(item_stats, participant_stats, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    _ax = _axes[0, 0]
    # Score distributions
    for _form in participant_stats['form_name'].unique():
        _data = participant_stats[participant_stats['form_name'] == _form]['mean_score']
        _ax.hist(_data, alpha=0.6, label=_form, bins=10)
    _ax.set_xlabel('Mean Score')
    _ax.set_title('Score Distribution by Form')
    _ax.legend()
    _ax = _axes[0, 1]
    _form_names = sorted(participant_stats['form_name'].unique())
    # Box plots
    _data = [participant_stats[participant_stats['form_name'] == f]['mean_score'].values for f in _form_names]
    _ax.boxplot(_data, labels=_form_names)
    _ax.set_ylabel('Mean Score')
    _ax.set_title('Score Comparison')
    _ax = _axes[1, 0]
    for _form in item_stats['form_name'].unique():
        _d = item_stats[item_stats['form_name'] == _form]
    # Difficulty vs Discrimination
        _ax.scatter(_d['difficulty'], _d['discrimination'], alpha=0.6, label=_form)
    _ax.set_xlabel('Difficulty')
    _ax.set_ylabel('Discrimination')
    _ax.axhline(0.2, color='r', linestyle='--', alpha=0.3)
    _ax.legend()
    _ax.set_title('Item Quality')
    plt.tight_layout()
    plt.savefig('../results/figures/analysis.png', dpi=300, bbox_inches='tight')
    print('✓ Figure saved')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
