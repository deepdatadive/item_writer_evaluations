import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Processing Block Size Analysis

    This notebook analyzes the impact of processing block size on item quality and psychometric properties.

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
            return None
    # Generate the report
    # Uncomment the options you want:
    # Basic HTML report (code visible)
    # generate_report(output_format='html')
    # HTML report with code hidden (output only)
    # Re-execute everything and generate report (like knittr)
    # generate_report(output_format='html', execute=True)
    # PDF report (requires LaTeX)
    # generate_report(output_format='pdf')
    # PDF via HTML (no LaTeX required, needs chromium)
    # generate_report(output_format='webpdf')
    generate_report(output_format='html', hide_code=True)  # Add options for better HTML output  # Embed images in HTML  # Use modern template
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
    mo.md(
        r"""
    ## 1. Load Data

    """
    )
    return


@app.cell
def _(Path, pd):
    EXPERIMENT_PATH = Path('..')
    DATA_FILE = 'ww2_processing_block_size.txt'

    data_path = EXPERIMENT_PATH / 'data' / 'raw' / DATA_FILE
    df = pd.read_csv(data_path, sep='\t')

    print(f"Loaded {len(df)} rows")
    print(f"Participants: {df['delivery_id'].nunique()}")
    print(f"Forms: {df['form_name'].unique()}")
    df.head()
    return EXPERIMENT_PATH, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Filter Survey Questions

    """
    )
    return


@app.cell
def _(df):
    survey_items = ['Survey - History', 'Survey - History Knowledge', 'Survey - General History Knowledge', 'Survey - Completion Code']
    df_test = df[~df['item_name'].isin(survey_items)].copy()
    print(f'Before filtering: {len(df)} rows')
    print(f'After removing surveys: {len(df_test)} rows')
    erroneous_item = (df_test['form_name'] == '1000_chat') & (df_test['item_version_name'] == '500_8_US Economic Dominance Post-WWII_v1')
    affected_participants = df_test[erroneous_item]['delivery_id'].unique()
    print(f'\n🔍 IDENTIFYING ERRONEOUS ITEM:')
    # Filter using item_name for surveys
    print(f'Item: 500_8_US Economic Dominance Post-WWII_v1')
    print(f'Should NOT appear in: 1000_chat form')
    print(f'\nParticipants who received this item: {len(affected_participants)}')
    print('\nDelivery IDs:')
    for _i, delivery_id in enumerate(sorted(affected_participants), 1):
    # Identify participants with the erroneous 500_8 item in 1000_chat
        print(f'{_i:3d}. {delivery_id}')
    print(f'\n{'=' * 70}')
    print('REMOVING ERRONEOUS ITEM...')
    print(f'{'=' * 70}')
    before_fix = len(df_test)
    df_test = df_test[~erroneous_item].copy()
    rows_removed = before_fix - len(df_test)
    print(f'Removed {rows_removed} rows from dataset')
    print(f'\nFinal dataset: {len(df_test)} rows')
    print(f'Unique items (by item_version_name): {df_test['item_version_name'].nunique()}')
    # Remove the erroneous 500_8 item from 1000_chat form
    print(f'Participants: {df_test['delivery_id'].nunique()}')
    return (df_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2.5 Check Items by Form

    """
    )
    return


@app.cell
def _(df_test):
    # Check which items are in each form
    for _form in sorted(df_test['form_name'].unique()):
        items = df_test[df_test['form_name'] == _form]['item_version_name'].unique()
        print(f'\n{_form} ({len(items)} items):')
        print('=' * 70)
        for _i, _item in enumerate(sorted(items), 1):
            print(f'{_i:2d}. {_item}')
    all_forms = df_test['form_name'].unique()
    # Find items that appear in one form but not others
    print(f'\n\n{'=' * 70}')
    print('ITEM DIFFERENCES BETWEEN FORMS:')
    print('=' * 70)
    for _form in all_forms:
        items_in_form = set(df_test[df_test['form_name'] == _form]['item_version_name'].unique())
        other_forms = [f for f in all_forms if f != _form]
        for other_form in other_forms:
            items_in_other = set(df_test[df_test['form_name'] == other_form]['item_version_name'].unique())
            only_in_form = items_in_form - items_in_other
            if only_in_form:
                print(f'\nItems ONLY in {_form} (not in {other_form}):')
                for _item in sorted(only_in_form):
                    print(f'  - {_item}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Descriptive Statistics by Form

    """
    )
    return


@app.cell
def _(df_test, stats):
    # Calculate participant-level statistics
    participant_stats = df_test.groupby(['delivery_id', 'form_name']).agg({'score': ['mean', 'sum', 'count'], 'total_seconds': 'sum'}).reset_index()
    participant_stats.columns = ['delivery_id', 'form_name', 'mean_score', 'total_correct', 'n_items', 'total_time']
    form_summary = participant_stats.groupby('form_name').agg({'mean_score': ['mean', 'std', 'min', 'max'], 'total_correct': ['mean', 'std'], 'n_items': 'first', 'total_time': ['mean', 'std'], 'delivery_id': 'count'}).round(3)
    form_summary.columns = ['_'.join(col) for col in form_summary.columns]
    form_summary.rename(columns={'delivery_id_count': 'n_participants'}, inplace=True)
    print('PERFORMANCE BY FORM:')
    print(form_summary)
    # Summary by form
    print('\nCOMPARISONS:')
    forms = participant_stats['form_name'].unique()
    for _i, form1 in enumerate(forms):
        for form2 in forms[_i + 1:]:
            g1 = participant_stats[participant_stats['form_name'] == form1]['mean_score']
            g2 = participant_stats[participant_stats['form_name'] == form2]['mean_score']
            t_stat, _p_val = stats.ttest_ind(g1, g2)
    # Compare all forms
            print(f'{form1} vs {form2}: p={_p_val:.4f} {('*' if _p_val < 0.05 else '')}')
    return (participant_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Item Analysis

    """
    )
    return


@app.cell
def _(df_test, pd):
    # Item statistics - use item_version_name for actual items
    item_stats = df_test.groupby(['item_version_name', 'form_name']).agg({'score': ['mean', 'std', 'count'], 'total_seconds': 'mean'}).reset_index()
    item_stats.columns = ['item_version_name', 'form_name', 'difficulty', 'sd', 'n', 'time']
    discriminations = []
    for _form in df_test['form_name'].unique():
        _wide = df_test[df_test['form_name'] == _form].pivot_table(index='delivery_id', columns='item_version_name', values='score')
        for _item in _wide.columns:
            _corr = _wide[_item].corr(_wide.sum(axis=1) - _wide[_item])
    # Calculate discrimination
            discriminations.append({'form_name': _form, 'item_version_name': _item, 'discrimination': _corr})
    disc_df = pd.DataFrame(discriminations)
    item_stats = item_stats.merge(disc_df, on=['form_name', 'item_version_name'])
    item_stats['flag'] = ''
    item_stats.loc[item_stats['discrimination'] < 0.15, 'flag'] += 'Low_Disc '
    item_stats.loc[item_stats['difficulty'] < 0.2, 'flag'] += 'Hard '
    item_stats.loc[item_stats['difficulty'] > 0.9, 'flag'] += 'Easy '
    item_stats['item_short'] = item_stats['item_version_name'].str.replace('_v1', '').str.replace('100_', '').str.replace('500_', '')
    item_stats['difficulty'] = item_stats['difficulty'].round(3)
    item_stats['discrimination'] = item_stats['discrimination'].round(3)
    item_stats['time'] = item_stats['time'].round(1)
    print(f'Total items analyzed: {len(item_stats)}')
    # Flag problematic items
    print(f'Problematic items flagged: {(item_stats['flag'] != '').sum()}')
    print('\n' + '=' * 100)
    print('ITEM STATISTICS (sorted by form, then difficulty)')
    print('=' * 100)
    display_cols = ['form_name', 'item_short', 'difficulty', 'discrimination', 'n', 'time', 'flag']
    # Add simplified item name for easier viewing
    item_stats_display = item_stats[display_cols].sort_values(['form_name', 'difficulty'])
    pd.set_option('display.max_rows', None)
    # Round for cleaner display
    pd.set_option('display.max_colwidth', 60)
    pd.set_option('display.width', None)
    print(item_stats_display.to_string(index=False))
    # Display sorted by form and difficulty
    # Set pandas display options for full table
    # Reset display options
    pd.set_option('display.max_rows', 100)
    return display_cols, item_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4.5 Interactive Searchable Table

    """
    )
    return


@app.cell
def _(display, display_cols, item_stats):
    # Interactive searchable table - use Ctrl+F to search in output
    # Style the dataframe with color coding
    def highlight_issues(row):
        colors = []
        for col in row.index:
            if col == 'difficulty':
                if row[col] < 0.2:
                    colors.append('background-color: #ffcccc')  # Red for too hard
                elif row[col] > 0.9:
                    colors.append('background-color: #ffffcc')  # Yellow for too easy
                elif 0.3 <= row[col] <= 0.7:
                    colors.append('background-color: #ccffcc')  # Green for good
                else:
                    colors.append('')
            elif col == 'discrimination':
                if row[col] < 0.15:
                    colors.append('background-color: #ffcccc')  # Red for poor
                elif row[col] >= 0.3:
                    colors.append('background-color: #ccffcc')  # Green for good
                else:
                    colors.append('')
            elif col == 'flag' and row[col] != '':
                colors.append('background-color: #ffcccc')  # Red if flagged
            else:
                colors.append('')
        return colors

    # Create styled version
    styled_table = item_stats[display_cols].sort_values(['form_name', 'item_short']).style\
        .apply(highlight_issues, axis=1)\
        .format({'difficulty': '{:.3f}', 'discrimination': '{:.3f}', 'time': '{:.1f}'})\
        .set_caption("Item Statistics by Form (Color coded: Green=Good, Yellow=Easy, Red=Problem)")\
        .set_properties(**{'text-align': 'left'})\
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold')]},
            {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]}
        ])

    display(styled_table)

    print("\n💡 TIP: Use Ctrl+F (or Cmd+F on Mac) to search for specific items in the table above!")
    print("💡 You can also filter by form or search for specific topics in item names.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4.6 Export to Excel for Easy Filtering

    """
    )
    return


@app.cell
def _(EXPERIMENT_PATH, item_stats, pd):
    # Export to Excel with formatting for easy filtering and searching
    excel_path = EXPERIMENT_PATH / 'results' / 'tables' / 'item_statistics_by_form.xlsx'
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for export
    export_data = item_stats[['form_name', 'item_version_name', 'item_short', 'difficulty', 
                              'discrimination', 'sd', 'n', 'time', 'flag']].copy()
    export_data = export_data.sort_values(['form_name', 'item_short'])

    # Write to Excel
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        export_data.to_excel(writer, sheet_name='Item Statistics', index=False)
    
        # Get the worksheet
        worksheet = writer.sheets['Item Statistics']
    
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

    print(f"✓ Exported to Excel: {excel_path}")
    print(f"\n💡 Open in Excel/Google Sheets for:")
    print(f"   - Column filtering")
    print(f"   - Sorting by any column")
    print(f"   - Advanced search and filtering")
    print(f"   - Pivot tables")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4.7 Item Quality Summary by Form

    """
    )
    return


@app.cell
def _(EXPERIMENT_PATH, item_stats, pd):
    # Create summary of item quality issues by form
    quality_summary = []
    for _form in sorted(item_stats['form_name'].unique()):
        form_items = item_stats[item_stats['form_name'] == _form]
        summary = {'form_name': _form, 'total_items': len(form_items), 'too_hard': (form_items['difficulty'] < 0.2).sum(), 'hard': ((form_items['difficulty'] >= 0.2) & (form_items['difficulty'] < 0.3)).sum(), 'good_difficulty': ((form_items['difficulty'] >= 0.3) & (form_items['difficulty'] <= 0.7)).sum(), 'easy': ((form_items['difficulty'] > 0.7) & (form_items['difficulty'] <= 0.9)).sum(), 'too_easy': (form_items['difficulty'] > 0.9).sum(), 'poor_disc': (form_items['discrimination'] < 0.15).sum(), 'low_disc': ((form_items['discrimination'] >= 0.15) & (form_items['discrimination'] < 0.2)).sum(), 'acceptable_disc': ((form_items['discrimination'] >= 0.2) & (form_items['discrimination'] < 0.3)).sum(), 'good_disc': (form_items['discrimination'] >= 0.3).sum(), 'any_issues': (form_items['flag'] != '').sum(), 'no_issues': (form_items['flag'] == '').sum(), 'mean_difficulty': form_items['difficulty'].mean(), 'mean_discrimination': form_items['discrimination'].mean()}
        quality_summary.append(summary)
    quality_df = pd.DataFrame(quality_summary)
    print('=' * 100)
    print('ITEM QUALITY SUMMARY BY FORM')
    print('=' * 100)
    print('\n📊 DIFFICULTY DISTRIBUTION:')  # Difficulty issues
    print(quality_df[['form_name', 'too_hard', 'hard', 'good_difficulty', 'easy', 'too_easy']].to_string(index=False))
    print('\n📈 DISCRIMINATION DISTRIBUTION:')
    print(quality_df[['form_name', 'poor_disc', 'low_disc', 'acceptable_disc', 'good_disc']].to_string(index=False))
    print('\n✅ OVERALL QUALITY:')
    print(quality_df[['form_name', 'total_items', 'no_issues', 'any_issues', 'mean_difficulty', 'mean_discrimination']].round(3).to_string(index=False))
    print('\n📊 PERCENTAGE OF ITEMS WITH ISSUES:')
    for _form in sorted(quality_df['form_name']):  # Discrimination issues
        row = quality_df[quality_df['form_name'] == _form].iloc[0]
        pct_issues = row['any_issues'] / row['total_items'] * 100
        pct_poor_disc = row['poor_disc'] / row['total_items'] * 100
        pct_difficulty_issues = (row['too_hard'] + row['too_easy']) / row['total_items'] * 100
        print(f'\n{_form}:')
        print(f'  Items with ANY issues: {row['any_issues']}/{row['total_items']} ({pct_issues:.1f}%)')  # Overall quality
        print(f'  Poor discrimination: {row['poor_disc']}/{row['total_items']} ({pct_poor_disc:.1f}%)')
        print(f'  Difficulty issues: {row['too_hard'] + row['too_easy']}/{row['total_items']} ({pct_difficulty_issues:.1f}%)')
    summary_path = EXPERIMENT_PATH / 'results' / 'tables' / 'quality_summary_by_form.csv'
    quality_df.to_csv(summary_path, index=False)  # Average stats
    # Display summary
    # Calculate percentages for better comparison
    # Save summary to file
    print(f'\n✓ Summary saved to: {summary_path}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. UPDATED Visualizations (Dark Theme for Presentation)

    """
    )
    return


@app.cell
def _(item_stats, muted_grey, np, participant_stats, plt):
    # DARK THEME VISUALIZATIONS - Use these for your presentation!
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
    colors = {'100_chat': '#00FFFF', '500_chat': '#FF1493', '1000_chat': '#FF6600'}
    # Color scheme - NEON colors for dark background (highly distinct!)
    _ax = _axes[0, 0]  # Cyan, Pink, Orange
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
    mo.md(
        r"""
    ## 5. Reliability

    """
    )
    return


@app.cell
def _(df_test, np):
    def cronbach_alpha(df_wide):
        n = df_wide.shape[1]
        if n < 2:
            return np.nan
        item_var = df_wide.var(ddof=1)
        total_var = df_wide.sum(axis=1).var(ddof=1)
        return n / (n - 1) * (1 - item_var.sum() / total_var)
    for _form in df_test['form_name'].unique():
        _wide = df_test[df_test['form_name'] == _form].pivot_table(index='delivery_id', columns='item_version_name', values='score')
        alpha = cronbach_alpha(_wide)
        n_items = _wide.shape[1]
        n_participants = _wide.shape[0]
        print(f'{_form}: α = {alpha:.3f} ({n_items} items, {n_participants} participants)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7. Survey Correlations with Test Performance

    """
    )
    return


@app.cell
def _(df, participant_stats):
    # Extract survey responses (Likert scale items)
    survey_items_list = [
        'Survey - History',
        'Survey - History Knowledge',
        'Survey - General History Knowledge'
    ]

    # Get survey data
    df_survey = df[df['item_name'].isin(survey_items_list)].copy()

    # Pivot survey data to wide format
    survey_wide = df_survey.pivot_table(
        index=['delivery_id', 'form_name'],
        columns='item_name',
        values='score',
        aggfunc='first'
    ).reset_index()

    print(f"Survey responses collected from {len(survey_wide)} participants")
    print(f"\nSurvey items (Likert scale 0-1, representing 1-5):")
    for col in survey_wide.columns:
        if col not in ['delivery_id', 'form_name']:
            print(f"  - {col}")

    # Merge with test performance
    survey_with_performance = survey_wide.merge(
        participant_stats[['delivery_id', 'form_name', 'mean_score', 'total_correct']], 
        on=['delivery_id', 'form_name']
    )

    print(f"\nMerged dataset: {len(survey_with_performance)} participants with both survey and test data")
    survey_with_performance.head()
    return survey_items_list, survey_with_performance


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 7.1 Correlation Analysis by Form

    """
    )
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
    mo.md(
        r"""
    ## 7.2 Summary Table of Correlations

    """
    )
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
    mo.md(
        r"""
    ## 7.3 Visualize Correlations

    """
    )
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
    _fig, _axes = plt.subplots(len(survey_items_list), 3, figsize=(16, 5 * len(survey_items_list)), facecolor='black')
    colors_form = {'100_chat': '#00FFFF', '500_chat': '#FF1493', '1000_chat': '#FF6600'}
    # Color scheme - NEON colors (matching main plots)
    for _i, _survey_item in enumerate(survey_items_list):  # Cyan, Pink, Orange
        for j, _form in enumerate(sorted(survey_with_performance['form_name'].unique())):
            _ax = _axes[_i, j] if len(survey_items_list) > 1 else _axes[j]
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
    mo.md(
        r"""
    ## 6. Visualizations

    """
    )
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
