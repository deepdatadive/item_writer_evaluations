"""
Plotting utilities and style configurations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plotting_style(style='seaborn-v0_8-darkgrid', context='notebook', 
                         palette='colorblind'):
    """
    Set up consistent plotting style across notebooks.
    
    Parameters
    ----------
    style : str
        Matplotlib style
    context : str
        Seaborn context (paper, notebook, talk, poster)
    palette : str
        Color palette name
    """
    try:
        plt.style.use(style)
    except:
        # Fallback if style not available
        sns.set_theme()
    
    sns.set_context(context)
    sns.set_palette(palette)
    
    # Set default figure size
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100


def save_figure(fig, filename, output_dir, formats=['png', 'pdf'], dpi=300):
    """
    Save figure in multiple formats with consistent naming.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    output_dir : str or Path
        Directory to save figures
    formats : list
        List of file formats to save
    dpi : int
        Resolution for raster formats
    
    Returns
    -------
    list
        Paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        saved_files.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_files


def plot_item_difficulty(item_stats, condition_col=None, ax=None):
    """
    Create a bar plot of item difficulty.
    
    Parameters
    ----------
    item_stats : pd.DataFrame
        Item statistics dataframe
    condition_col : str, optional
        Column name for conditions (for grouped bars)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if condition_col and condition_col in item_stats.columns:
        item_stats_pivot = item_stats.pivot(index='item_id', 
                                             columns=condition_col, 
                                             values='mean_score')
        item_stats_pivot.plot(kind='bar', ax=ax)
        ax.set_ylabel('Mean Score (Difficulty)')
        ax.legend(title=condition_col)
    else:
        ax.bar(range(len(item_stats)), item_stats['mean_score'])
        ax.set_xticks(range(len(item_stats)))
        ax.set_xticklabels(item_stats['item_id'], rotation=45, ha='right')
        ax.set_ylabel('Mean Score (Difficulty)')
    
    ax.set_xlabel('Item')
    ax.set_title('Item Difficulty by Item')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% difficulty')
    
    return ax
