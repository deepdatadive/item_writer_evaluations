"""
Psychometric analysis utilities.
"""

import pandas as pd
import numpy as np
from scipy import stats


def calculate_item_stats(df, item_col='item_id', response_col='response', 
                         correct_col='correct', group_col=None):
    """
    Calculate basic item statistics (difficulty, discrimination, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data in long format
    item_col : str
        Column name for item identifier
    response_col : str
        Column name for response
    correct_col : str
        Column name for correctness (0/1)
    group_col : str, optional
        Column name for grouping (e.g., condition)
    
    Returns
    -------
    pd.DataFrame
        Item statistics
    """
    if group_col:
        grouped = df.groupby([group_col, item_col])
    else:
        grouped = df.groupby(item_col)
    
    stats_dict = {
        'n_responses': grouped.size(),
        'mean_score': grouped[correct_col].mean() if correct_col in df.columns else None,
        'sd_score': grouped[correct_col].std() if correct_col in df.columns else None,
    }
    
    # Remove None values
    stats_dict = {k: v for k, v in stats_dict.items() if v is not None}
    
    item_stats = pd.DataFrame(stats_dict).reset_index()
    
    return item_stats


def calculate_reliability(df, participant_col='participant_id', 
                         item_col='item_id', score_col='correct'):
    """
    Calculate reliability metrics (Cronbach's alpha).
    
    Parameters
    ----------
    df : pd.DataFrame
        Data in long format
    participant_col : str
        Column name for participant identifier
    item_col : str
        Column name for item identifier
    score_col : str
        Column name for score
    
    Returns
    -------
    dict
        Dictionary containing reliability metrics
    """
    # Pivot to wide format
    wide_df = df.pivot(index=participant_col, columns=item_col, values=score_col)
    
    # Calculate Cronbach's alpha
    n_items = wide_df.shape[1]
    item_variances = wide_df.var(axis=0, ddof=1)
    total_variance = wide_df.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    
    return {
        'cronbach_alpha': alpha,
        'n_items': n_items,
        'n_participants': wide_df.shape[0],
        'mean_item_variance': item_variances.mean(),
        'total_variance': total_variance
    }


def calculate_item_total_correlation(df, participant_col='participant_id',
                                     item_col='item_id', score_col='correct'):
    """
    Calculate item-total correlations for discrimination.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data in long format
    participant_col : str
        Column name for participant identifier
    item_col : str
        Column name for item identifier
    score_col : str
        Column name for score
    
    Returns
    -------
    pd.DataFrame
        Item-total correlations
    """
    # Pivot to wide format
    wide_df = df.pivot(index=participant_col, columns=item_col, values=score_col)
    
    # Calculate total score
    total_score = wide_df.sum(axis=1)
    
    # Calculate correlation for each item
    correlations = {}
    for item in wide_df.columns:
        # Corrected item-total (excluding the item itself)
        corrected_total = total_score - wide_df[item]
        corr = wide_df[item].corr(corrected_total)
        correlations[item] = corr
    
    return pd.DataFrame({
        'item_id': list(correlations.keys()),
        'item_total_correlation': list(correlations.values())
    })
