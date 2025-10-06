"""
Data loading and validation utilities.
"""

import pandas as pd
import json
from pathlib import Path


def load_experiment_data(experiment_path, data_file='responses.csv', data_type='raw'):
    """
    Load experiment data from a standardized structure.
    
    Parameters
    ----------
    experiment_path : str or Path
        Path to the experiment directory
    data_file : str
        Name of the data file to load
    data_type : str
        Either 'raw' or 'processed'
    
    Returns
    -------
    pd.DataFrame
        Loaded data
    dict
        Metadata for the experiment
    """
    exp_path = Path(experiment_path)
    
    # Load metadata
    metadata_path = exp_path / 'data' / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Load data
    data_path = exp_path / 'data' / data_type / data_file
    
    if data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return df, metadata


def validate_long_format(df, required_columns=None):
    """
    Validate that a dataframe is in long format with required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    required_columns : list, optional
        List of required column names
    
    Returns
    -------
    bool
        True if valid
    list
        List of validation issues (empty if valid)
    """
    issues = []
    
    if required_columns is None:
        required_columns = ['participant_id', 'item_id', 'response']
    
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for duplicates (same participant + item)
    if 'participant_id' in df.columns and 'item_id' in df.columns:
        duplicates = df.duplicated(subset=['participant_id', 'item_id'])
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate responses")
    
    # Check for missing values in key columns
    for col in required_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append(f"Column '{col}' has {missing} missing values")
    
    return len(issues) == 0, issues


def load_all_experiments(experiments_dir='experiments'):
    """
    Load metadata for all experiments in the experiments directory.
    
    Parameters
    ----------
    experiments_dir : str or Path
        Path to experiments directory
    
    Returns
    -------
    pd.DataFrame
        Summary of all experiments
    """
    exp_path = Path(experiments_dir)
    experiments = []
    
    for exp_dir in exp_path.iterdir():
        if exp_dir.is_dir():
            metadata_path = exp_dir / 'data' / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata['path'] = str(exp_dir)
                    experiments.append(metadata)
    
    if experiments:
        return pd.DataFrame(experiments)
    else:
        return pd.DataFrame()
