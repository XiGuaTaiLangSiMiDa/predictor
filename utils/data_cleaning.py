import numpy as np
import pandas as pd

def clean_dataframe(df):
    """Clean the dataframe by handling NaN values and outliers"""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Forward fill NaN values
    df = df.ffill()
    # Backward fill any remaining NaN values at the beginning
    df = df.bfill()
    
    # Handle outliers using IQR method for each numeric column
    df = handle_outliers(df)
    
    return df

def validate_data(df, columns=None):
    """Validate data quality and print statistics"""
    if columns is None:
        columns = df.columns
    
    print("\nData Quality Check:")
    print(f"Number of rows: {len(df)}")
    
    for col in columns:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"NaN values: {df[col].isna().sum()}")
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            print(f"Unique values: {df[col].nunique()}")

def handle_outliers(df, columns=None, method='iqr'):
    """Handle outliers in specified columns"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df = df.copy()
    
    if method == 'iqr':
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    elif method == 'zscore':
        for column in columns:
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            df[column] = df[column].mask(abs(z_scores) > 3, df[column].median())
    
    return df

def check_data_quality(df):
    """Perform comprehensive data quality check"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isna().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'numeric_stats': {}
    }
    
    # Calculate statistics for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        quality_report['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'zeros': (df[col] == 0).sum(),
            'negative_values': (df[col] < 0).sum()
        }
    
    return quality_report

def print_data_quality_report(report):
    """Print data quality report in a readable format"""
    print("\nData Quality Report")
    print("===================")
    print(f"\nDataset Size:")
    print(f"Rows: {report['total_rows']}")
    print(f"Columns: {report['total_columns']}")
    print(f"Duplicate rows: {report['duplicates']}")
    
    print("\nMissing Values:")
    for col, missing in report['missing_values'].items():
        if missing > 0:
            print(f"{col}: {missing} ({(missing/report['total_rows'])*100:.2f}%)")
    
    print("\nNumeric Column Statistics:")
    for col, stats in report['numeric_stats'].items():
        print(f"\n{col}:")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Zero values: {stats['zeros']}")
        print(f"  Negative values: {stats['negative_values']}")

# Export all functions
__all__ = [
    'clean_dataframe',
    'validate_data',
    'handle_outliers',
    'check_data_quality',
    'print_data_quality_report'
]
