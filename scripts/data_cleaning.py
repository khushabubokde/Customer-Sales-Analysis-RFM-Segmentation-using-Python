import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_raw_data(filepath):
    """Load raw sales data from CSV."""
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print(f"Missing values before cleaning:\n{df.isnull().sum()}\n")
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    print(f"Missing values after cleaning:\n{df.isnull().sum()}\n")
    return df

def remove_duplicates(df):
    """Remove duplicate records."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed} duplicate rows\n")
    return df

def handle_outliers(df, numeric_cols=None):
    """Remove outliers using IQR method."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        if outliers > 0:
            print(f"Found {outliers} outliers in {col}")
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def convert_data_types(df):
    """Convert columns to appropriate data types."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    
    return df

def create_features(df):
    """Create new features from existing data."""
    if 'order_date' in df.columns:
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if 'quantity' in df.columns and 'sales' in df.columns:
        df['price_per_unit'] = df['sales'] / df['quantity']
    
    return df

def clean_data(input_filepath, output_filepath):
    """Complete data cleaning pipeline."""
    print("=" * 50)
    print("Starting Data Cleaning Process")
    print("=" * 50 + "\n")
    
    # Load data
    df = load_raw_data(input_filepath)
    print(f"Original dataset shape: {df.shape}\n")
    
    # Clean data
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = convert_data_types(df)
    df = handle_outliers(df)
    df = create_features(df)
    
    # Save cleaned data
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Cleaned data saved to: {output_filepath}\n")
    
    return df

if __name__ == "__main__":
    input_path = "data/raw/sales_data.csv"
    output_path = "data/processed/cleaned_data.csv"
    
    try:
        df_cleaned = clean_data(input_path, output_path)
        print("Data cleaning completed successfully!")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please ensure the raw data file exists.")
