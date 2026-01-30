import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def describe_data(df):
    """Generate comprehensive descriptive statistics."""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    return df.describe()

def correlation_analysis(df):
    """Analyze correlations between numeric variables."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    print(f"\nCorrelation Matrix:\n{correlation_matrix}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation_matrix

def distribution_analysis(df):
    """Analyze distributions of numeric variables."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create subplots for each numeric column
    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 4*len(numeric_cols)))
    
    if len(numeric_cols) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, col in enumerate(numeric_cols):
        # Histogram
        axes[idx, 0].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx, 0].set_title(f'Distribution of {col}')
        axes[idx, 0].set_xlabel(col)
        axes[idx, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[idx, 1].boxplot(df[col])
        axes[idx, 1].set_title(f'Box Plot of {col}')
        axes[idx, 1].set_ylabel(col)
        
        # Print statistics
        skewness = stats.skew(df[col].dropna())
        kurtosis = stats.kurtosis(df[col].dropna())
        print(f"\n{col}:")
        print(f"  Skewness: {skewness:.3f}")
        print(f"  Kurtosis: {kurtosis:.3f}")
    
    plt.tight_layout()
    plt.savefig('visualizations/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def category_analysis(df):
    """Analyze categorical variables."""
    print("\n" + "=" * 60)
    print("CATEGORICAL ANALYSIS")
    print("=" * 60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Value counts:\n{df[col].value_counts().head(10)}")

def time_series_analysis(df):
    """Analyze time-based patterns if date column exists."""
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    
    if not date_cols:
        print("\nNo date column found for time series analysis")
        return
    
    print("\n" + "=" * 60)
    print("TIME SERIES ANALYSIS")
    print("=" * 60)
    
    date_col = date_cols[0]
    
    if 'sales' in df.columns or 'amount' in df.columns:
        amount_col = 'sales' if 'sales' in df.columns else 'amount'
        
        # Monthly trends
        df[date_col] = pd.to_datetime(df[date_col])
        monthly_sales = df.groupby(df[date_col].dt.to_period('M'))[amount_col].sum()
        
        plt.figure(figsize=(14, 6))
        monthly_sales.plot(kind='line', marker='o', color='steelblue', linewidth=2)
        plt.title(f'Monthly {amount_col.capitalize()} Trend')
        plt.xlabel('Month')
        plt.ylabel(f'{amount_col.capitalize()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/time_series_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n{amount_col.capitalize()} by Month:\n{monthly_sales}")

def rfm_analysis(df):
    """Perform RFM (Recency, Frequency, Monetary) analysis."""
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    customer_cols = [col for col in df.columns if 'customer' in col.lower() or 'id' in col.lower()]
    amount_cols = [col for col in df.columns if 'sales' in col.lower() or 'amount' in col.lower()]
    
    if not (date_cols and customer_cols and amount_cols):
        print("\nInsufficient columns for RFM analysis")
        return None
    
    print("\n" + "=" * 60)
    print("RFM ANALYSIS")
    print("=" * 60)
    
    date_col = date_cols[0]
    customer_col = customer_cols[0]
    amount_col = amount_cols[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    current_date = df[date_col].max()
    
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (current_date - x.max()).days,  # Recency
        customer_col: 'count',  # Frequency
        amount_col: 'sum'  # Monetary
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm[rfm['Monetary'] > 0]
    
    print(f"\nRFM Summary:\n{rfm.describe()}")
    
    # Visualize RFM
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(rfm['Recency'], bins=30, edgecolor='black', color='steelblue')
    axes[0].set_title('Recency Distribution')
    axes[0].set_xlabel('Days Since Purchase')
    
    axes[1].hist(rfm['Frequency'], bins=30, edgecolor='black', color='steelblue')
    axes[1].set_title('Frequency Distribution')
    axes[1].set_xlabel('Number of Purchases')
    
    axes[2].hist(rfm['Monetary'], bins=30, edgecolor='black', color='steelblue')
    axes[2].set_title('Monetary Distribution')
    axes[2].set_xlabel('Total Spending')
    
    plt.tight_layout()
    plt.savefig('visualizations/rfm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rfm

def exploratory_analysis(filepath):
    """Run complete exploratory analysis."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    describe_data(df)
    correlation_analysis(df)
    distribution_analysis(df)
    category_analysis(df)
    time_series_analysis(df)
    rfm_analysis(df)
    
    print("\n" + "=" * 60)
    print("Analysis completed! Visualizations saved to 'visualizations/' folder")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    filepath = "data/processed/cleaned_data.csv"
    try:
        df = exploratory_analysis(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
