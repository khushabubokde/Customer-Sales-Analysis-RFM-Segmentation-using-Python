import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SalesForecastingModel:
    """Sales forecasting and prediction models."""
    
    def __init__(self, df, target_column='sales'):
        self.df = df
        self.target_column = target_column
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, feature_columns=None):
        """Prepare features for modeling."""
        if feature_columns is None:
            # Select numeric columns except target
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in feature_columns:
                feature_columns.remove(self.target_column)
        
        X = self.df[feature_columns].fillna(self.df[feature_columns].mean())
        y = self.df[self.target_column]
        
        return X, y, feature_columns
    
    def train_linear_regression(self, test_size=0.2):
        """Train linear regression model."""
        X, y, features = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Metrics
        results = {
            'model_type': 'Linear Regression',
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'predictions': y_pred_test,
            'actual': y_test,
            'features': features
        }
        
        return results
    
    def train_random_forest(self, test_size=0.2, n_estimators=100):
        """Train random forest model."""
        X, y, features = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Metrics
        results = {
            'model_type': 'Random Forest',
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'predictions': y_pred_test,
            'actual': y_test,
            'feature_importance': feature_importance
        }
        
        return results

def customer_lifetime_value(df):
    """Calculate Customer Lifetime Value (CLV)."""
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    customer_cols = [col for col in df.columns if 'customer' in col.lower() or 'id' in col.lower()]
    amount_cols = [col for col in df.columns if 'sales' in col.lower() or 'amount' in col.lower()]
    
    if not (date_cols and customer_cols and amount_cols):
        print("Insufficient columns for CLV calculation")
        return None
    
    date_col = date_cols[0]
    customer_col = customer_cols[0]
    amount_col = amount_cols[0]
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Calculate CLV metrics
    clv = df.groupby(customer_col).agg({
        amount_col: 'sum',  # Total spending
        customer_col: 'count'  # Number of purchases
    })
    
    clv.columns = ['total_spending', 'purchase_count']
    clv['avg_order_value'] = clv['total_spending'] / clv['purchase_count']
    clv = clv.sort_values('total_spending', ascending=False)
    
    print("\n" + "=" * 60)
    print("CUSTOMER LIFETIME VALUE ANALYSIS")
    print("=" * 60)
    print(f"\nTop 10 Customers by CLV:")
    print(clv.head(10))
    
    # Visualize CLV distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(clv['total_spending'], bins=30, edgecolor='black', color='steelblue')
    plt.title('Customer Lifetime Value Distribution')
    plt.xlabel('Total Spending ($)')
    plt.ylabel('Number of Customers')
    
    plt.subplot(1, 2, 2)
    plt.scatter(clv['purchase_count'], clv['total_spending'], alpha=0.6, color='steelblue')
    plt.xlabel('Purchase Count')
    plt.ylabel('Total Spending ($)')
    plt.title('Purchase Frequency vs Spending')
    
    plt.tight_layout()
    plt.savefig('visualizations/customer_lifetime_value.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return clv

def print_model_results(results):
    """Print model results."""
    print(f"\n{results['model_type']} Results:")
    print(f"  Train R²: {results['train_r2']:.4f}")
    print(f"  Test R²: {results['test_r2']:.4f}")
    print(f"  Train RMSE: ${results['train_rmse']:.2f}")
    print(f"  Test RMSE: ${results['test_rmse']:.2f}")
    print(f"  Train MAE: ${results['train_mae']:.2f}")
    print(f"  Test MAE: ${results['test_mae']:.2f}")
    
    if 'feature_importance' in results:
        print(f"\n  Top 5 Important Features:")
        print(results['feature_importance'].head())

def predictive_modeling(filepath):
    """Run predictive modeling pipeline."""
    print("\n" + "=" * 60)
    print("PREDICTIVE MODELING")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    
    # Check for sales/amount column
    amount_cols = [col for col in df.columns if 'sales' in col.lower() or 'amount' in col.lower()]
    if not amount_cols:
        print("No sales or amount column found for modeling")
        return
    
    target_col = amount_cols[0]
    
    # Initialize models
    forecaster = SalesForecastingModel(df, target_column=target_col)
    
    # Train models
    results_lr = forecaster.train_linear_regression()
    results_rf = forecaster.train_random_forest()
    
    print_model_results(results_lr)
    print_model_results(results_rf)
    
    # Calculate CLV
    customer_lifetime_value(df)
    
    # Visualize predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear Regression
    axes[0].scatter(results_lr['actual'], results_lr['predictions'], alpha=0.6, color='steelblue')
    axes[0].plot([results_lr['actual'].min(), results_lr['actual'].max()], 
                 [results_lr['actual'].min(), results_lr['actual'].max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f"Linear Regression (R² = {results_lr['test_r2']:.3f})")
    
    # Random Forest
    axes[1].scatter(results_rf['actual'], results_rf['predictions'], alpha=0.6, color='steelblue')
    axes[1].plot([results_rf['actual'].min(), results_rf['actual'].max()], 
                 [results_rf['actual'].min(), results_rf['actual'].max()], 
                 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f"Random Forest (R² = {results_rf['test_r2']:.3f})")
    
    plt.tight_layout()
    plt.savefig('visualizations/model_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 60)
    print("Modeling completed! Results saved to 'visualizations/' folder")
    print("=" * 60)

if __name__ == "__main__":
    filepath = "data/processed/cleaned_data.csv"
    try:
        predictive_modeling(filepath)
    except Exception as e:
        print(f"Error during modeling: {str(e)}")
