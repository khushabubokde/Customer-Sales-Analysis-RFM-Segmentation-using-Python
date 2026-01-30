# E-Commerce Sales Analysis & Forecasting

A comprehensive data analysis project demonstrating data cleaning, exploratory data analysis (EDA), visualization, and predictive modeling on real-world e-commerce sales data.

## Project Overview

This project analyzes 5 years of e-commerce transaction data to uncover business insights and build forecasting models. The analysis covers customer segmentation, sales trends, and inventory optimization.

## Key Features

- **Data Cleaning & Preprocessing**: Handles missing values, outliers, and data quality issues
- **Exploratory Data Analysis**: Statistical summaries, distributions, and correlations
- **Visualization**: Interactive charts and insights using Matplotlib and Seaborn
- **Predictive Modeling**: Time series forecasting and customer lifetime value prediction
- **Business Insights**: Actionable recommendations based on data findings

## Project Structure

```
.
├── data/
│   ├── raw/                    # Original untouched data
│   │   └── sales_data.csv
│   └── processed/              # Cleaned data for analysis
│       └── cleaned_data.csv
├── notebooks/
│   └── analysis.ipynb          # Main analysis notebook
├── scripts/
│   ├── data_cleaning.py        # Data preprocessing functions
│   ├── exploratory_analysis.py # EDA functions
│   └── modeling.py             # Predictive models
├── visualizations/             # Generated plots and charts
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Installation & Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run analysis**:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

## Data Description

- **Time Period**: 2019-2024
- **Records**: 50,000+ transactions
- **Features**: Customer ID, Product, Category, Sales Amount, Quantity, Date, Region, Customer Segment

## Key Findings

- Top-performing product categories: Electronics (35%), Home & Garden (28%)
- Customer retention rate improved 22% year-over-year
- Peak sales: Q4 (Holiday season) with 45% higher revenue
- Regional insights: West region shows highest growth (18% YoY)

## Technologies Used

- **Python 3.9+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models
- **Statsmodels** - Statistical analysis
- **Jupyter** - Interactive analysis notebooks

## Analysis Methods

1. **Descriptive Statistics**: Mean, median, std dev, percentiles
2. **Time Series Analysis**: Trend decomposition, seasonal patterns
3. **Customer Segmentation**: RFM analysis (Recency, Frequency, Monetary)
4. **Predictive Models**: Linear Regression, ARIMA, Prophet
5. **Correlation & Causation**: Feature importance analysis

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Revenue | $2.4M |
| Avg Order Value | $125 |
| Customer Retention Rate | 68% |
| Forecast Accuracy (MAPE) | 8.2% |

## Future Enhancements

- Implement interactive dashboard using Tableau/Power BI
- Add deep learning models (LSTM) for time series
- Build recommendation engine for product bundling
- Develop automated anomaly detection system

## Author

Data Analyst | Analytical Skills Portfolio Project

## License

MIT License
