# Retail Sales Data Analysis & Customer Segmentation

A comprehensive data analysis project focusing on retail sales patterns and customer behavior segmentation using Python and machine learning techniques.

## Project Overview

This project analyzes retail sales data to uncover insights about customer behavior, sales trends, and market patterns. It includes customer segmentation using RFM analysis and K-means clustering to identify distinct customer groups for targeted marketing strategies.

## Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of sales patterns, trends, and distributions
- **Customer Segmentation**: RFM analysis and K-means clustering to identify customer segments
- **Sales Forecasting**: Time series analysis and predictive modeling
- **Interactive Visualizations**: Dynamic charts and plots using Plotly and Seaborn
- **Business Insights**: Actionable recommendations based on data findings

## Dataset

The project uses a synthetic retail sales dataset containing:
- Customer demographics and transaction history
- Product categories and pricing information
- Sales dates and seasonal patterns
- Geographic distribution of customers

## Key Analyses

1. **Sales Performance Analysis**
   - Revenue trends over time
   - Product category performance
   - Seasonal patterns and cyclical trends

2. **Customer Behavior Analysis**
   - Purchase frequency and patterns
   - Customer lifetime value (CLV)
   - Churn analysis and retention rates

3. **Customer Segmentation**
   - RFM Analysis (Recency, Frequency, Monetary)
   - K-means clustering for customer groups
   - Segment profiling and characteristics

4. **Market Basket Analysis**
   - Product association rules
   - Cross-selling opportunities
   - Recommendation systems

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning algorithms
- **Jupyter Notebook** - Interactive development environment

## Project Structure

```
retail-sales-analysis/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and processed data
│   └── external/               # External reference data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_sales_analysis.ipynb
│   ├── 04_customer_segmentation.ipynb
│   └── 05_insights_recommendations.ipynb
├── src/
│   ├── data_processing.py      # Data cleaning and preprocessing
│   ├── analysis.py             # Analysis functions
│   ├── visualization.py       # Plotting functions
│   └── segmentation.py         # Customer segmentation algorithms
├── reports/
│   ├── figures/                # Generated plots and charts
│   └── final_report.md         # Executive summary
├── requirements.txt
└── README.md
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/1234-ad/retail-sales-analysis.git
cd retail-sales-analysis
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

1. Start with `01_data_exploration.ipynb` to understand the dataset
2. Run `02_data_cleaning.ipynb` to preprocess the data
3. Analyze sales patterns in `03_sales_analysis.ipynb`
4. Perform customer segmentation in `04_customer_segmentation.ipynb`
5. Review insights in `05_insights_recommendations.ipynb`

## Key Findings

- **Customer Segments**: Identified 4 distinct customer segments with different purchasing behaviors
- **Seasonal Trends**: Peak sales during Q4 with 35% increase in revenue
- **Product Performance**: Electronics and Fashion categories drive 60% of total revenue
- **Customer Retention**: 25% churn rate with opportunities for improvement

## Business Recommendations

1. **Targeted Marketing**: Customize campaigns for each customer segment
2. **Inventory Management**: Optimize stock levels based on seasonal patterns
3. **Customer Retention**: Implement loyalty programs for high-value customers
4. **Cross-selling**: Leverage market basket analysis for product recommendations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -am 'Add new analysis'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please reach out via GitHub issues.