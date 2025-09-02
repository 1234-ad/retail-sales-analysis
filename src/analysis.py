"""
Analysis Module for Retail Sales Analysis
Comprehensive statistical analysis and business insights
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RetailAnalyzer:
    """
    Comprehensive analysis class for retail sales data
    """
    
    def __init__(self, data):
        self.data = data
        self.insights = {}
        
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        stats_summary = {}
        
        # Numerical columns analysis
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        stats_summary['numerical'] = self.data[numerical_cols].describe()
        
        # Categorical columns analysis
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        stats_summary['categorical'] = {}
        
        for col in categorical_cols:
            stats_summary['categorical'][col] = {
                'unique_values': self.data[col].nunique(),
                'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                'frequency': self.data[col].value_counts().head()
            }
        
        return stats_summary
    
    def sales_performance_analysis(self):
        """Analyze sales performance metrics"""
        performance = {}
        
        # Overall metrics
        performance['total_revenue'] = self.data['total_amount'].sum()
        performance['total_transactions'] = len(self.data)
        performance['unique_customers'] = self.data['customer_id'].nunique()
        performance['avg_order_value'] = self.data['total_amount'].mean()
        performance['median_order_value'] = self.data['total_amount'].median()
        
        # Time-based analysis
        daily_sales = self.data.groupby(self.data['transaction_date'].dt.date)['total_amount'].sum()
        performance['avg_daily_sales'] = daily_sales.mean()
        performance['peak_sales_day'] = daily_sales.idxmax()
        performance['peak_sales_amount'] = daily_sales.max()
        
        # Monthly trends
        monthly_sales = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['total_amount'].sum()
        performance['monthly_growth_rate'] = monthly_sales.pct_change().mean() * 100
        performance['best_month'] = monthly_sales.idxmax()
        performance['worst_month'] = monthly_sales.idxmin()
        
        # Category performance
        category_performance = self.data.groupby('category').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        category_performance.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 'Unique_Customers']
        performance['category_analysis'] = category_performance
        
        return performance
    
    def customer_analysis(self):
        """Comprehensive customer behavior analysis"""
        customer_metrics = {}
        
        # Customer-level aggregation
        customer_summary = self.data.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'transaction_date': ['min', 'max'],
            'age': 'first',
            'gender': 'first',
            'city': 'first'
        })
        
        # Flatten column names
        customer_summary.columns = ['Total_Spent', 'Avg_Order_Value', 'Purchase_Frequency', 
                                  'First_Purchase', 'Last_Purchase', 'Age', 'Gender', 'City']
        
        # Calculate customer lifetime value
        customer_summary['Customer_Lifetime_Days'] = (customer_summary['Last_Purchase'] - 
                                                     customer_summary['First_Purchase']).dt.days
        
        # Customer segmentation by value
        customer_summary['Value_Segment'] = pd.qcut(customer_summary['Total_Spent'], 
                                                   q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Customer metrics
        customer_metrics['total_customers'] = len(customer_summary)
        customer_metrics['avg_customer_value'] = customer_summary['Total_Spent'].mean()
        customer_metrics['avg_purchase_frequency'] = customer_summary['Purchase_Frequency'].mean()
        customer_metrics['customer_retention_days'] = customer_summary['Customer_Lifetime_Days'].mean()
        
        # Demographic analysis
        customer_metrics['age_analysis'] = {
            'avg_age': customer_summary['Age'].mean(),
            'age_distribution': customer_summary['Age'].describe()
        }
        
        customer_metrics['gender_distribution'] = customer_summary['Gender'].value_counts(normalize=True)
        customer_metrics['city_distribution'] = customer_summary['City'].value_counts().head()
        
        # Value segment analysis
        value_segment_analysis = customer_summary.groupby('Value_Segment').agg({
            'Total_Spent': ['mean', 'sum'],
            'Purchase_Frequency': 'mean',
            'Age': 'mean'
        }).round(2)
        customer_metrics['value_segment_analysis'] = value_segment_analysis
        
        return customer_metrics, customer_summary
    
    def product_analysis(self):
        """Analyze product performance and trends"""
        product_metrics = {}
        
        # Product performance
        product_performance = self.data.groupby(['product_id', 'category']).agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': 'sum',
            'price': 'first'
        }).round(2)
        
        product_performance.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 
                                     'Total_Quantity_Sold', 'Unit_Price']
        
        # Top performing products
        product_metrics['top_products_by_revenue'] = product_performance.nlargest(10, 'Total_Revenue')
        product_metrics['top_products_by_quantity'] = product_performance.nlargest(10, 'Total_Quantity_Sold')
        
        # Category analysis
        category_analysis = self.data.groupby('category').agg({
            'total_amount': ['sum', 'mean'],
            'quantity': 'sum',
            'product_id': 'nunique',
            'customer_id': 'nunique'
        }).round(2)
        category_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Quantity', 
                                   'Unique_Products', 'Unique_Customers']
        product_metrics['category_analysis'] = category_analysis
        
        # Price analysis
        product_metrics['price_analysis'] = {
            'avg_price': self.data['price'].mean(),
            'price_range': self.data['price'].max() - self.data['price'].min(),
            'price_distribution': self.data['price'].describe()
        }
        
        return product_metrics
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns and trends"""
        seasonal_metrics = {}
        
        # Monthly analysis
        monthly_analysis = self.data.groupby('month').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        monthly_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 'Unique_Customers']
        seasonal_metrics['monthly_patterns'] = monthly_analysis
        
        # Quarterly analysis
        quarterly_analysis = self.data.groupby('quarter').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        quarterly_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 'Unique_Customers']
        seasonal_metrics['quarterly_patterns'] = quarterly_analysis
        
        # Day of week analysis
        dow_analysis = self.data.groupby('day_of_week').agg({
            'total_amount': ['sum', 'mean', 'count']
        }).round(2)
        dow_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders']
        dow_analysis.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        seasonal_metrics['day_of_week_patterns'] = dow_analysis
        
        # Weekend vs Weekday
        weekend_analysis = self.data.groupby('is_weekend').agg({
            'total_amount': ['sum', 'mean', 'count']
        }).round(2)
        weekend_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders']
        weekend_analysis.index = ['Weekday', 'Weekend']
        seasonal_metrics['weekend_vs_weekday'] = weekend_analysis
        
        return seasonal_metrics
    
    def statistical_tests(self):
        """Perform statistical tests for business insights"""
        test_results = {}
        
        # Test 1: Gender difference in spending
        male_spending = self.data[self.data['gender'] == 'M']['total_amount']
        female_spending = self.data[self.data['gender'] == 'F']['total_amount']
        
        t_stat, p_value = stats.ttest_ind(male_spending, female_spending)
        test_results['gender_spending_test'] = {
            'test': 'Independent t-test',
            'hypothesis': 'Male and female customers have different average spending',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'male_avg': male_spending.mean(),
            'female_avg': female_spending.mean()
        }
        
        # Test 2: Weekend vs Weekday spending
        weekend_spending = self.data[self.data['is_weekend'] == 1]['total_amount']
        weekday_spending = self.data[self.data['is_weekend'] == 0]['total_amount']
        
        t_stat, p_value = stats.ttest_ind(weekend_spending, weekday_spending)
        test_results['weekend_spending_test'] = {
            'test': 'Independent t-test',
            'hypothesis': 'Weekend and weekday spending patterns are different',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'weekend_avg': weekend_spending.mean(),
            'weekday_avg': weekday_spending.mean()
        }
        
        # Test 3: Category spending ANOVA
        categories = self.data['category'].unique()
        category_groups = [self.data[self.data['category'] == cat]['total_amount'] for cat in categories]
        
        f_stat, p_value = stats.f_oneway(*category_groups)
        test_results['category_spending_anova'] = {
            'test': 'One-way ANOVA',
            'hypothesis': 'Different product categories have different average spending',
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return test_results
    
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        # Select numerical columns
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numerical_data.corr()
        
        # Find strong correlations (>0.5 or <-0.5)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        return correlation_matrix, strong_correlations
    
    def predictive_modeling(self):
        """Build simple predictive models"""
        models_results = {}
        
        # Prepare data for modeling
        # Predict total_amount based on other features
        feature_columns = ['age', 'quantity', 'price', 'discount', 'day_of_week', 'month', 'quarter']
        
        # Handle categorical variables
        model_data = self.data.copy()
        model_data['gender_encoded'] = model_data['gender'].map({'M': 1, 'F': 0})
        model_data = pd.get_dummies(model_data, columns=['category', 'city'], prefix=['cat', 'city'])
        
        # Select features that exist in the data
        available_features = [col for col in feature_columns if col in model_data.columns]
        categorical_features = [col for col in model_data.columns if col.startswith(('cat_', 'city_'))]
        all_features = available_features + categorical_features + ['gender_encoded']
        
        X = model_data[all_features]
        y = model_data['total_amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        models_results['linear_regression'] = {
            'mae': mean_absolute_error(y_test, lr_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'r2_score': lr_model.score(X_test, y_test)
        }
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        models_results['random_forest'] = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2_score': rf_model.score(X_test, y_test),
            'feature_importance': dict(zip(all_features, rf_model.feature_importances_))
        }
        
        return models_results
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        insights = []
        
        # Sales performance insights
        performance = self.sales_performance_analysis()
        insights.append(f"Total revenue: ${performance['total_revenue']:,.2f}")
        insights.append(f"Average order value: ${performance['avg_order_value']:.2f}")
        insights.append(f"Monthly growth rate: {performance['monthly_growth_rate']:.1f}%")
        
        # Customer insights
        customer_metrics, _ = self.customer_analysis()
        insights.append(f"Average customer value: ${customer_metrics['avg_customer_value']:.2f}")
        insights.append(f"Average purchase frequency: {customer_metrics['avg_purchase_frequency']:.1f} orders per customer")
        
        # Product insights
        product_metrics = self.product_analysis()
        top_category = product_metrics['category_analysis']['Total_Revenue'].idxmax()
        insights.append(f"Top performing category: {top_category}")
        
        # Seasonal insights
        seasonal_metrics = self.seasonal_analysis()
        best_quarter = seasonal_metrics['quarterly_patterns']['Total_Revenue'].idxmax()
        insights.append(f"Best performing quarter: Q{best_quarter}")
        
        # Statistical test insights
        test_results = self.statistical_tests()
        if test_results['gender_spending_test']['significant']:
            insights.append("Significant difference in spending between male and female customers")
        
        return insights

def main():
    """Example usage of RetailAnalyzer"""
    print("Retail Analysis Module")
    print("Use this module with your processed retail data")
    print("Example usage:")
    print("analyzer = RetailAnalyzer(data)")
    print("performance = analyzer.sales_performance_analysis()")
    print("insights = analyzer.generate_business_insights()")

if __name__ == "__main__":
    main()