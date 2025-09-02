"""
Visualization Module for Retail Sales Analysis
Creates comprehensive charts and plots for data analysis and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RetailVisualizer:
    """
    Comprehensive visualization class for retail sales analysis
    """
    
    def __init__(self, data, figsize=(12, 8)):
        self.data = data
        self.figsize = figsize
        
    def sales_overview_dashboard(self):
        """Create a comprehensive sales overview dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Retail Sales Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Total Sales by Month
        monthly_sales = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['total_amount'].sum()
        axes[0, 0].plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Monthly Sales Trend')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Total Sales ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Sales by Category
        category_sales = self.data.groupby('category')['total_amount'].sum().sort_values(ascending=False)
        axes[0, 1].bar(category_sales.index, category_sales.values)
        axes[0, 1].set_title('Sales by Product Category')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Total Sales ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Customer Age Distribution
        axes[0, 2].hist(self.data['age'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Customer Age Distribution')
        axes[0, 2].set_xlabel('Age')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Sales by Day of Week
        dow_sales = self.data.groupby('day_of_week')['total_amount'].sum()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar(days, dow_sales.values)
        axes[1, 0].set_title('Sales by Day of Week')
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Total Sales ($)')
        
        # 5. Price Distribution
        axes[1, 1].hist(self.data['price'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Product Price Distribution')
        axes[1, 1].set_xlabel('Price ($)')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. Sales by Gender
        gender_sales = self.data.groupby('gender')['total_amount'].sum()
        axes[1, 2].pie(gender_sales.values, labels=gender_sales.index, autopct='%1.1f%%')
        axes[1, 2].set_title('Sales Distribution by Gender')
        
        plt.tight_layout()
        plt.show()
        
    def customer_behavior_analysis(self):
        """Analyze customer behavior patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Behavior Analysis', fontsize=16, fontweight='bold')
        
        # 1. Purchase Frequency Distribution
        purchase_freq = self.data.groupby('customer_id').size()
        axes[0, 0].hist(purchase_freq, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Purchase Frequency Distribution')
        axes[0, 0].set_xlabel('Number of Purchases')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # 2. Customer Lifetime Value Distribution
        clv = self.data.groupby('customer_id')['total_amount'].sum()
        axes[0, 1].hist(clv, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Customer Lifetime Value Distribution')
        axes[0, 1].set_xlabel('Total Spent ($)')
        axes[0, 1].set_ylabel('Number of Customers')
        
        # 3. Average Order Value by Age Group
        if 'age_group' in self.data.columns:
            aov_age = self.data.groupby('age_group')['total_amount'].mean()
            axes[1, 0].bar(aov_age.index, aov_age.values)
            axes[1, 0].set_title('Average Order Value by Age Group')
            axes[1, 0].set_xlabel('Age Group')
            axes[1, 0].set_ylabel('Average Order Value ($)')
        
        # 4. Seasonal Trends
        seasonal_sales = self.data.groupby('quarter')['total_amount'].sum()
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        axes[1, 1].bar(quarters, seasonal_sales.values)
        axes[1, 1].set_title('Seasonal Sales Trends')
        axes[1, 1].set_xlabel('Quarter')
        axes[1, 1].set_ylabel('Total Sales ($)')
        
        plt.tight_layout()
        plt.show()
    
    def rfm_visualization(self, rfm_data):
        """Visualize RFM analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Analysis Visualization', fontsize=16, fontweight='bold')
        
        # 1. RFM Distribution
        axes[0, 0].hist(rfm_data['Recency'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        
        axes[0, 1].hist(rfm_data['Frequency'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        axes[0, 2].hist(rfm_data['Monetary'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total Spent ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        # 2. RFM Scores Distribution
        axes[1, 0].hist(rfm_data['R_Score'], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Recency Score Distribution')
        axes[1, 0].set_xlabel('R Score')
        axes[1, 0].set_ylabel('Number of Customers')
        
        axes[1, 1].hist(rfm_data['F_Score'], bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Frequency Score Distribution')
        axes[1, 1].set_xlabel('F Score')
        axes[1, 1].set_ylabel('Number of Customers')
        
        axes[1, 2].hist(rfm_data['M_Score'], bins=5, alpha=0.7, color='salmon', edgecolor='black')
        axes[1, 2].set_title('Monetary Score Distribution')
        axes[1, 2].set_xlabel('M Score')
        axes[1, 2].set_ylabel('Number of Customers')
        
        plt.tight_layout()
        plt.show()
        
        # RFM Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('RFM Metrics Correlation Heatmap')
        plt.show()
    
    def segment_visualization(self, rfm_data):
        """Visualize customer segments"""
        if 'Segment' not in rfm_data.columns:
            print("No segments found. Please create segments first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Segment Distribution
        segment_counts = rfm_data['Segment'].value_counts()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Customer Segment Distribution')
        
        # 2. Segment Characteristics
        segment_metrics = rfm_data.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
        segment_metrics.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average RFM Metrics by Segment')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Average Value')
        axes[0, 1].legend(title='Metrics')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Segment Revenue Contribution
        segment_revenue = rfm_data.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
        axes[1, 0].bar(segment_revenue.index, segment_revenue.values)
        axes[1, 0].set_title('Total Revenue by Segment')
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Total Revenue ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. RFM Scatter Plot
        scatter = axes[1, 1].scatter(rfm_data['Frequency'], rfm_data['Monetary'], 
                                   c=rfm_data['Recency'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Monetary')
        axes[1, 1].set_title('RFM Scatter Plot (Color = Recency)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Recency')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_sales_dashboard(self):
        """Create interactive dashboard using Plotly"""
        # Monthly sales trend
        monthly_sales = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['total_amount'].sum().reset_index()
        monthly_sales['transaction_date'] = monthly_sales['transaction_date'].astype(str)
        
        fig1 = px.line(monthly_sales, x='transaction_date', y='total_amount',
                      title='Monthly Sales Trend', markers=True)
        fig1.update_layout(xaxis_title='Month', yaxis_title='Total Sales ($)')
        fig1.show()
        
        # Category performance
        category_sales = self.data.groupby('category')['total_amount'].sum().reset_index()
        fig2 = px.bar(category_sales, x='category', y='total_amount',
                     title='Sales by Product Category')
        fig2.update_layout(xaxis_title='Category', yaxis_title='Total Sales ($)')
        fig2.show()
        
        # Customer age vs spending
        customer_summary = self.data.groupby('customer_id').agg({
            'age': 'first',
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        fig3 = px.scatter(customer_summary, x='age', y='total_amount', 
                         size='transaction_id', hover_data=['customer_id'],
                         title='Customer Age vs Total Spending')
        fig3.update_layout(xaxis_title='Age', yaxis_title='Total Spending ($)')
        fig3.show()
    
    def create_executive_summary_chart(self):
        """Create executive summary visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Executive Summary - Key Metrics', fontsize=16, fontweight='bold')
        
        # Key metrics
        total_revenue = self.data['total_amount'].sum()
        total_customers = self.data['customer_id'].nunique()
        total_transactions = len(self.data)
        avg_order_value = self.data['total_amount'].mean()
        
        # 1. Revenue by Quarter
        quarterly_revenue = self.data.groupby('quarter')['total_amount'].sum()
        axes[0, 0].bar(['Q1', 'Q2', 'Q3', 'Q4'], quarterly_revenue.values, color='steelblue')
        axes[0, 0].set_title(f'Quarterly Revenue (Total: ${total_revenue:,.0f})')
        axes[0, 0].set_ylabel('Revenue ($)')
        
        # 2. Top Categories
        top_categories = self.data.groupby('category')['total_amount'].sum().nlargest(5)
        axes[0, 1].barh(top_categories.index, top_categories.values, color='lightcoral')
        axes[0, 1].set_title('Top 5 Product Categories')
        axes[0, 1].set_xlabel('Revenue ($)')
        
        # 3. Customer Metrics
        metrics = ['Total Customers', 'Total Transactions', 'Avg Order Value']
        values = [total_customers, total_transactions, avg_order_value]
        colors = ['gold', 'lightgreen', 'plum']
        
        bars = axes[1, 0].bar(metrics, values, color=colors)
        axes[1, 0].set_title('Key Business Metrics')
        axes[1, 0].set_ylabel('Count / Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:,.0f}', ha='center', va='bottom')
        
        # 4. Monthly Growth Rate
        monthly_sales = self.data.groupby(self.data['transaction_date'].dt.to_period('M'))['total_amount'].sum()
        growth_rate = monthly_sales.pct_change() * 100
        axes[1, 1].plot(range(len(growth_rate)), growth_rate.values, marker='o', color='darkgreen')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Monthly Growth Rate (%)')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Growth Rate (%)')
        
        plt.tight_layout()
        plt.show()

def main():
    """Example usage of RetailVisualizer"""
    print("Retail Visualization Module")
    print("Use this module with your processed retail data")
    print("Example usage:")
    print("visualizer = RetailVisualizer(data)")
    print("visualizer.sales_overview_dashboard()")
    print("visualizer.customer_behavior_analysis()")

if __name__ == "__main__":
    main()