#!/usr/bin/env python3
"""
Retail Sales Analysis & Customer Segmentation
Main execution script for the complete analysis pipeline

This script runs the entire retail sales analysis workflow:
1. Data generation/loading and preprocessing
2. Exploratory data analysis
3. Customer segmentation (RFM + K-means)
4. Business insights generation
5. Visualization and reporting

Usage:
    python main.py [--data-path PATH] [--output-dir DIR] [--n-customers N] [--n-transactions N]

Author: Data Analytics Team
Date: September 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Import custom modules
from data_processing import DataProcessor
from analysis import RetailAnalyzer
from segmentation import CustomerSegmentation
from visualization import RetailVisualizer

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'reports/figures',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Project directories created successfully!")

def load_or_generate_data(data_path=None, n_customers=1000, n_transactions=5000):
    """Load existing data or generate sample data"""
    processor = DataProcessor()
    
    if data_path and os.path.exists(data_path):
        print(f"📁 Loading data from {data_path}...")
        data = processor.load_data(data_path)
    else:
        print(f"🔄 Generating sample data ({n_customers} customers, {n_transactions} transactions)...")
        data = processor.generate_sample_data(n_customers=n_customers, n_transactions=n_transactions)
    
    return processor, data

def perform_data_analysis(data):
    """Perform comprehensive data analysis"""
    print("\n📊 PERFORMING DATA ANALYSIS")
    print("=" * 40)
    
    analyzer = RetailAnalyzer(data)
    
    # Generate comprehensive analysis
    print("🔍 Generating descriptive statistics...")
    desc_stats = analyzer.descriptive_statistics()
    
    print("💰 Analyzing sales performance...")
    sales_performance = analyzer.sales_performance_analysis()
    
    print("👥 Analyzing customer behavior...")
    customer_metrics, customer_summary = analyzer.customer_analysis()
    
    print("📦 Analyzing product performance...")
    product_metrics = analyzer.product_analysis()
    
    print("📅 Analyzing seasonal patterns...")
    seasonal_metrics = analyzer.seasonal_analysis()
    
    print("🧪 Performing statistical tests...")
    test_results = analyzer.statistical_tests()
    
    print("🔗 Analyzing correlations...")
    correlation_matrix, strong_correlations = analyzer.correlation_analysis()
    
    print("🤖 Building predictive models...")
    model_results = analyzer.predictive_modeling()
    
    print("💡 Generating business insights...")
    insights = analyzer.generate_business_insights()
    
    return {
        'descriptive_stats': desc_stats,
        'sales_performance': sales_performance,
        'customer_metrics': customer_metrics,
        'customer_summary': customer_summary,
        'product_metrics': product_metrics,
        'seasonal_metrics': seasonal_metrics,
        'test_results': test_results,
        'correlation_matrix': correlation_matrix,
        'strong_correlations': strong_correlations,
        'model_results': model_results,
        'insights': insights
    }

def perform_customer_segmentation(data):
    """Perform customer segmentation analysis"""
    print("\n🎯 PERFORMING CUSTOMER SEGMENTATION")
    print("=" * 45)
    
    segmenter = CustomerSegmentation(data)
    
    print("📈 Calculating RFM metrics...")
    rfm_data = segmenter.calculate_rfm()
    
    print("🏷️ Creating RFM segments...")
    rfm_segments = segmenter.create_rfm_segments()
    
    print("🔍 Finding optimal number of clusters...")
    optimal_k, inertias, silhouette_scores = segmenter.find_optimal_clusters(max_clusters=8)
    
    print(f"🎯 Performing K-means clustering with {optimal_k} clusters...")
    rfm_clustered, kmeans_model = segmenter.kmeans_segmentation(n_clusters=optimal_k)
    
    print("📊 Analyzing segments...")
    segment_analysis = segmenter.analyze_segments()
    
    print("💼 Generating business recommendations...")
    recommendations = segmenter.get_segment_recommendations()
    
    return {
        'rfm_data': rfm_data,
        'rfm_segments': rfm_segments,
        'rfm_clustered': rfm_clustered,
        'optimal_k': optimal_k,
        'segment_analysis': segment_analysis,
        'recommendations': recommendations,
        'kmeans_model': kmeans_model
    }

def create_visualizations(data, analysis_results, segmentation_results):
    """Create comprehensive visualizations"""
    print("\n📈 CREATING VISUALIZATIONS")
    print("=" * 35)
    
    visualizer = RetailVisualizer(data)
    
    print("📊 Creating sales overview dashboard...")
    plt.figure()
    visualizer.sales_overview_dashboard()
    plt.savefig('reports/figures/sales_overview_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("👥 Creating customer behavior analysis...")
    plt.figure()
    visualizer.customer_behavior_analysis()
    plt.savefig('reports/figures/customer_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("🎯 Creating RFM visualization...")
    plt.figure()
    visualizer.rfm_visualization(segmentation_results['rfm_clustered'])
    plt.savefig('reports/figures/rfm_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("🏷️ Creating segment visualization...")
    plt.figure()
    visualizer.segment_visualization(segmentation_results['rfm_clustered'])
    plt.savefig('reports/figures/customer_segments.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📋 Creating executive summary chart...")
    plt.figure()
    visualizer.create_executive_summary_chart()
    plt.savefig('reports/figures/executive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ All visualizations saved to reports/figures/")

def save_results(data, analysis_results, segmentation_results, output_dir='data/processed'):
    """Save analysis results to files"""
    print(f"\n💾 SAVING RESULTS TO {output_dir}")
    print("=" * 40)
    
    # Save processed data
    data.to_csv(f'{output_dir}/processed_retail_data.csv', index=False)
    print(f"✅ Processed data saved to {output_dir}/processed_retail_data.csv")
    
    # Save customer summary
    analysis_results['customer_summary'].to_csv(f'{output_dir}/customer_summary.csv')
    print(f"✅ Customer summary saved to {output_dir}/customer_summary.csv")
    
    # Save segmentation results
    segmentation_results['rfm_clustered'].to_csv(f'{output_dir}/customer_segments.csv', index=False)
    print(f"✅ Customer segments saved to {output_dir}/customer_segments.csv")
    
    # Save performance metrics
    performance_metrics = []
    for cluster_id in sorted(segmentation_results['rfm_clustered']['Cluster'].unique()):
        cluster_data = segmentation_results['rfm_clustered'][segmentation_results['rfm_clustered']['Cluster'] == cluster_id]
        metrics = {
            'Cluster': cluster_id,
            'Customer_Count': len(cluster_data),
            'Total_Revenue': cluster_data['Monetary'].sum(),
            'Avg_Customer_Value': cluster_data['Monetary'].mean(),
            'Avg_Recency': cluster_data['Recency'].mean(),
            'Avg_Frequency': cluster_data['Frequency'].mean()
        }
        performance_metrics.append(metrics)
    
    performance_df = pd.DataFrame(performance_metrics)
    performance_df.to_csv(f'{output_dir}/segment_performance.csv', index=False)
    print(f"✅ Segment performance saved to {output_dir}/segment_performance.csv")

def print_executive_summary(analysis_results, segmentation_results):
    """Print executive summary to console"""
    print("\n" + "="*60)
    print("📋 EXECUTIVE SUMMARY")
    print("="*60)
    
    # Key metrics
    total_revenue = analysis_results['sales_performance']['total_revenue']
    total_customers = analysis_results['sales_performance']['unique_customers']
    avg_order_value = analysis_results['sales_performance']['avg_order_value']
    
    print(f"\n💰 FINANCIAL METRICS:")
    print(f"   • Total Revenue: ${total_revenue:,.2f}")
    print(f"   • Average Order Value: ${avg_order_value:.2f}")
    print(f"   • Revenue per Customer: ${total_revenue/total_customers:.2f}")
    
    # Customer insights
    print(f"\n👥 CUSTOMER INSIGHTS:")
    print(f"   • Total Customers: {total_customers:,}")
    print(f"   • Average Customer Value: ${analysis_results['customer_metrics']['avg_customer_value']:.2f}")
    print(f"   • Average Purchase Frequency: {analysis_results['customer_metrics']['avg_purchase_frequency']:.1f}")
    
    # Segmentation results
    segment_counts = segmentation_results['rfm_clustered']['Cluster'].value_counts().sort_index()
    print(f"\n🎯 CUSTOMER SEGMENTS:")
    for cluster_id in segment_counts.index:
        cluster_data = segmentation_results['rfm_clustered'][segmentation_results['rfm_clustered']['Cluster'] == cluster_id]
        count = len(cluster_data)
        percentage = count / len(segmentation_results['rfm_clustered']) * 100
        revenue = cluster_data['Monetary'].sum()
        print(f"   • Cluster {cluster_id}: {count} customers ({percentage:.1f}%) - ${revenue:,.2f} revenue")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    for insight in analysis_results['insights'][:5]:  # Top 5 insights
        print(f"   • {insight}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   • Review detailed analysis in Jupyter notebooks")
    print(f"   • Implement customer segmentation in CRM system")
    print(f"   • Launch targeted marketing campaigns")
    print(f"   • Monitor segment performance and adjust strategies")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Retail Sales Analysis & Customer Segmentation')
    parser.add_argument('--data-path', type=str, help='Path to existing data file')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory for results')
    parser.add_argument('--n-customers', type=int, default=1000, help='Number of customers for sample data')
    parser.add_argument('--n-transactions', type=int, default=5000, help='Number of transactions for sample data')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    print("🚀 RETAIL SALES ANALYSIS & CUSTOMER SEGMENTATION")
    print("=" * 55)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup project structure
    setup_directories()
    
    # Load or generate data
    processor, data = load_or_generate_data(
        data_path=args.data_path,
        n_customers=args.n_customers,
        n_transactions=args.n_transactions
    )
    
    # Clean and preprocess data
    print("\n🧹 CLEANING AND PREPROCESSING DATA")
    print("=" * 40)
    processed_data = processor.clean_data()
    
    # Perform comprehensive analysis
    analysis_results = perform_data_analysis(processed_data)
    
    # Perform customer segmentation
    segmentation_results = perform_customer_segmentation(processed_data)
    
    # Create visualizations
    if not args.skip_viz:
        create_visualizations(processed_data, analysis_results, segmentation_results)
    
    # Save results
    save_results(processed_data, analysis_results, segmentation_results, args.output_dir)
    
    # Print executive summary
    print_executive_summary(analysis_results, segmentation_results)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎉 All analysis completed! Check the following:")
    print("   📁 data/processed/ - for processed data and results")
    print("   📊 reports/figures/ - for visualizations")
    print("   📓 notebooks/ - for detailed Jupyter notebook analysis")
    print("   📋 reports/final_report.md - for executive summary")

if __name__ == "__main__":
    main()