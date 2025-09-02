"""
Customer Segmentation Module for Retail Sales Analysis
Implements RFM analysis and K-means clustering for customer segmentation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Customer segmentation using RFM analysis and K-means clustering
    """
    
    def __init__(self, data):
        self.data = data
        self.rfm_data = None
        self.segmented_data = None
        self.scaler = StandardScaler()
        
    def calculate_rfm(self, customer_col='customer_id', date_col='transaction_date', 
                     amount_col='total_amount', reference_date=None):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
        """
        if reference_date is None:
            reference_date = self.data[date_col].max() + timedelta(days=1)
        
        # Calculate RFM metrics
        rfm = self.data.groupby(customer_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            amount_col: ['count', 'sum']  # Frequency and Monetary
        }).round(2)
        
        # Flatten column names
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm['R_Score'] = rfm['R_Score'].astype(int)
        rfm['F_Score'] = rfm['F_Score'].astype(int)
        rfm['M_Score'] = rfm['M_Score'].astype(int)
        
        # Calculate RFM combined score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        self.rfm_data = rfm
        return rfm
    
    def create_rfm_segments(self):
        """
        Create customer segments based on RFM scores
        """
        if self.rfm_data is None:
            print("Please calculate RFM first using calculate_rfm()")
            return None
        
        def segment_customers(row):
            if row['RFM_Score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif row['RFM_Score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif row['RFM_Score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return 'Potential Loyalists'
            elif row['RFM_Score'] in ['533', '532', '531', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
                return 'New Customers'
            elif row['RFM_Score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return 'Promising'
            elif row['RFM_Score'] in ['155', '254', '245', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '125', '124']:
                return 'Need Attention'
            elif row['RFM_Score'] in ['155', '144', '214', '215', '115', '114']:
                return 'About to Sleep'
            elif row['RFM_Score'] in ['155', '144', '214', '215', '115', '114']:
                return 'At Risk'
            elif row['RFM_Score'] in ['155', '144', '214', '215', '115', '114']:
                return 'Cannot Lose Them'
            elif row['RFM_Score'] in ['144', '214', '215', '115', '114']:
                return 'Hibernating'
            else:
                return 'Lost'
        
        # Simplified segmentation logic
        def simplified_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 3 and f <= 2:
                return 'Potential Loyalists'
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m >= 3:
                return 'Cannot Lose Them'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'
            else:
                return 'Need Attention'
        
        self.rfm_data['Segment'] = self.rfm_data.apply(simplified_segment, axis=1)
        return self.rfm_data
    
    def kmeans_segmentation(self, n_clusters=4, features=['Recency', 'Frequency', 'Monetary']):
        """
        Perform K-means clustering on RFM data
        """
        if self.rfm_data is None:
            print("Please calculate RFM first using calculate_rfm()")
            return None
        
        # Prepare data for clustering
        X = self.rfm_data[features].copy()
        
        # Handle any infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to RFM data
        self.rfm_data['Cluster'] = clusters
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return self.rfm_data, kmeans
    
    def find_optimal_clusters(self, max_clusters=10, features=['Recency', 'Frequency', 'Monetary']):
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        """
        if self.rfm_data is None:
            print("Please calculate RFM first using calculate_rfm()")
            return None
        
        X = self.rfm_data[features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, clusters))
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        
        # Silhouette scores
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, inertias, silhouette_scores
    
    def analyze_segments(self):
        """
        Analyze and profile customer segments
        """
        if self.rfm_data is None:
            print("Please calculate RFM and create segments first")
            return None
        
        # Segment analysis
        segment_analysis = {}
        
        if 'Segment' in self.rfm_data.columns:
            segment_summary = self.rfm_data.groupby('Segment').agg({
                'Recency': ['mean', 'median'],
                'Frequency': ['mean', 'median'],
                'Monetary': ['mean', 'median'],
                'customer_id': 'count'
            }).round(2)
            segment_summary.columns = ['Recency_Mean', 'Recency_Median', 'Frequency_Mean', 
                                     'Frequency_Median', 'Monetary_Mean', 'Monetary_Median', 'Count']
            segment_analysis['RFM_Segments'] = segment_summary
        
        if 'Cluster' in self.rfm_data.columns:
            cluster_summary = self.rfm_data.groupby('Cluster').agg({
                'Recency': ['mean', 'median'],
                'Frequency': ['mean', 'median'],
                'Monetary': ['mean', 'median'],
                'customer_id': 'count'
            }).round(2)
            cluster_summary.columns = ['Recency_Mean', 'Recency_Median', 'Frequency_Mean', 
                                     'Frequency_Median', 'Monetary_Mean', 'Monetary_Median', 'Count']
            segment_analysis['K_Means_Clusters'] = cluster_summary
        
        return segment_analysis
    
    def get_segment_recommendations(self):
        """
        Generate business recommendations for each segment
        """
        recommendations = {
            'Champions': {
                'description': 'Best customers who buy frequently and recently',
                'strategy': 'Reward them, ask for reviews, upsell premium products',
                'actions': ['VIP treatment', 'Early access to new products', 'Referral programs']
            },
            'Loyal Customers': {
                'description': 'Regular customers with good purchase history',
                'strategy': 'Keep them engaged, offer loyalty programs',
                'actions': ['Loyalty rewards', 'Personalized offers', 'Birthday discounts']
            },
            'Potential Loyalists': {
                'description': 'Recent customers with potential for growth',
                'strategy': 'Nurture relationship, increase purchase frequency',
                'actions': ['Onboarding campaigns', 'Product recommendations', 'Free shipping']
            },
            'At Risk': {
                'description': 'Good customers who haven\'t purchased recently',
                'strategy': 'Win them back with targeted campaigns',
                'actions': ['Reactivation emails', 'Special discounts', 'Survey feedback']
            },
            'Cannot Lose Them': {
                'description': 'High-value customers who are becoming inactive',
                'strategy': 'Immediate intervention required',
                'actions': ['Personal outreach', 'Exclusive offers', 'Account manager contact']
            },
            'Lost': {
                'description': 'Customers who haven\'t purchased in a long time',
                'strategy': 'Aggressive win-back campaigns or let go',
                'actions': ['Win-back campaigns', 'Deep discounts', 'Product updates']
            },
            'Need Attention': {
                'description': 'Customers with mixed signals',
                'strategy': 'Understand their needs better',
                'actions': ['Customer surveys', 'A/B testing', 'Personalized communication']
            }
        }
        
        return recommendations

def main():
    """Example usage of CustomerSegmentation"""
    # This would typically use real data
    print("Customer Segmentation Module")
    print("Use this module with your processed retail data")
    print("Example usage:")
    print("segmenter = CustomerSegmentation(data)")
    print("rfm = segmenter.calculate_rfm()")
    print("segments = segmenter.create_rfm_segments()")
    print("analysis = segmenter.analyze_segments()")

if __name__ == "__main__":
    main()