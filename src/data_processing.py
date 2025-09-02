"""
Data Processing Module for Retail Sales Analysis
Handles data cleaning, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    A comprehensive data processing class for retail sales data
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path):
        """Load data from various file formats"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def generate_sample_data(self, n_customers=1000, n_transactions=5000):
        """Generate synthetic retail sales data for demonstration"""
        np.random.seed(42)
        
        # Generate customer data
        customers = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(40, 15, n_customers).astype(int),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
            'registration_date': pd.date_range('2020-01-01', '2023-01-01', periods=n_customers)
        })
        
        # Generate product data
        products = pd.DataFrame({
            'product_id': range(1, 101),
            'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], 100),
            'price': np.random.uniform(10, 500, 100).round(2)
        })
        
        # Generate transaction data
        transactions = pd.DataFrame({
            'transaction_id': range(1, n_transactions + 1),
            'customer_id': np.random.choice(customers['customer_id'], n_transactions),
            'product_id': np.random.choice(products['product_id'], n_transactions),
            'quantity': np.random.choice([1, 2, 3, 4, 5], n_transactions, p=[0.5, 0.25, 0.15, 0.07, 0.03]),
            'transaction_date': pd.date_range('2022-01-01', '2024-01-01', periods=n_transactions),
            'discount': np.random.uniform(0, 0.3, n_transactions).round(2)
        })
        
        # Merge data
        self.data = transactions.merge(customers, on='customer_id').merge(products, on='product_id')
        self.data['total_amount'] = self.data['price'] * self.data['quantity'] * (1 - self.data['discount'])
        
        print(f"Sample data generated. Shape: {self.data.shape}")
        return self.data
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            print("No data to clean. Please load data first.")
            return None
        
        df = self.data.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Convert date columns
        date_columns = ['transaction_date', 'registration_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Remove outliers (using IQR method)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['customer_id', 'product_id', 'transaction_id']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Feature engineering
        df = self.create_features(df)
        
        self.processed_data = df
        print(f"Data cleaned successfully. Shape: {df.shape}")
        return df
    
    def create_features(self, df):
        """Create additional features for analysis"""
        # Time-based features
        df['year'] = df['transaction_date'].dt.year
        df['month'] = df['transaction_date'].dt.month
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['quarter'] = df['transaction_date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Customer age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Price categories
        df['price_category'] = pd.cut(df['price'], bins=[0, 50, 100, 200, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Customer tenure (days since registration)
        df['customer_tenure'] = (df['transaction_date'] - df['registration_date']).dt.days
        
        return df
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        if self.processed_data is None:
            print("No processed data available. Please clean data first.")
            return None
        
        summary = {
            'shape': self.processed_data.shape,
            'columns': list(self.processed_data.columns),
            'data_types': self.processed_data.dtypes.to_dict(),
            'missing_values': self.processed_data.isnull().sum().to_dict(),
            'numeric_summary': self.processed_data.describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Categorical columns summary
        categorical_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = self.processed_data[col].value_counts().to_dict()
        
        return summary
    
    def save_processed_data(self, file_path):
        """Save processed data to file"""
        if self.processed_data is None:
            print("No processed data to save.")
            return False
        
        try:
            if file_path.endswith('.csv'):
                self.processed_data.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                self.processed_data.to_excel(file_path, index=False)
            
            print(f"Processed data saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

def main():
    """Example usage of DataProcessor"""
    processor = DataProcessor()
    
    # Generate sample data
    data = processor.generate_sample_data(n_customers=1000, n_transactions=5000)
    
    # Clean and process data
    processed_data = processor.clean_data()
    
    # Get summary
    summary = processor.get_data_summary()
    print("\nData Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {len(summary['columns'])}")
    
    return processed_data

if __name__ == "__main__":
    main()