"""
Feature Engineering Script for Customer Churn Prediction
Transforms raw data into ML-ready features
"""

import argparse
import logging
from typing import Dict, List, Tuple

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering for customer churn prediction"""
    
    def __init__(self, config: Dict):
        """Initialize feature engineering"""
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config['aws']['region'])
        self.scalers = {}
        self.encoders = {}
        
    def load_data_from_s3(self, s3_uri: str) -> pd.DataFrame:
        """Load data from S3"""
        bucket, key = s3_uri.replace('s3://', '').split('/', 1)
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        logger.info(f"Loaded {len(df)} records from {s3_uri}")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features"""
        logger.info("Creating demographic features...")
        
        # Age groups (if age is available)
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '55+']
            )
        
        # Tenure groups
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72, 1000],
                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr']
            )
            
            # Tenure features
            df['tenure_squared'] = df['tenure'] ** 2
            df['tenure_log'] = np.log1p(df['tenure'])
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral and usage pattern features"""
        logger.info("Creating behavioral features...")
        
        # Monthly charges patterns
        if 'monthly_charges' in df.columns:
            df['monthly_charges_log'] = np.log1p(df['monthly_charges'])
            
        # Total charges patterns
        if 'total_charges' in df.columns and 'tenure' in df.columns:
            # Average monthly charge over tenure
            df['avg_monthly_charge'] = df['total_charges'] / (df['tenure'] + 1)
            
            # Charge change rate
            df['charge_change_rate'] = (
                (df['monthly_charges'] - df['avg_monthly_charge']) / 
                (df['avg_monthly_charge'] + 1)
            )
        
        # Service-related features
        service_cols = [col for col in df.columns if 'service' in col.lower()]
        if service_cols:
            df['total_services'] = df[service_cols].sum(axis=1)
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features"""
        logger.info("Creating aggregated features...")
        
        # Payment and contract features
        if 'payment_method' in df.columns and 'contract' in df.columns:
            # Interaction features
            df['payment_contract_interaction'] = (
                df['payment_method'].astype(str) + '_' + 
                df['contract'].astype(str)
            )
        
        # Calculate customer lifetime value (CLV) proxy
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            df['customer_lifetime_value'] = df['monthly_charges'] * df['tenure']
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features if timestamp exists"""
        logger.info("Creating temporal features...")
        
        # If we have timestamps, extract temporal patterns
        if 'signup_date' in df.columns:
            df['signup_date'] = pd.to_datetime(df['signup_date'])
            df['signup_month'] = df['signup_date'].dt.month
            df['signup_year'] = df['signup_date'].dt.year
            df['signup_quarter'] = df['signup_date'].dt.quarter
            df['signup_day_of_week'] = df['signup_date'].dt.dayofweek
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        if 'churn' in categorical_cols:
            categorical_cols.remove('churn')
        
        # Remove ID columns
        categorical_cols = [col for col in categorical_cols if 'id' not in col.lower()]
        
        for col in categorical_cols:
            # Binary categorical
            if df[col].nunique() == 2:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                self.encoders[col] = le
            # Multi-class categorical - one-hot encoding
            elif df[col].nunique() <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def scale_numerical_features(
        self,
        df: pd.DataFrame,
        exclude_cols: List[str] = None
    ) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("Scaling numerical features...")
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Exclude certain columns
        if exclude_cols:
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Remove target and ID columns
        numerical_cols = [
            col for col in numerical_cols 
            if 'id' not in col.lower() and col != 'churn'
        ]
        
        # Scale features
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers['standard'] = scaler
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("Handling missing values...")
        
        # Numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        return df
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature processing pipeline"""
        logger.info("Starting feature processing pipeline...")
        
        # Handle missing values first
        df = self.handle_missing_values(df)
        
        # Create features
        df = self.create_demographic_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_aggregated_features(df)
        df = self.create_temporal_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale numerical features (after encoding)
        df = self.scale_numerical_features(df)
        
        logger.info(f"Feature processing completed. Final shape: {df.shape}")
        return df
    
    def upload_processed_data(
        self,
        df: pd.DataFrame,
        output_prefix: str
    ) -> str:
        """Upload processed data to S3"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bucket = self.config['aws']['s3']['data_bucket']
        key = f"processed/{output_prefix}/{timestamp}/features.csv"
        
        # Upload processed data
        csv_buffer = df.to_csv(index=False)
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer
        )
        
        s3_uri = f"s3://{bucket}/{key}"
        logger.info(f"Processed data uploaded to {s3_uri}")
        
        return s3_uri


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Feature engineering for churn prediction')
    parser.add_argument('--input', required=True, help='Input S3 URI')
    parser.add_argument('--output-prefix', default='customer_churn', help='Output prefix')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize feature engineering
    fe = FeatureEngineering(config)
    
    # Load data
    df = fe.load_data_from_s3(args.input)
    
    # Process features
    df_processed = fe.process_features(df)
    
    # Upload processed data
    output_uri = fe.upload_processed_data(df_processed, args.output_prefix)
    
    logger.info(f"Feature engineering completed. Output: {output_uri}")


if __name__ == "__main__":
    main()
