"""
Data Ingestion Script for MLOps Pipeline
Ingests customer data from various sources and stores in S3
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data ingestion from various sources to S3"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataIngestion
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config['aws']['region'])
        self.data_bucket = config['aws']['s3']['data_bucket']
        
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate data schema
        
        Args:
            df: Input dataframe
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = [
            'customer_id',
            'tenure',
            'monthly_charges',
            'total_charges',
            'churn'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for null values in critical columns
        if df[required_columns].isnull().any().any():
            logger.warning("Found null values in required columns")
            
        return True
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'total_records': len(df),
            'null_counts': df.isnull().sum().to_dict(),
            'duplicate_count': df.duplicated().sum(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        quality_metrics['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Categorical column value counts
        categorical_cols = df.select_dtypes(include=['object']).columns
        quality_metrics['categorical_counts'] = {
            col: df[col].value_counts().to_dict()
            for col in categorical_cols
        }
        
        logger.info(f"Data quality metrics: {quality_metrics}")
        return quality_metrics
    
    def ingest_csv(self, file_path: str, dataset_name: str) -> pd.DataFrame:
        """
        Ingest data from CSV file
        
        Args:
            file_path: Path to CSV file
            dataset_name: Name for the dataset
            
        Returns:
            Ingested dataframe
        """
        logger.info(f"Ingesting CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            
            # Validate schema
            if not self.validate_data_schema(df):
                raise ValueError("Data schema validation failed")
            
            # Quality checks
            quality_metrics = self.validate_data_quality(df)
            
            # Upload to S3
            self.upload_to_s3(df, dataset_name, quality_metrics)
            
            return df
            
        except Exception as e:
            logger.error(f"Error ingesting CSV: {str(e)}")
            raise
    
    def upload_to_s3(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        quality_metrics: Dict[str, Any]
    ) -> str:
        """
        Upload dataframe to S3
        
        Args:
            df: Dataframe to upload
            dataset_name: Name for the dataset
            quality_metrics: Data quality metrics
            
        Returns:
            S3 URI of uploaded data
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"raw/{dataset_name}/{timestamp}/data.csv"
        
        try:
            # Upload data
            csv_buffer = df.to_csv(index=False)
            self.s3_client.put_object(
                Bucket=self.data_bucket,
                Key=s3_key,
                Body=csv_buffer
            )
            
            # Upload quality metrics
            metrics_key = f"raw/{dataset_name}/{timestamp}/quality_metrics.json"
            self.s3_client.put_object(
                Bucket=self.data_bucket,
                Key=metrics_key,
                Body=json.dumps(quality_metrics, indent=2)
            )
            
            s3_uri = f"s3://{self.data_bucket}/{s3_key}"
            logger.info(f"Data uploaded to {s3_uri}")
            
            return s3_uri
            
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise
    
    def ingest_from_database(
        self,
        query: str,
        dataset_name: str,
        connection_params: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Ingest data from database
        
        Args:
            query: SQL query
            dataset_name: Name for the dataset
            connection_params: Database connection parameters
            
        Returns:
            Ingested dataframe
        """
        logger.info(f"Ingesting data from database with query: {query[:100]}...")
        
        # This is a placeholder - implement based on your database type
        # Example for PostgreSQL:
        # import psycopg2
        # conn = psycopg2.connect(**connection_params)
        # df = pd.read_sql(query, conn)
        # conn.close()
        
        raise NotImplementedError("Database ingestion not implemented yet")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Ingest customer data')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--dataset', default='customer_churn', help='Dataset name')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ingestion
    ingestion = DataIngestion(config)
    
    # Ingest data
    df = ingestion.ingest_csv(args.input, args.dataset)
    
    logger.info(f"Data ingestion completed successfully. Total records: {len(df)}")


if __name__ == "__main__":
    main()
