"""
Data Validation Script
Validates data quality and schema compliance
"""

import argparse
import json
import logging
from typing import Dict, Any, List

import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and schema"""
    
    def __init__(self, config: Dict):
        """Initialize validator"""
        self.config = config
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'failed': []
        }
    
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate data schema"""
        logger.info("Validating schema...")
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.validation_results['failed'].append(
                f"Missing required columns: {missing_cols}"
            )
            return False
        
        self.validation_results['passed'].append("Schema validation passed")
        return True
    
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """Validate data types"""
        logger.info("Validating data types...")
        
        type_issues = []
        
        # Check for numeric columns that should be numeric
        numeric_cols = ['tenure', 'monthly_charges', 'total_charges']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues.append(f"{col} should be numeric")
        
        if type_issues:
            self.validation_results['failed'].extend(type_issues)
            return False
        
        self.validation_results['passed'].append("Data type validation passed")
        return True
    
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> bool:
        """Check for excessive missing values"""
        logger.info("Checking missing values...")
        
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > threshold]
        
        if not high_missing.empty:
            self.validation_results['warnings'].append(
                f"Columns with >{threshold*100}% missing: {high_missing.to_dict()}"
            )
        else:
            self.validation_results['passed'].append("Missing value check passed")
        
        return True
    
    def check_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> bool:
        """Check for outliers in numerical columns"""
        logger.info("Checking for outliers...")
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        outlier_info = {}
        
        for col in numerical_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_count = (z_scores > z_threshold).sum()
            outlier_pct = outlier_count / len(df) * 100
            
            if outlier_pct > 5:  # More than 5% outliers
                outlier_info[col] = f"{outlier_pct:.2f}%"
        
        if outlier_info:
            self.validation_results['warnings'].append(
                f"Columns with >5% outliers: {outlier_info}"
            )
        else:
            self.validation_results['passed'].append("Outlier check passed")
        
        return True
    
    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """Check for duplicate records"""
        logger.info("Checking for duplicates...")
        
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df) * 100
        
        if dup_count > 0:
            self.validation_results['warnings'].append(
                f"Found {dup_count} duplicates ({dup_pct:.2f}%)"
            )
        else:
            self.validation_results['passed'].append("No duplicates found")
        
        return True
    
    def validate_ranges(self, df: pd.DataFrame) -> bool:
        """Validate value ranges"""
        logger.info("Validating value ranges...")
        
        range_issues = []
        
        # Tenure should be non-negative
        if 'tenure' in df.columns:
            if (df['tenure'] < 0).any():
                range_issues.append("Negative tenure values found")
        
        # Charges should be positive
        if 'monthly_charges' in df.columns:
            if (df['monthly_charges'] <= 0).any():
                range_issues.append("Non-positive monthly_charges found")
        
        if range_issues:
            self.validation_results['failed'].extend(range_issues)
            return False
        
        self.validation_results['passed'].append("Range validation passed")
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        total_checks = (
            len(self.validation_results['passed']) +
            len(self.validation_results['warnings']) +
            len(self.validation_results['failed'])
        )
        
        report = {
            'total_checks': total_checks,
            'passed': len(self.validation_results['passed']),
            'warnings': len(self.validation_results['warnings']),
            'failed': len(self.validation_results['failed']),
            'details': self.validation_results,
            'status': 'PASSED' if len(self.validation_results['failed']) == 0 else 'FAILED'
        }
        
        return report


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Validate data')
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--config', default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} records for validation")
    
    # Initialize validator
    validator = DataValidator(config)
    
    # Run validations
    required_cols = ['customer_id', 'tenure', 'monthly_charges', 'churn']
    validator.validate_schema(df, required_cols)
    validator.validate_data_types(df)
    validator.check_missing_values(df)
    validator.check_outliers(df)
    validator.check_duplicates(df)
    validator.validate_ranges(df)
    
    # Generate report
    report = validator.generate_report()
    
    # Print report
    print(json.dumps(report, indent=2))
    
    # Save report
    with open('validation_report.json', 'w') as f:
        json.dumps(report, f, indent=2)
    
    logger.info(f"Validation {report['status']}")
    
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == "__main__":
    exit(main())
