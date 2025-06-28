"""
Advanced Data Preprocessing Pipeline for Production-Level AI Engineering.

This module demonstrates:
- Professional data preprocessing pipelines
- Data validation and quality checks
- Scalable data processing with Dask
- Feature engineering and selection
- Data versioning with DVC
- Production-ready data loaders
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
import dvc.api
import structlog
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure structured logging
logger = structlog.get_logger()

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str
    target_column: str
    text_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    drop_columns: Optional[List[str]] = None
    validation_split: float = 0.2
    test_split: float = 0.2
    random_state: int = 42
    max_features: int = 1000
    n_components: int = 100
    use_dask: bool = False
    dask_npartitions: int = 4


class DataValidator:
    """Comprehensive data validation and quality checks."""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataframe(self, df: pd.DataFrame, config: DataConfig) -> Dict[str, Any]:
        """Perform comprehensive data validation."""
        logger.info("Starting data validation", shape=df.shape)
        
        results = {
            "shape": df.shape,
            "missing_values": {},
            "duplicates": {},
            "data_types": {},
            "outliers": {},
            "cardinality": {},
            "correlations": {},
            "quality_score": 0.0
        }
        
        # Check missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        results["missing_values"] = {
            "counts": missing_counts.to_dict(),
            "percentages": missing_percentages.to_dict()
        }
        
        # Check duplicates
        duplicate_rows = df.duplicated().sum()
        results["duplicates"] = {
            "count": duplicate_rows,
            "percentage": (duplicate_rows / len(df)) * 100
        }
        
        # Check data types
        results["data_types"] = df.dtypes.to_dict()
        
        # Check cardinality for categorical columns
        if config.categorical_columns:
            for col in config.categorical_columns:
                if col in df.columns:
                    unique_count = df[col].nunique()
                    results["cardinality"][col] = {
                        "unique_count": unique_count,
                        "unique_percentage": (unique_count / len(df)) * 100
                    }
        
        # Check for outliers in numerical columns
        if config.numerical_columns:
            for col in config.numerical_columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    results["outliers"][col] = {
                        "count": outliers,
                        "percentage": (outliers / len(df)) * 100
                    }
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(results)
        results["quality_score"] = quality_score
        
        self.validation_results = results
        logger.info("Data validation completed", quality_score=quality_score)
        
        return results
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 100.0
        
        # Penalize for missing values
        missing_penalty = sum(results["missing_values"]["percentages"].values()) * 0.5
        score -= missing_penalty
        
        # Penalize for duplicates
        score -= results["duplicates"]["percentage"] * 2
        
        # Penalize for high cardinality (potential data quality issues)
        for card_info in results["cardinality"].values():
            if card_info["unique_percentage"] > 50:
                score -= 10
        
        return max(0.0, score)


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Advanced text feature extraction with multiple techniques."""
    
    def __init__(self, max_features: int = 1000, use_tfidf: bool = True, use_count: bool = False):
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        self.use_count = use_count
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """Fit the text feature extractors."""
        text_data = X if isinstance(X, pd.Series) else pd.Series(X)
        
        if self.use_tfidf:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            self.tfidf_vectorizer.fit(text_data)
        
        if self.use_count:
            self.count_vectorizer = CountVectorizer(
                max_features=self.max_features // 2,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            self.count_vectorizer.fit(text_data)
        
        return self
    
    def transform(self, X):
        """Transform text data to features."""
        text_data = X if isinstance(X, pd.Series) else pd.Series(X)
        features = []
        
        if self.use_tfidf and self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform(text_data)
            features.append(tfidf_features)
        
        if self.use_count and self.count_vectorizer:
            count_features = self.count_vectorizer.transform(text_data)
            features.append(count_features)
        
        if features:
            from scipy.sparse import hstack
            return hstack(features)
        else:
            return np.zeros((len(text_data), 1))


class AdvancedImputer(BaseEstimator, TransformerMixin):
    """Advanced imputation strategies."""
    
    def __init__(self, strategy: str = 'auto', k_neighbors: int = 5):
        self.strategy = strategy
        self.k_neighbors = k_neighbors
        self.imputers = {}
    
    def fit(self, X, y=None):
        """Fit imputers for different column types."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            if X_df[col].dtype in ['int64', 'float64']:
                if self.strategy == 'knn':
                    self.imputers[col] = KNNImputer(n_neighbors=self.k_neighbors)
                else:
                    self.imputers[col] = SimpleImputer(strategy='mean')
            else:
                self.imputers[col] = SimpleImputer(strategy='most_frequent')
            
            # Fit only if there are missing values
            if bool(X_df[col].isnull().any()):
                self.imputers[col].fit(X_df[[col]])
        
        return self
    
    def transform(self, X):
        """Transform data with imputation."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_imputed = X_df.copy()
        
        for col, imputer in self.imputers.items():
            if col in X_df.columns:
                X_imputed[col] = imputer.transform(X_df[[col]]).flatten()
        
        return X_imputed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Advanced feature selection with multiple strategies."""
    
    def __init__(self, method: str = 'mutual_info', k: int = 100, threshold: float = 0.01):
        self.method = method
        self.k = k
        self.threshold = threshold
        self.selected_features_ = []
        self.feature_scores_ = {}
    
    def fit(self, X, y):
        """Fit feature selector."""
        if self.method == 'mutual_info':
            self._mutual_info_selection(X, y)
        elif self.method == 'variance':
            self._variance_selection(X)
        elif self.method == 'correlation':
            self._correlation_selection(X, y)
        
        return self
    
    def _mutual_info_selection(self, X, y):
        """Select features based on mutual information."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            if X_df[col].dtype in ['int64', 'float64']:
                try:
                    score = mutual_info_score(y, X_df[col])
                    self.feature_scores_[col] = score
                except:
                    self.feature_scores_[col] = 0
        
        # Select top k features
        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        self.selected_features_ = [f[0] for f in sorted_features[:self.k]]
    
    def _variance_selection(self, X):
        """Select features based on variance."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        variances = X_df.var()
        if isinstance(variances, pd.Series):
            self.selected_features_ = variances.nlargest(self.k).index.tolist()
        else:
            self.selected_features_ = [X_df.columns[0]]
    
    def _correlation_selection(self, X, y):
        """Select features based on correlation with target."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        correlations = X_df.corrwith(pd.Series(y)).abs()
        self.selected_features_ = correlations.nlargest(self.k).index.tolist()
    
    def transform(self, X):
        """Transform data to selected features."""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return X_df[self.selected_features_]


class AdvancedDataPipeline:
    """Production-level data preprocessing pipeline."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator()
        self.pipeline = None
        self.feature_names_ = []
        self.validation_results = {}
        
        # Setup Dask client if needed
        if config.use_dask:
            self.client = Client(n_workers=config.dask_npartitions)
        else:
            self.client = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data with support for multiple formats."""
        logger.info("Loading data", path=self.config.data_path)
        
        file_path = Path(self.config.data_path)
        
        if file_path.suffix.lower() == '.csv':
            if self.config.use_dask:
                df = dd.read_csv(self.config.data_path).compute()
            else:
                df = pd.read_csv(self.config.data_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(self.config.data_path)
        elif file_path.suffix.lower() == '.json':
            df = pd.read_json(self.config.data_path)
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.config.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info("Data loaded successfully", shape=df.shape)
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        return self.validator.validate_dataframe(df, self.config)
    
    def create_pipeline(self) -> Pipeline:
        """Create comprehensive data preprocessing pipeline."""
        logger.info("Creating data preprocessing pipeline")
        
        # Define preprocessing steps
        preprocessing_steps = []
        
        # Text preprocessing
        if self.config.text_columns:
            text_transformer = Pipeline([
                ('text_features', TextFeatureExtractor(
                    max_features=self.config.max_features,
                    use_tfidf=True,
                    use_count=True
                )),
                ('dimension_reduction', TruncatedSVD(n_components=self.config.n_components))
            ])
            
            preprocessing_steps.append(('text', text_transformer, self.config.text_columns))
        
        # Numerical preprocessing
        if self.config.numerical_columns:
            numerical_transformer = Pipeline([
                ('imputer', AdvancedImputer(strategy='knn')),
                ('scaler', StandardScaler())
            ])
            
            preprocessing_steps.append(('numerical', numerical_transformer, self.config.numerical_columns))
        
        # Categorical preprocessing
        if self.config.categorical_columns:
            categorical_transformer = Pipeline([
                ('imputer', AdvancedImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False))
            ])
            
            preprocessing_steps.append(('categorical', categorical_transformer, self.config.categorical_columns))
        
        # Create column transformer
        if preprocessing_steps:
            preprocessor = ColumnTransformer(
                transformers=preprocessing_steps,
                remainder='drop'
            )
        else:
            preprocessor = Pipeline([
                ('imputer', AdvancedImputer()),
                ('scaler', StandardScaler())
            ])
        
        # Create full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', FeatureSelector(method='mutual_info', k=self.config.max_features))
        ])
        
        logger.info("Pipeline created successfully")
        return self.pipeline
    
    def process_data(self, df: pd.DataFrame) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
        """Process data through the pipeline."""
        logger.info("Processing data through pipeline")
        
        # Validate data first
        self.validation_results = self.validate_data(df)
        
        # Drop specified columns
        if self.config.drop_columns:
            df = df.drop(columns=self.config.drop_columns)
        
        # Separate features and target
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        # Create pipeline if not exists
        if self.pipeline is None:
            self.create_pipeline()
        
        # Ensure pipeline exists
        assert self.pipeline is not None, "Pipeline was not created"
        
        # Fit and transform
        X_processed = self.pipeline.fit_transform(X, y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=self.config.validation_split + self.config.test_split,
            random_state=self.config.random_state, stratify=y if len(y.unique()) < 10 else None
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.config.test_split / (self.config.validation_split + self.config.test_split),
            random_state=self.config.random_state, stratify=y_temp if len(pd.Series(y_temp).unique()) < 10 else None
        )
        
        logger.info(
            "Data processing completed",
            train_shape=np.array(X_train).shape,
            val_shape=np.array(X_val).shape,
            test_shape=np.array(X_test).shape
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_pipeline(self, path: str):
        """Save the fitted pipeline."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info("Pipeline saved", path=path)
    
    def load_pipeline(self, path: str):
        """Load a fitted pipeline."""
        import joblib
        self.pipeline = joblib.load(path)
        logger.info("Pipeline loaded", path=path)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call process_data() first.")
        
        # Get feature names
        feature_names = []
        if hasattr(self.pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate importance scores
        importance_scores = {}
        for i, name in enumerate(feature_names):
            try:
                score = mutual_info_score(y, X.iloc[:, i])
                importance_scores[name] = score
            except:
                importance_scores[name] = 0
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive data processing report."""
        report = {
            "data_validation": self.validation_results,
            "pipeline_steps": [step[0] for step in self.pipeline.steps] if self.pipeline else [],
            "feature_count": len(self.feature_names_) if self.feature_names_ else 0,
            "processing_summary": {
                "text_columns_processed": len(self.config.text_columns) if self.config.text_columns else 0,
                "numerical_columns_processed": len(self.config.numerical_columns) if self.config.numerical_columns else 0,
                "categorical_columns_processed": len(self.config.categorical_columns) if self.config.categorical_columns else 0
            }
        }
        
        return report


def create_dvc_pipeline(config: DataConfig, output_path: str = "data/processed"):
    """Create DVC pipeline for data versioning."""
    dvc_stages = {
        "load_data": {
            "cmd": f"python -c \"from app.data_pipeline import AdvancedDataPipeline; import json; config = DataConfig(**{config.__dict__}); pipeline = AdvancedDataPipeline(config); df = pipeline.load_data(); df.to_parquet('{output_path}/raw_data.parquet')\"",
            "deps": [config.data_path],
            "outs": [f"{output_path}/raw_data.parquet"]
        },
        "process_data": {
            "cmd": f"python -c \"from app.data_pipeline import AdvancedDataPipeline; import json; config = DataConfig(**{config.__dict__}); pipeline = AdvancedDataPipeline(config); df = pipeline.load_data(); (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.process_data(df); pipeline.save_pipeline('{output_path}/pipeline.pkl')\"",
            "deps": [f"{output_path}/raw_data.parquet"],
            "outs": [f"{output_path}/pipeline.pkl"]
        }
    }
    
    return dvc_stages


if __name__ == "__main__":
    # Example usage
    config = DataConfig(
        data_path="data/sample_dataset.csv",
        target_column="target",
        text_columns=["text"],
        categorical_columns=["category"],
        numerical_columns=["feature1", "feature2"],
        max_features=500,
        n_components=50
    )
    
    # Create and run pipeline
    pipeline = AdvancedDataPipeline(config)
    df = pipeline.load_data()
    
    # Validate data
    validation_results = pipeline.validate_data(df)
    print("Data validation results:", validation_results)
    
    # Process data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.process_data(df)
    
    # Generate report
    report = pipeline.generate_report()
    print("Processing report:", report)
    
    # Save pipeline
    pipeline.save_pipeline("models/data_pipeline.pkl") 