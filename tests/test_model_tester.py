import pytest
import pandas as pd
import numpy as np
import os
from app.model_tester import run_basic_model

def test_run_basic_model_with_valid_data(sample_dataset, tmp_path):
    # Save sample dataset to temp file
    dataset_path = tmp_path / "test_data.csv"
    sample_dataset.to_csv(dataset_path, index=False)
    
    results = run_basic_model(str(dataset_path), target_column="growth_rate")
    assert isinstance(results, dict)
    assert "rmse" in results
    assert "r2" in results
    assert "feature_importance" in results
    assert all(0 <= v <= 1 for v in results["feature_importance"].values())
    assert -1 <= results["r2"] <= 1  # R² can be negative for poor fits
    assert results["rmse"] >= 0  # RMSE is always non-negative

def test_run_basic_model_with_binary_target():
    # Create regression dataset
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': np.random.uniform(0, 1, 100)  # Continuous target
    })
    path = "test_binary.csv"
    data.to_csv(path, index=False)
    
    results = run_basic_model(path, target_column="target")
    assert isinstance(results, dict)
    assert "rmse" in results
    assert "r2" in results
    assert results["rmse"] >= 0
    
    # Cleanup
    os.remove(path)

def test_run_basic_model_with_invalid_target():
    with pytest.raises(ValueError):
        data = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        path = "test_invalid.csv"
        data.to_csv(path, index=False)
        run_basic_model(path, target_column="non_existent_column")
        os.remove(path)

def test_run_basic_model_metrics_consistency(sample_dataset, tmp_path):
    dataset_path = tmp_path / "test_metrics.csv"
    sample_dataset.to_csv(dataset_path, index=False)
    
    results = run_basic_model(str(dataset_path), target_column="growth_rate")
    assert results["rmse"] >= 0
    assert -1 <= results["r2"] <= 1
    
    # Check feature importance consistency
    importance_sum = sum(results["feature_importance"].values())
    assert abs(importance_sum - 1.0) < 0.01  # Feature importances should sum to approximately 1 