"""Basic model testing module."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Union
import os

def run_basic_model(data: Union[str, pd.DataFrame], target_column: str) -> Dict[str, Any]:
    """Run a basic model on the data and return metrics."""
    # Load data if it's a file path
    if isinstance(data, str):
        data = pd.read_csv(data)
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "rmse": rmse,
        "r2": r2,
        "feature_importance": dict(zip(X.columns, model.feature_importances_))
    }

if __name__ == "__main__":
    data_dir = "data/example_datasets"
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            print(run_basic_model(os.path.join(data_dir, file), target_column="label")) 