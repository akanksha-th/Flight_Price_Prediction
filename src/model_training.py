from dataclasses import dataclass
from typing import Dict, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------- Config ----------------------
@dataclass
class TrainingConfig:
    param_grid: Optional[Dict] = None         # Hyperparameter grid
    scoring: str = "neg_root_mean_squared_error"
    cv: int = 5
    test_size: float = 0.2
    random_state: int = 42
    save_path: str = "models"
    model_name: str = "trained_model.pkl"
    maximize_metric: bool = False             # True if metric needs to be maximized

# ---------------------- Trainer Class ----------------------
class ModelTrainer:
    def __init__(self, model, X: pd.DataFrame, y: pd.Series, config: TrainingConfig):
        self.model = model
        self.X = X
        self.y = y
        self.config = config
        os.makedirs(self.config.save_path, exist_ok=True)
        self.best_model = None
        self.best_score = None
        self.model_path = os.path.join(self.config.save_path, self.config.model_name)

    def train(self):
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=self.config.test_size, random_state=self.config.random_state
        )

        if self.config.param_grid:
            # GridSearchCV hyperparameter tuning
            grid = GridSearchCV(
                estimator=self.model,
                param_grid=self.config.param_grid,
                scoring=self.config.scoring,
                cv=self.config.cv,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            # No tuning, fit original model
            best_model = self.model
            best_model.fit(X_train, y_train)

        # Evaluate
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        metric = -rmse if self.config.scoring.startswith("neg_") else rmse
        print(f"Validation RMSE: {rmse:.4f}, R2: {r2:.4f}")

        # Load existing model for comparison
        if os.path.exists(self.model_path):
            existing_model = joblib.load(self.model_path)
            y_existing_pred = existing_model.predict(X_val)
            existing_rmse = np.sqrt(mean_squared_error(y_val, y_existing_pred))
            print(f"Existing model RMSE: {existing_rmse:.4f}")

            # Update model only if improved
            improved = (metric < existing_rmse) if not self.config.maximize_metric else (metric > existing_rmse)
            if improved:
                joblib.dump(best_model, self.model_path)
                print(f"Model improved. Saved new model to {self.model_path}")
            else:
                best_model = existing_model
                print("No improvement. Existing model retained.")
        else:
            # No existing model, save first time
            joblib.dump(best_model, self.model_path)
            print(f"Saved first trained model to {self.model_path}")

        self.best_model = best_model
        self.best_score = metric
        return best_model

# ---------------------- Example Usage ----------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    # Sample dataset
    df = pd.DataFrame({
        'distance': [100, 200, 300, 400, 500, 600],
        'price': [50, 80, 60, 90, 100, 110],
        'delay': [5, 15, 10, 20, 25, 30]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0])

    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, None]
    }

    config = TrainingConfig(param_grid=param_grid, cv=2)
    trainer = ModelTrainer(model, df, y, config)
    best_model = trainer.train()
