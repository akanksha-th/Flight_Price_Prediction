from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod
import joblib
import os

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, BaggingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# ---------------------- Config Dataclass ----------------------
@dataclass
class ModelConfig:
    baseline_models: Optional[List[str]] = None  # List of baseline model names
    ensemble_method: Optional[str] = None        # 'voting', 'bagging', 'boosting', 'stacking'
    random_state: int = 42
    save_path: str = "models"                    # directory to save models
    model_name: str = "trained_model.pkl"       # filename

# ---------------------- Base Builder ----------------------
class BaseModelBuilder(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = self._init_baselines()
        self.ensemble_model = None
        os.makedirs(self.config.save_path, exist_ok=True)

    @abstractmethod
    def build_ensemble(self):
        pass

    def _init_baselines(self):
        """Initialize baseline models based on config"""
        baseline_map = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=self.config.random_state),
            'lasso': Lasso(random_state=self.config.random_state),
            'rf': RandomForestRegressor(random_state=self.config.random_state),
            'gbr': GradientBoostingRegressor(random_state=self.config.random_state),
            'dt': DecisionTreeRegressor(random_state=self.config.random_state),
            'svr': SVR()
        }
        models = {}
        if not self.config.baseline_models:
            self.config.baseline_models = ['linear', 'ridge', 'rf', 'gbr']
        for name in self.config.baseline_models:
            if name not in baseline_map:
                raise ValueError(f"Unknown baseline model: {name}")
            models[name] = baseline_map[name]
        return models

    # ---------------------- Save / Load ----------------------
    def save_model(self, model=None, filename=None):
        model_to_save = model or self.ensemble_model or list(self.models.values())[0]
        file_path = os.path.join(self.config.save_path, filename or self.config.model_name)
        joblib.dump(model_to_save, file_path)
        print(f"Model saved at: {file_path}")

    def load_model(self, filename=None):
        file_path = os.path.join(self.config.save_path, filename or self.config.model_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved model found at: {file_path}")
        loaded_model = joblib.load(file_path)
        print(f"Model loaded from: {file_path}")
        return loaded_model

# ---------------------- Regressor Builder ----------------------
class RegressorBuilder(BaseModelBuilder):
    def build_ensemble(self):
        if not self.config.ensemble_method:
            return None

        ensemble_map = {
            'voting': VotingRegressor,
            'bagging': BaggingRegressor,
            'boosting': AdaBoostRegressor,
            'stacking': StackingRegressor
        }

        if self.config.ensemble_method not in ensemble_map:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")

        if self.config.ensemble_method in ['voting', 'stacking']:
            estimator_list = [(name, model) for name, model in self.models.items()]
            if self.config.ensemble_method == 'voting':
                self.ensemble_model = VotingRegressor(estimators=estimator_list)
            else:
                # Stacking with Ridge as final estimator
                self.ensemble_model = StackingRegressor(
                    estimators=estimator_list,
                    final_estimator=Ridge()
                )
        else:
            # Bagging / Boosting wrap a single baseline
            base_model = list(self.models.values())[0]
            if self.config.ensemble_method == 'bagging':
                self.ensemble_model = BaggingRegressor(base_model, n_estimators=10, random_state=self.config.random_state)
            else:
                self.ensemble_model = AdaBoostRegressor(base_model, n_estimators=50, random_state=self.config.random_state)

        return self.ensemble_model

# ---------------------- Factory ----------------------
class ModelBuilderFactory:
    @staticmethod
    def get_builder(config: ModelConfig) -> BaseModelBuilder:
        return RegressorBuilder(config)

# ---------------------- Example Usage ----------------------
if __name__ == "__main__":
    config = ModelConfig(
        baseline_models=['linear', 'rf', 'gbr'],
        ensemble_method='stacking',
        save_path='models',
        model_name='flight_price_model.pkl'
    )
    builder = ModelBuilderFactory.get_builder(config)

    print("Baseline Models:")
    for name, model in builder.models.items():
        print(f"{name}: {model}")

    ensemble_model = builder.build_ensemble()
    print("\nEnsemble Model:")
    print(ensemble_model)

    # Save model
    builder.save_model()

    # Load model
    loaded_model = builder.load_model()
