import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
from statsmodels.stats.outliers_influence import OLSInfluence

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

class ModelDiagnostics(ABC):
    def __init__(self, model, X: pd.DataFrame, y: pd.Series):
        """
        model: fitted regression model (statsmodels or sklearn)
        X: features DataFrame
        y: target Series
        """
        self.model = model
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.save_path = None
        
        # Statsmodels support
        if hasattr(model, "fittedvalues"):
            self.fitted = model.fittedvalues
            self.residuals = model.resid
            infl = model.get_influence()
            self.leverage = infl.hat_matrix_diag
            self.cooks = infl.cooks_distance[0]
        else:
            # Sklearn support
            self.fitted = model.predict(X)
            self.residuals = y - self.fitted
            self.leverage = np.zeros(len(y))
            self.cooks = np.zeros(len(y))

    @abstractmethod
    def plot(self, df:pd.DataFrame):
        pass

    @staticmethod
    def run_model_diagnostics(model, X, y, save_path=None):
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        EDA6Plot(model, X, y).plot(save_path)
        InfluenceAndLeverageDiagnostics(model, X, y).plot(save_path)
        ModelFitEvaluation(model, X, y).plot(save_path)
        NonlinearityAndPartialEffects(model, X, y).plot(save_path)


class EDA6Plot(ModelDiagnostics):
    def res_vs_fitted(self, ax):
        """check linearity & equal variance"""
        sns.scatterplot(x=self.fitted, y=self.residuals, ax=ax)
        ax.axhline(0, color='r', linestyle='-')
        ax.set_title("Residuals vs Fitted")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residual values")

    def normal_qq(self, ax):
        """check normality of residuals"""
        sm.qqplot(self.residuals, line='45', ax=ax)
        ax.set_title("Normal QQ Plot")

    def scale_loc_plot(self, ax):
        """sqrt(|residuals|) vs fitted, check homoscedasticity"""
        sns.scatterplot(x=self.fitted, y=np.sqrt(np.abs(self.reiduals)), ax=ax)
        ax.set_title("Scale-Location")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Sqrt(Residuals)")

    def res_vs_lev(self, ax):
        """outlier & influential point detection"""
        sns.scatterplot(x=self.leverage, y=self.residuals, ax=ax)
        ax.set_title("Residuals vs Leverage")
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Residuals")

    def cooks_distance(self, ax):
        """identify influential observations"""
        sns.scatterplot(x=np.arange(len(self.cooks)), y=self.cooks, ax=ax)
        ax.set_title("Cook's Distance")
        ax.set_xlabel("Observation")
        ax.set_ylabel("Cook's distance")

    def res_histogram(self, ax):
        """distribution check"""
        sns.histplot(self.residuals, kde=True, ax=ax)
        ax.set_title("Histogram of Residuals")

    def plot(self, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        self.res_vs_fitted(ax=axes[0, 0])
        self.normal_qq(ax=axes[0, 1])
        self.scale_loc_plot(ax=axes[0, 2])
        self.res_vs_lev(ax=axes[1, 0])
        self.cooks_distance(ax=axes[1, 1])
        self.res_histogram(ax=axes[1, 2])

        plt.suptitle("Model Diagnostics: 6-Plot", fontsize=16)
        plt.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}/model_diagnostics/eda_6plot.png", bbox_inches='tight')
        else:
            plt.show()

        plt.close(fig)


class InfluenceAndLeverageDiagnostics(ModelDiagnostics):
    def plot(self, save_path=None):
        pass


class ModelFitEvaluation(ModelDiagnostics):
    def plot(self, save_path=None):
        pass


class NonlinearityAndPartialEffects(ModelDiagnostics):
    def plot(self, save_path=None):
        pass


if __name__ == "__main__":
    # df = pd.read_csv("path-to-the dataframe")
    # ModelDiagnostics.run_model_diagnostics(df)
    pass