import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from abc import ABC, abstractmethod

sns.set_style("whitegrid")
"""
DPI (dots per inch) = number of pixels per inch of the figure
Higher DPI → sharper and more detailed images
Lower DPI → faster rendering, but less crisp
"""
plt.rcParams["figure.dpi"] = 120


class BaseEDAPlot(ABC):
    def __init__(self, df: pd.DataFrame, col_name: str):
        self._df = df
        if col_name not in self._df.columns:
            raise ValueError("Column- {col_name} not found in the provided dataframe.")
        self._col = self._df[col_name].dropna()
        self._col_name = col_name

    @abstractmethod
    def plot(self, deep=False, save_path=None):
        pass

    @staticmethod
    def auto_run(df: pd.DataFrame, deep=False, save_path=None):
        if save_path:
            os.makedirs(f"{save_path}/univariate_analysis", exist_ok=True)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                eda = NumericEDA(df, col)
            else:
                eda = CategoricalEDA(df, col)
            eda.plot(deep=deep, save_path=save_path)


# ===== Numeric EDA Class =====
class NumericEDA(BaseEDAPlot):
    def run_sequence_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self._col.values, marker='o', linestyle='-')
        ax.set_title("Run Sequence Plot")
        ax.set_xlabel("Index")
        ax.set_ylabel(self._col_name)
    
    def lag_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        pd.plotting.lag_plot(self._col, ax=ax)
        ax.set_title("Lag Plot")

    def histogram(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        ax.hist(self._col, bins=12, edgecolor='black')
        ax.set_title("Histogram")

    def normal_probability_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        stats.probplot(self._col, dist='norm', plot=ax)
        ax.set_title("Normal Probability Plot")

    def boxplot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        sns.boxplot(x=self._col, ax=ax)
        ax.set_title("Boxplot")

    def kde_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        sns.kdeplot(self._col, fill=True, ax=ax)
        ax.set_title("KDE / Density Plot")

    def _diagnostics(self):
        """Quick statistical checks for numeric assumptions"""
        print(f"--Assumption Check for {self._col_name}")

        # 1. Drift check
        x = np.arange(len(self._col))
        slope, _, _, _, _ = stats.linregress(x, self._col)
        drift_msg = "No strong drift" if abs(slope) < 0.01 else "Possible drift detected"
        print(f"Run Sequence: slope={slope:.4f} -> {drift_msg}")

        # 2. Autocorrelation
        autocorr = self._col.autocorr()
        auto_msg = "Random" if abs(autocorr) < 0.2 else "Autocorrelation present"
        print(f"Lag Plot: lag-1 autocorrelation: {autocorr:.3f} -> {auto_msg}")

        # 3. Skewness & kurtosis
        skewness = stats.skew(self._col)
        kurt = stats.kurtosis(self._col)
        print(f"Histogram: skewness={skewness:.3f}, kurtosis={kurt:.3f}")

        # 4. Normality --- stats.shapiro() will throw an error if the column has fewer than 3 unique values
        try:
            _, shapiro_p = stats.shapiro(self._col)
            normal_msg = "Normal distribution" if shapiro_p > 0.05 else "Not-normal distribution"
        except ValueError:
            shapiro_p = np.nan
            normal_msg = "Too few data points for normality test"
        print(f"Normal Probability Plot: Shapiro-Wilk p={shapiro_p:.3f} -> {normal_msg}")

    def plot(self, deep=False, save_path=None):
        if deep:
            fig, axes = plt.subplots(3, 2, figsize=(12, 12))

            self.run_sequence_plot(ax=axes[0, 0])
            self.lag_plot(ax=axes[0, 1])
            self.histogram(ax=axes[1, 0])
            self.normal_probability_plot(ax=axes[1, 1])
            self.boxplot(ax=axes[2, 0])
            self.kde_plot(ax=axes[2, 1])

            fig.suptitle(f"Deep Univariate Analysis: {self._col_name}", fontsize=16)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10,8))

            self.run_sequence_plot(ax=axes[0, 0])
            self.lag_plot(ax=axes[0, 1])
            self.histogram(ax=axes[1, 0])
            self.normal_probability_plot(ax=axes[1, 1])

            fig.suptitle(f"4-Plot: {self._col_name}", fontsize=16)

        plt.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}/univariate_analysis/{self._col_name}_numeric_plot.png", bbox_inches='tight')
        else:
            plt.show()

        plt.close(fig)
        self._diagnostics()


# ===== Categorical EDA Class =====
class CategoricalEDA(BaseEDAPlot):
    def countplot(self, counts, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        if (counts.shape[0] <= 25):
            sns.countplot(x=self._col, order=counts.index, ax=ax)
            ax.set_title(f"Count Plot of {self._col_name}")

        else:
            print("\nToo many categories to plot clearly - showing top 25")
            sns.countplot(x=self._col, order=counts.index[:25], ax=ax)
            ax.set_title(f"Top 25 Categories of {self._col_name}")
        ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels(), rotation=80)

    def percentage_bar_plot(self, counts, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f"Percentage Bar Plot of {self._col_name}")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(ax.get_xticks(), labels=ax.get_xticklabels(), rotation=80)

    def plot(self, deep=False, save_path=None):
        counts = self._col.value_counts()
        unique_vals = counts.shape[0] # or self._col.nunique
        print(f"Summary for {self._col_name} -- Unique Values: {unique_vals}")

        if deep:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5))
            self.countplot(counts, ax=axes[0])
            self.percentage_bar_plot(counts, ax=axes[1])
            fig.suptitle(f"Deep Univariate Analysis - {self._col_name}", fontsize=16)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            self.countplot(counts, ax=ax)
            fig.suptitle(f"Count Plot - {self._col_name}", fontsize=16)

        plt.tight_layout()

        if save_path:
            fig.savefig(f"{save_path}/univariate_analysis/{self._col_name}_cotegorical_plot.png", bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)

        # Imbalance check
        top_ratio = counts.iloc[0]/counts.sum() # How much percentage of all data lies in the top category
        if top_ratio > 0.8:
            print("Strong Imbalance detected - dominant category > 80%")
        elif top_ratio > 0.5:
            print("Moderate Imbalance -dominant category > 50%")
        else:
            print("Balanced Distribution")


if __name__ == "__main__":
    # df = pd.read_csv("path-to-csv-file")
    # BaseEDAPlot.auto_run(df, deep=True, save_path=path)
    pass