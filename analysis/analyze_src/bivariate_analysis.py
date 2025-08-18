import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
from scipy import stats
from abc import ABC, abstractmethod

sns.set_style('whitegrid')
plt.rcParams["figure.dpi"] = 120


class BaseBivariatePlot(ABC):
    def __init__(self, df: pd.DataFrame, col1: str, col2: str):
        self._df = df
        missing = [col for col in (col1, col2) if col not in df.columns]
        if missing:
            raise ValueError("Check columns again. Not found in the DataFrame")
        self._col1 = col1
        self._col2= col2
    
    @abstractmethod
    def plot(self, save_path=None):
        pass

    @staticmethod
    def run_biv_eda(df: pd.DataFrame, save_path=None, corr_thresh=0.3, p_thresh=0.05):
        if save_path:
            os.makedirs(f"{save_path}/bivariate_analysis", exist_ok=True)
        
        col_pairs = itertools.combinations(df.columns, 2)

        for col1, col2 in col_pairs:
            # Determine types
            col1_is_num = pd.api.types.is_numeric_dtype(df[col1])
            col2_is_num = pd.api.types.is_numeric_dtype(df[col2])

            if col1_is_num and col2_is_num:
                corr = df[col1].corr(df[col2])
                if abs(corr) < corr_thresh:
                    continue
                bi_eda = NumericVsNumericEDA(df, col1, col2)

            elif col1_is_num and not col2_is_num:
                p_val = NumericVsCategoricalEDA.quick_pval(df, col1, col2)
                if p_val > p_thresh:
                    continue
                bi_eda = NumericVsCategoricalEDA(df, col1, col2)
            
            elif not col1_is_num and col2_is_num:
                p_val = NumericVsCategoricalEDA.quick_pval(df, col2, col1)
                if p_val > p_thresh:
                    continue
                bi_eda = NumericVsCategoricalEDA(df, col2, col1)
            
            else:
                p_val = CategoricalVsCategoricalEDA.quick_pval(df, col1, col2)
                if p_val > p_thresh:
                    continue
                bi_eda = CategoricalVsCategoricalEDA(df, col1, col2)
                
            bi_eda.plot(save_path=save_path)


class NumericVsNumericEDA(BaseBivariatePlot):
    """ Only plot if |correlation| > threshold """
    # def scatter_plot(self, ax=None):
    #     if ax is None:
    #         _, ax = plt.subplots()
    #     sns.scatterplot(x=self._df[self._col1], y=self._df[self._col2], ax=ax, s=40)
    #     ax.set_title("Scatter Plot")
    #     ax.set_xlabel(self._col1)
    #     ax.set_ylabel(self._col2)

    def scatter_with_reg(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        sns.regplot(x=self._df[self._col1], y=self._df[self._col2], ax=ax, scatter_kws={'s':30}, line_kws={'color': 'red'})
        ax.set_title("Scatter Plot with Regression Line")
        ax.set_xlabel(self._col1)
        ax.set_ylabel(self._col2)

    def hexbin_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        hb = ax.hexbin(self._df[self._col1], self._df[self._col2], gridsize=30, cmap='Blues', mincnt=1)
        ax.set_title("Hexbin Plot")
        ax.set_xlabel(self._col1)
        ax.set_ylabel(self._col2)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("Counts")

    def _correlation_stats(self):
        """Print Pearson & Spearman correlation."""
        col1_data = self._df[self._col1]
        col2_data = self._df[self._col2]

        pearson_corr, pearson_p = stats.pearsonr(col1_data, col2_data)
        spearman_corr, spearman_p = stats.spearmanr(col1_data, col2_data)

        print(f"--- Correlation Stats for {self._col1} vs {self._col2} ---")
        print(f"Pearson Correlation: {pearson_corr:.3f} (p={pearson_p:.3e})")
        print(f"Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")

    def plot(self, save_path=None):
        # fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # self.scatter_plot(ax=axes[0, 0])
        # self.scatter_with_reg(ax=axes[0, 1])
        # self.hexbin_plot(ax=axes[1, 0])
        # fig.delaxes(axes[1, 1])

        fig, axes = plt.subplots(1, 2, figsize=(8, 5))

        self.scatter_with_reg(ax=axes[0])
        self.hexbin_plot(ax=axes[1])
        
        plt.suptitle(f"Bivariate Analysis (Numeric vs Numeric): {self._col1} vs {self._col2}", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}/bivariate_analysis/{self._col1}_vs_{self._col2}_numeric.png", bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)
        self._correlation_stats()


class NumericVsCategoricalEDA(BaseBivariatePlot):

    @staticmethod
    def quick_pval(df, numeric, categorical):
        groups = [grp.dropna().values for _, grp in df.groupby(categorical, observed=True)[numeric]]
        # Filter out tiny groups
        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) < 2:
            return np.nan  # too few data points for test
        try:
            _, p_val = stats.f_oneway(*valid_groups)
        except ValueError:
            _, p_val = stats.kruskal(*valid_groups)
        return p_val
    
    def boxplot(self, ax=None):
        """Shows distribution of numeric values for each category."""
        if ax is None:
            _, ax = plt.subplots()
        sns.boxplot(x=self._df[self._col2], y=self._df[self._col1], ax=ax)
        ax.set_title(f"Boxplot: {self._col1} vs {self._col2}")
        ax.tick_params(axis='x', rotation=70)

    def violin_plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        sns.violinplot(x=self._df[self._col2], y=self._df[self._col1], ax=ax)
        ax.set_title(f"Violinplot: {self._col1} vs {self._col2}")
        ax.tick_params(axis='x', rotation=70)

    def barplot_mean(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        sns.barplot(x=self._df[self._col2], y=self._df[self._col1], estimator=np.mean, ax=ax)
        ax.set_title(f"Mean {self._col1} per {self._col2}")
        ax.tick_params(axis='x', rotation=70)

    def stripplot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        sns.stripplot(x=self._df[self._col2], y=self._df[self._col1], ax=ax, jitter=True, size=4)
        ax.set_title(f"Stripplot: {self._col1} vs {self._col2}")
        ax.tick_params(axis='x', rotation=70)

    def _stats(self):
        p_val = self.quick_pval(self._df, self._col1, self._col2)
        print(f"--- {self._col1} vs {self._col2} ---")
        print(f"ANOVA/Kruskal p-value: {p_val:.3e}")

    def plot(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(10,8))

        self.boxplot(ax=axes[0, 0])
        self.violin_plot(ax=axes[0, 1])
        self.barplot_mean(ax=axes[1, 0])
        self.stripplot(ax=axes[1, 1])
        
        plt.suptitle(f"Bivariate Analysis (Numeric vs Categorical): {self._col1} vs {self._col2}", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}/bivariate_analysis/{self._col1}_vs_{self._col2}_num&cat.png", bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)
        self._stats()


class CategoricalVsCategoricalEDA(BaseBivariatePlot):

    @staticmethod
    def quick_pval(df, col1, col2):
        table = pd.crosstab(df[col1], df[col2])
        _, p_val, _, _ = stats.chi2_contingency(table)
        return p_val
    
    def stacked_bar(self, ax):
        table = pd.crosstab(self._df[self._col1], self._df[self._col2])
        table.div(table.sum(1), axis=0).plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Stacked Bar Chart")
        ax.set_ylabel("Proportion")

    def heatmap(self, ax):
        table = pd.crosstab(self._df[self._col1], self._df[self._col2])
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Count Heatmap")

    def _stats(self):
        p_val = self.quick_pval(self._df, self._col1, self._col2)
        print(f"--- {self._col1} vs {self._col2} ---")
        print(f"Chi-square p-value: {p_val:.3e}")
    
    def plot(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))

        self.stacked_bar(ax=axes[0])
        self.heatmap(ax=axes[1])
        
        plt.suptitle(f"Bivariate Analysis (Categorical vs Categorical): {self._col1} vs {self._col2}", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(f"{save_path}/bivariate_analysis/{self._col1}_vs_{self._col2}_categorical.png", bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)
        self._stats()


if __name__ == "__main__":
    # df = pd.read_csv("path-to-csv-file")
    # BaseBivariatePlot.run_biv_eda(df, corr_thresh=0.3, p_thresh=0.05)
    pass