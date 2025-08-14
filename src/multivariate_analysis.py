import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120


class BaseMultivariate(ABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
        self.cat_cols = df.select_dtypes(exclude=np.number).columns.to_list()

    @abstractmethod
    def plot(self, save_path=None):
        pass

    @staticmethod
    def run_multi_eda(df: pd.DataFrame, save_path=None):
        if save_path:
            os.makedirs(f"{save_path}/multivariate_analysis", exist_ok=True)
        multi_eda = MultivariateEDA(df)
        multi_eda.plot(save_path=save_path)

class MultivariateEDA(BaseMultivariate):
    def vif_analysis(self, save_path=None, vif_thresh=10):
        """Compute and display VIF for numeric features."""
        if len(self.numeric_cols) < 2:
            print("Not enough numeric variables for VIF.")
            return
        
        X = self._df[self.numeric_cols].dropna()

        vif_data = pd.DataFrame({
            "feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })
        vif_data["Multicollinearity Detected"] = vif_data["VIF"] > vif_thresh

        print("\n--- Variance Inflation Factor ---")
        print(vif_data.sort_values(by="VIF", ascending=False))

        if save_path:
            vif_data.to_csv(f"{save_path}/vif_analysis.csv", index=False)


    def pca_analysis(self, n_components=None, hue=None, save_path=None):
        """Run PCA and plot variance explained"""
        if len(self.numeric_cols) < 2:
            print("Not enough numeric variables for this analysis.")
            return
        
        X = self._df[self.numeric_cols].dropna()
        pca = PCA()
        pca.fit(X)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1, len(pca.explained_variance_ratio_)+1),
                np.cumsum(pca.explained_variance_ratio_), marker='o')
        ax.set_title("PCA - Cumulative Explained Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.grid(True)

        if save_path:
            fig.savefig(f"{save_path}/pca_scree.png", bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

        # PCA scatter for first two components — try for multiple hues
        if X.shape[1] >= 2:
            pcs = pca.transform(X)
            pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"])
            for hue in self.cat_cols:
                if hue in self._df.columns:
                    pc_df[hue] = self._df.loc[X.index, hue]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue=hue, ax=ax, alpha=0.7)
                    ax.set_title(f"PCA - First Two Components (Hue: {hue})")

                    if save_path:
                        fig.savefig(f"{save_path}/pca_scatter_{hue}.png", bbox_inches='tight')
                    else:
                        plt.show()
                    plt.close(fig)

    def pairplot(self, hue=None, max_vars=6, save_path=None):
        """Plot pairplot for numeric variables"""
        if len(self.numeric_cols) < 2:
            print("Not enough numeric variables for this analysis.")
            return
        
        # Select top `max_vars` by variance
        var_order = self._df[self.numeric_cols].var().sort_values(ascending=False).index
        selected_cols = var_order[:max_vars]

        g = sns.pairplot(self._df[selected_cols], hue=hue, diag_kind='hist')
        if save_path:
            g.savefig(f"{save_path}/pairplot.png")
        else:
            plt.show()
        plt.close()

    def plot(self, save_path=None):
        if save_path:
            os.makedirs(f"{save_path}/multivariate_analysis", exist_ok=True)

        self.vif_analysis(save_path=save_path)
        self.pca_analysis(save_path=save_path)
        self.pairplot(save_path=save_path)


if __name__ == "__main__":
    # df = pd.read_csv("path-to-the-dataframe")
    # BaseMultivariate.run_multi_eda(df)
    pass