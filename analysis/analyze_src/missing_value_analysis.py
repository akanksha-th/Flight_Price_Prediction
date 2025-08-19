import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MissingValueHeatmap:
    """
    Generates a heatmap of missing values in a DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self._df = df

    def plot(self, figsize=(12,6), cmap='virdis', save_path=None):
        """
        Plots the missing value heatmap

        Args:
            figsize: size of the figure
            cmap: color map for the heatmap
            save_path: saves the plot to this path
        """

        plt.figure(figsize=figsize)
        sns.heatmap(self._df.isnull(), cbar=True, cmap=cmap)

        plt.title("Missing Values Heatmap", fontsize=14)
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        if save_path:
            plt.savefig(save_path, bbox_incjes="tight")

        else:
            plt.show()
            plt.close()


if __name__ == "__main__":
    # df = pd.DataFrame({
    #     "A": [1, 2, None, 4],
    #     "B": [None, 2, 3, None],
    #     "C": [1, None, 3, 4]
    # })
    # mvh = MissingValueHeatmap(df)
    # mvh.plot(figsize=(8,5), cmap='magma')
    pass