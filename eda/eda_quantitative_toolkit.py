import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from IPython.display import display, HTML

# This EDA toolkit will include the following:
    # 1. Descriptive Statistics
    # 2. Univariate, Bivariate, Multivariate Analysis
    # 3. Hypothesis Testing
    # 4. Data Visualizations
    # 5. Correlations (numeric and categorical)
    # 6. Missing Data Hnadling
    # 7. Outlier Detection


def display_html(size=3, content="content"):
    display(HTML(f"<h{size}>{content}>/h{size}>"))

def eda_4plot(y, title="EDA 4-PLOT"):
    fig, ax = plt.subplots(2,2, figsize =(12,10))
    fig.suptitle(title)
    
    # Run Sequence Plot
    ax[0, 0].plot(y, marker='o', )
    pass