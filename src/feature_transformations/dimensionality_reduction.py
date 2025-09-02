from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------- Config Dataclass ----------------------
@dataclass
class DimReducerConfig:
    method: str = "pca"       # 'pca', 'kernel_pca', 'factor', 'tsne', 'umap', 'svd', 'autoencoder'
    n_components: int = 2
    kernel: str = "rbf"       # for KernelPCA
    random_state: int = 42
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 16
    autoencoder_lr: float = 1e-3

# ---------------------- Base Class ----------------------
class BaseDimReducer(ABC):
    def __init__(self, df: pd.DataFrame, config: DimReducerConfig):
        self._df = df.copy()
        self.config = config
        self._reducer = None
        self._numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Scale data before reduction
        self._scaler = StandardScaler()
        self._scaled_data = self._scaler.fit_transform(self._df[self._numeric_cols])

    @abstractmethod
    def fit_transform(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# ---------------------- PCA ----------------------
class PCAClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._reducer = PCA(n_components=config.n_components, random_state=config.random_state)

    def fit_transform(self):
        components = self._reducer.fit_transform(self._scaled_data)
        return pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(self.config.n_components)])

    def transform(self, df):
        scaled = self._scaler.transform(df[self._numeric_cols])
        components = self._reducer.transform(scaled)
        return pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(self.config.n_components)])

# ---------------------- Kernel PCA ----------------------
class KernelPCAClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._reducer = KernelPCA(n_components=config.n_components, kernel=config.kernel, random_state=config.random_state)

    def fit_transform(self):
        components = self._reducer.fit_transform(self._scaled_data)
        return pd.DataFrame(components, columns=[f"KPCA{i+1}" for i in range(self.config.n_components)])

    def transform(self, df):
        scaled = self._scaler.transform(df[self._numeric_cols])
        components = self._reducer.transform(scaled)
        return pd.DataFrame(components, columns=[f"KPCA{i+1}" for i in range(self.config.n_components)])

# ---------------------- Factor Analysis ----------------------
class FactorAnalysisClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._reducer = FactorAnalysis(n_components=config.n_components, random_state=config.random_state)

    def fit_transform(self):
        components = self._reducer.fit_transform(self._scaled_data)
        return pd.DataFrame(components, columns=[f"FA{i+1}" for i in range(self.config.n_components)])

    def transform(self, df):
        scaled = self._scaler.transform(df[self._numeric_cols])
        components = self._reducer.transform(scaled)
        return pd.DataFrame(components, columns=[f"FA{i+1}" for i in range(self.config.n_components)])

# ---------------------- Truncated SVD ----------------------
class TruncatedSVDClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._reducer = TruncatedSVD(n_components=config.n_components, random_state=config.random_state)

    def fit_transform(self):
        components = self._reducer.fit_transform(self._scaled_data)
        return pd.DataFrame(components, columns=[f"SVD{i+1}" for i in range(self.config.n_components)])

    def transform(self, df):
        scaled = self._scaler.transform(df[self._numeric_cols])
        components = self._reducer.transform(scaled)
        return pd.DataFrame(components, columns=[f"SVD{i+1}" for i in range(self.config.n_components)])

# ---------------------- t-SNE ----------------------
class TSNEClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._reducer = TSNE(n_components=config.n_components, random_state=config.random_state)

    def fit_transform(self):
        components = self._reducer.fit_transform(self._scaled_data)
        return pd.DataFrame(components, columns=[f"TSNE{i+1}" for i in range(self.config.n_components)])

    def transform(self, df):
        raise NotImplementedError("t-SNE does not support transform. Use fit_transform on full dataset.")

# ---------------------- UMAP ----------------------
class UMAPClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self._reducer = umap.UMAP(n_components=config.n_components, random_state=config.random_state)

    def fit_transform(self):
        components = self._reducer.fit_transform(self._scaled_data)
        return pd.DataFrame(components, columns=[f"UMAP{i+1}" for i in range(self.config.n_components)])

    def transform(self, df):
        scaled = self._scaler.transform(df[self._numeric_cols])
        components = self._reducer.transform(scaled)
        return pd.DataFrame(components, columns=[f"UMAP{i+1}" for i in range(self.config.n_components)])

# ---------------------- Autoencoder ----------------------
class AutoencoderClass(BaseDimReducer):
    def __init__(self, df, config):
        super().__init__(df, config)
        self.n_input = self._scaled_data.shape[1]
        self.n_latent = config.n_components
        self.epochs = config.autoencoder_epochs
        self.batch_size = config.autoencoder_batch_size
        self.lr = config.autoencoder_lr
        self.model = self._build_model()

    def _build_model(self):
        class AE(nn.Module):
            def __init__(self, n_input, n_latent):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_input, max(n_latent * 2, n_input // 2)),
                    nn.ReLU(),
                    nn.Linear(max(n_latent * 2, n_input // 2), n_latent)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(n_latent, max(n_latent * 2, n_input // 2)),
                    nn.ReLU(),
                    nn.Linear(max(n_latent * 2, n_input // 2), n_input)
                )
            def forward(self, x):
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon, z
        return AE(self.n_input, self.n_latent)

    def fit_transform(self):
        X = torch.tensor(self._scaled_data, dtype=torch.float32)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                x_batch = batch[0]
                x_recon, _ = self.model(x_batch)
                loss = criterion(x_recon, x_batch)
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            _, latent = self.model.encoder(X), self.model.encoder(X)
            latent = latent.numpy()
        return pd.DataFrame(latent, columns=[f"AE{i+1}" for i in range(self.n_latent)])

    def transform(self, df):
        X = torch.tensor(self._scaler.transform(df[self._numeric_cols]), dtype=torch.float32)
        with torch.no_grad():
            latent = self.model.encoder(X)
        return pd.DataFrame(latent.numpy(), columns=[f"AE{i+1}" for i in range(self.n_latent)])



class DimReducerFactory:
    @staticmethod
    def get_reducer(df: pd.DataFrame, config: DimReducerConfig) -> BaseDimReducer:
        method_map = {
            'pca': PCAClass,
            'kernel_pca': KernelPCAClass,
            'factor': FactorAnalysisClass,
            'svd': TruncatedSVDClass,
            'tsne': TSNEClass,
            'umap': UMAPClass,
            'autoencoder': AutoencoderClass
        }
        if config.method not in method_map:
            raise ValueError(f"Unknown dimensionality reduction method: {config.method}")
        return method_map[config.method](df, config)

# ---------------------- Example ----------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        'distance': [100, 200, 300, 400],
        'price': [50, 80, 60, 90],
        'delay': [5, 15, 10, 20]
    })
    config = DimReducerConfig(method='pca', n_components=2)
    reducer = DimReducerFactory.get_reducer(df, config)
    df_reduced = reducer.fit_transform()
    print(df_reduced)