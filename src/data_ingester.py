import os
import pandas as pd
import opendatasets as od
import zipfile
from abc import ABC, abstractmethod
from typing import Optional


class DataIngester(ABC):
    """
    Abstract base class for data ingestion from various sources and format.
    """

    def __init__(self, source: str, download_dir: str = "./data"):
        """
        :param source: URL, filepath etc
        :param download_dir: directory to store the extracted or downloaded data.
        """
        self.source = source
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None

        @abstractmethod
        def ingest(self) -> pd.DataFrame:
            """Ingests data and returns a dataframe"""
            pass

        @staticmethod
        def _read_csv(filepath: str, **kwargs) -> pd.DataFrame:
            return pd.read_csv(filepath, **kwargs)
        
        @staticmethod
        def _read_json(filepath: str, **kwargs) -> pd.DataFrame:
            return pd.read_json(filepath, **kwargs)
        
        @staticmethod
        def _read_parquet(filepath: str, **kwargs) -> pd.DataFrame:
            return pd.read_parquet(filepath, **kwargs)
        
        @staticmethod
        def _extract_zip(zip_path: str, extract_to: str):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)


class FileDataIngester(DataIngester):
    """
    Ingest data from a local file.
    Supports CSV, JSON, Parquet, ZIP.
    """

    def ingest(self) -> pd.DataFrame:
        ext = os.path.splitext(self.source)[1].lower() # Extracts the file extnsion

        if ext == ".csv":
            self.df = self._read_csv(self.source)
        elif ext == ".json":
            self.df = self._read_json(self.source)
        elif ext == ".parquet":
            self.df = self._read_parquet(self.source)
        elif ext == ".zip":
            extract_path = os.path.join(self.download_dir, "extracted")
            os.makedirs(extract_path, exist_ok=True)
            self._extract_zip(self.source, extract_path)

            # Search extracted files for a known format
            for root, _, files in os.walk(extract_path):
                for f in files:
                    f_ext = os.path.splitext(f)[1].lower()
                    if f_ext in ['.csv', '.json', '.parquet']:
                        filepath = os.path.join(root, f)
                        if ext == ".csv":
                            self.df = self._read_csv(self.source)
                        elif ext == ".json":
                            self.df = self._read_json(self.source)
                        elif ext == ".parquet":
                            self.df = self._read_parquet(self.source)
                        return self.df
            raise FileNotFoundError("No supported data file inside ZIP archive")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return self.df


class URLDataIngester(DataIngester):
    """
    Downloads dataset from URL or open datasets repo and loads into a DataFrame.
    Uses opendatasets for supported URLs.
    """

    def ingest(self) -> pd.DataFrame:
        od.download(self.source, self.download_dir)

        for root, _, files in os.walk(self.download_dir):
            for f in files:
                f_ext = os.path.splitext(f)[1].lower()
                if f_ext in [".csv", ".json", ".parquet"]:
                    file_path = os.path.join(root, f)
                    if f_ext == ".csv":
                        self.df = self._read_csv(file_path)
                    elif f_ext == ".json":
                        self.df = self._read_json(file_path)
                    elif f_ext == ".parquet":
                        self.df = self._read_parquet(file_path)
                    return self.df
        raise FileNotFoundError("No supported data file found after download.")


def get_data_ingester(source: str, download_dir: str = './data') -> DataIngester:
    if source.startswith("http") or source.startswith("www"):
        return URLDataIngester(source, download_dir)
    elif os.path.isfile(source):
        return FileDataIngester(source, download_dir)
    else:
        raise ValueError(f"Could not recognize source type for: {source}")


if __name__ == "__main__":
    source_path_or_url = "your-dataset-url-or-path"
    ingester = get_data_ingester(source_path_or_url)
    df = ingester.ingest()
    print(df.head())