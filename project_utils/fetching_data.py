import opendatasets as od
import os

url = 'https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction'
data_dir = 'data'

# Ensure the directory exists
os.makedirs(data_dir, exist_ok=True)

# Download dataset into the specified directory
od.download(dataset_id_or_url=url, data_dir=data_dir)