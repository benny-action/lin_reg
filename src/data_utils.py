import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)

def generate_housing_data(n_samples=1000):
    """
    Gen synthetic housing data with realistic relationships
    """

    square_footage = np.random.normal(2000, 800, n_samples)
    square_footage = np.clip(square_footage, 500, 5000)

    bedrooms = np.random.choice([1, 2, 3, 4, 5, ], n_samples, p=[0.1, 0.25, 0.35, 0.35, 0.05])

    age = np.random.exponential(15, n_samples)
    age = np.clip(age, 0, 100)

    location_score = np.random.normal(6, 2, n_samples)
    location_score = np.clip(location_score, 1, 10)

    base_price = (
        square_footage * 150 +
        bedrooms * 15000 +
        -age * 1000 +
        location_score * 20000
    )

    noise = np.random.normal(0, 25000, n_samples)
    market_factor= np.random.normal(1.0, 0.1, n_samples)
    
    price = (base_price + noise) * market_factor
    price = np.clip(price, 50000, 1000000)

    data = pd.DataFrame({
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'age': age,
        'location_score': location_score,
        'price': price
        })

    return data

def explore_data(data):
    """
    Visualise the data, understand the data.
    """
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    fig.suptitle('Housing Data Exploration', fontsize=16)
    
    #price distribution
    #sqfootage vs price
    #bedrooms v price
    #age v price
    #location v price
    #correlation heatmap?
