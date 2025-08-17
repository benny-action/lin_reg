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
    axes[0, 0].hist(data['price'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    #sqfootage vs price
    axes[0, 1].scatter(data['square_footage'], data['price'], alpha=0.6, s=20)
    axes[0, 1].set_title('Square Footage vs Price')
    axes[0, 1].set_xlabel('Square Footage')
    axes[0, 1].set_ylabel('Price ($)')

    #bedrooms v price
    data.boxplot(column='price', by='bedrooms', ax=axes[0, 2])
    axes[0, 2].set_title('Price by Number of Bedrooms')
    axes[0, 2].set_xlabel('Bedrooms')

    #age v price
    axes[1, 0].scatter(data['age'], data['price'], alpha=0.6, s=20, color='orange')
    axes[1, 0].set_title('Age vs Price')
    axes[1, 0].set_xlabel('Age (years)')
    axes[1, 0].set_ylabel('Price ($)')

    #location v price
    axes[1, 1].scatter(data['location_score'], data['price'], alpha=0.6, s=20, color='green')
    axes[1, 1].set_title('Location Score vs Price')
    axes[1, 1].set_xlabel('Location Score')
    axes[1, 1].set_ylabel('Price ($)')

    #correlation heatmap?
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes(1, 2))
    axes[1, 2].set_title('Feature Correlations')

    plt.tight_layout()
    plt.show()

    print("\n === Data Summary ===")
    print(data.describe())
    print(f"\nDataset shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
def prepare_data_for_training(data, test_split=0.2):
    """
    Prepare data for training - split and convert into pytorch tensors
    """
    features = ['square_footage', 'bedrooms', 'age', 'location_score']
    X = data[features].values
    y = data['price'].values.reshape(-1, 1) #2d ify that thing

    n_train = int(len(data)) * (1 - test_split)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    #finish

