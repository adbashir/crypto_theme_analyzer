# Project by Adnan Bashir 
# Start Date : 1st October, 2023
# End Date : 10th October, 2023




import yfinance as yf  # Yahoo Finance to download crypto data
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
from scipy.spatial.distance import euclidean  # Euclidean distance for similarity assessment
from sklearn.preprocessing import StandardScaler  # Data normalization
from sklearn.linear_model import LinearRegression  # Linear regression for trend analysis
import matplotlib.pyplot as plt  # Data visualization
from matplotlib.colors import to_rgba_array  # Color handling for plots

# Function to download cryptocurrency data using Yahoo Finance
def download_crypto_data(crypto_list, start_date, end_date):
    data = {}
    for crypto in crypto_list:
        ticker = f"{crypto}-USD"  # Format ticker for USD market
        data[crypto] = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to create data windows for analysis
WINDOW_SIZE = 50  # Size of each data window for analysis
def create_windows(data, window_size):
    windows = {}
    for crypto in data:
        series = data[crypto]['Close'].dropna()  # Use closing prices, drop NA values
        # Create overlapping windows of data
        windows[crypto] = [series[i:i+window_size].values for i in range(0, len(series)-window_size+1, window_size)]
    return windows

# Calculate the trend slope using linear regression
def calculate_trend_slope(window):
    X = np.arange(len(window)).reshape(-1, 1)  # Time as independent variable
    y = window.reshape(-1, 1)  # Price as dependent variable
    model = LinearRegression().fit(X, y)  # Fit linear model
    return model.coef_[0][0]  # Return slope of the fitted line

# Feature extraction with normalization and scaling
def extract_features(window, distinctiveness_factor):
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    normalized_window = scaler.fit_transform(window.reshape(-1, 1)).flatten()
    features = {
        'mean': np.mean(normalized_window),
        'std_dev': np.std(normalized_window) * distinctiveness_factor,  # Adjust std deviation
        'trend_slope': calculate_trend_slope(window) * distinctiveness_factor  # Adjust trend slope
    }
    return np.array(list(features.values()))

# Extract features for all windows across all cryptocurrencies
def extract_all_features(windows, distinctiveness_factor):
    feature_windows = {}
    for crypto in windows:
        # Extract and scale features for each window
        feature_windows[crypto] = np.array([extract_features(w, distinctiveness_factor) for w in windows[crypto]])
    return feature_windows

# Initialize themes and set up for dynamic discovery
MAX_THEMES = 10  # Maximum number of themes to identify
themes = {crypto: [] for crypto in crypto_list}  # Initialize themes for each cryptocurrency

# Function to find the closest theme based on Euclidean distance
def find_closest_theme(feature_vec, crypto_themes, distinctiveness_factor):
    min_distance = float('inf')
    closest_theme = None
    for theme_id, theme_features in enumerate(crypto_themes):
        # Adjust distance calculation
        distance = euclidean(feature_vec, theme_features) * (1 - distinctiveness_factor)
        if distance < min_distance:
            min_distance = distance
            closest_theme = theme_id
    return closest_theme, min_distance

# Update themes based on new data
def update_themes(feature_windows, themes, distinctiveness_factor):
    threshold = 0.5 * (1 - distinctiveness_factor)  # Adjust threshold for theme similarity
    for crypto in feature_windows:
        for feature_vec in feature_windows[crypto]:
            closest_theme, distance = find_closest_theme(feature_vec, themes[crypto], distinctiveness_factor)
            if len(themes[crypto]) < MAX_THEMES and (closest_theme is None or distance > threshold):
                themes[crypto].append(feature_vec)  # Add new theme
            elif closest_theme is not None and distance <= threshold:
                # Update theme with new data, weighted by distinctiveness_factor
                themes[crypto][closest_theme] = ((1 - distinctiveness_factor) * themes[crypto][closest_theme] +
                                                  distinctiveness_factor * feature_vec)

# Visualize themes with semi-transparent windows
def plot_all_themes(original_data, windows, themes, distinctiveness_factor):
    for crypto in original_data:
        plt.figure(figsize=(15, 8))
        unique_colors = to_rgba_array(plt.cm.tab10.colors)  # Distinct colors for themes

        date_index = original_data[crypto].index  # Get dates for x-axis

        for i, window in enumerate(windows[crypto]):
            start_idx = i * WINDOW_SIZE
            if start_idx + WINDOW_SIZE > len(date_index):
                continue  # Skip incomplete windows
            end_idx = start_idx + WINDOW_SIZE
            window_dates = date_index[start_idx:end_idx]

            theme_id, _ = find_closest_theme(extract_features(window, distinctiveness_factor), themes[crypto], distinctiveness_factor)
            if theme_id is not None:
                plt.fill_between(window_dates, window, color=unique_colors[theme_id % len(unique_colors)], alpha=0.3 + 0.7 * distinctiveness_factor)

        plt.plot(date_index, original_data[crypto]['Close'], label='Original Data', color='black', linewidth=2)
        plt.title(f"Windows and Themes for {crypto}")  # Title with crypto name
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# Main execution script
distinctiveness_factor = 0.2  # Adjust for window distinctiveness
crypto_list = ['BTC', 'ETH', 'BNB', 'USDT', 'SOL']  # List of cryptocurrencies to analyze
data = download_crypto_data(crypto_list, '2018-01-01', '2023-01-01')  # Download data
windows = create_windows(data, WINDOW_SIZE)  # Create analysis windows
feature_windows = extract_all_features(windows, distinctiveness_factor)  # Extract features
update_themes(feature_windows, themes, distinctiveness_factor)  # Update themes
plot_all_themes(data, windows, themes, distinctiveness_factor)  # Visualize themes
