import missingno as msno
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv(r"D:\DSC\archive (2)\city_hour.csv")

# Convert 'Datetime' to datetime format and sort data by 'City' and 'Datetime'
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values(by=['City', 'Datetime'])

# msno.matrix(data)
# plt.show()

# # Heatmap to analyze missing data patterns
# msno.heatmap(data)
# plt.show()

# Fill missing values for each city
for city in data['City'].unique():
    city_data = data[data['City'] == city]
    
    # Forward fill, backward fill, and linear interpolation
    city_data = city_data.ffill().bfill()
    city_data = city_data.infer_objects(copy=False).interpolate()
    
    # Update the data DataFrame with imputed values for this city
    data.loc[city_data.index, :] = city_data

# Denoising: Apply rolling mean with a window size of 3 (adjustable)
numeric_cols = data.select_dtypes(include=[np.number]).columns
# Remove 'AQI' column if it exists in numeric_cols
if 'AQI' in numeric_cols:
    numeric_cols = numeric_cols.drop('AQI')
print("Numeric columns:", numeric_cols)
data[numeric_cols] = data[numeric_cols].rolling(window=3, min_periods=1).mean()

# Outlier Removal: Remove data points with Z-scores > 3 in numeric columns
z_scores = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
data = data[(np.abs(z_scores) < 3).all(axis=1)]

# Min-Max Scaling: Apply Min-Max Scaling to the data
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])


# Save the dataset again after scaling the selected columns
data.to_csv("imputed_city_hour_scaled.csv", index=False)

print("Processed data with selected columns scaled saved to imputed_city_hour_scaled.csv.")