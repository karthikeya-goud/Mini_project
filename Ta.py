import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define custom places with latitude, longitude
places = [
    {'Place_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974, 'Name': 'Place_1'},  # New York City
    {'Place_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242, 'Name': 'Place_2'},  # Brooklyn, NYC
    {'Place_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428, 'Name': 'Place_3'},  # Manhattan, NYC
    {'Place_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604, 'Name': 'Place_4'},  # Midtown, NYC
    {'Place_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535, 'Name': 'Place_5'},  # Park Slope, Brooklyn
    {'Place_ID': 6, 'Latitude': 40.758896, 'Longitude': -73.985130, 'Name': 'Place_6'},  # Times Square
    {'Place_ID': 7, 'Latitude': 40.730610, 'Longitude': -73.999669, 'Name': 'Place_7'},  # East Village
    {'Place_ID': 8, 'Latitude': 40.692202, 'Longitude': -73.974375, 'Name': 'Place_8'},  # Downtown Brooklyn
    {'Place_ID': 9, 'Latitude': 40.7580, 'Longitude': -73.9855, 'Name': 'Place_9'},      # Midtown
    {'Place_ID': 10, 'Latitude': 40.7500, 'Longitude': -73.9800, 'Name': 'Place_10'},    # Chelsea
    {'Place_ID': 11, 'Latitude': 40.6895, 'Longitude': -74.0445, 'Name': 'Place_11'},    # Statue of Liberty
    {'Place_ID': 12, 'Latitude': 40.705627, 'Longitude': -73.9783, 'Name': 'Place_12'}   # Brooklyn Bridge Park
]

# Simulate traffic data and ad performance for each place and time period
time_interval = timedelta(minutes=30)  # Time intervals of 30 minutes
start_time = datetime(2025, 3, 21, 9, 0)  # Starting time: 9 AM
end_time = datetime(2025, 3, 21, 19, 0)  # Ending time: 7 PM

node_data = []

# Generate data for each time slot
current_time = start_time
while current_time <= end_time:
    for place in places:
        # Simulating the number of people (between 50 and 200)
        people_count = random.randint(50, 200)
        
        # Simulating the traffic condition (Low, Medium, High)
        traffic_condition = random.choice(['Low', 'Medium', 'High'])
        
        # Simulating ad engagement (Clicks, Impressions, and Engagement Rate)
        clicks = random.randint(5, 50)
        impressions = random.randint(100, 500)
        engagement_rate = clicks / impressions * 100 if impressions > 0 else 0
        
        # Appending the data for this timestamp and place
        node_data.append({
            'Timestamp': current_time,
            'Place_ID': place['Place_ID'],
            'Latitude': place['Latitude'],
            'Longitude': place['Longitude'],
            'People_Count': people_count,
            'Traffic_Condition': traffic_condition,
            'Clicks': clicks,
            'Impressions': impressions,
            'Engagement_Rate': engagement_rate
        })
    
    # Move to the next timestamp
    current_time += time_interval

# Create DataFrame
df = pd.DataFrame(node_data)

# Save DataFrame to CSV
df.to_csv('simulated_node_data.csv', index=False)

# Display first few rows of dataset
print(df.head())





from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

# Load the dataset
df = pd.read_csv('simulated_node_data.csv')

# Feature Engineering: Create the feature set X and target y
X = df[['People_Count', 'Traffic_Condition', 'Clicks', 'Impressions']]
y = df['Engagement_Rate']

# One-Hot Encoding for categorical features (Traffic_Condition)
X = pd.get_dummies(X, columns=['Traffic_Condition'], drop_first=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model
joblib.dump(rf_model, 'ad_engagement_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Saving the model and scaler for future use








import numpy as np
from itertools import permutations
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
rf_model = joblib.load('ad_engagement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Custom places and distances (distances are in km)
places = [
    {'Place_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974, 'Name': 'Place_1'},
    {'Place_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242, 'Name': 'Place_2'},
    {'Place_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428, 'Name': 'Place_3'},
    {'Place_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604, 'Name': 'Place_4'},
    {'Place_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535, 'Name': 'Place_5'},
    {'Place_ID': 6, 'Latitude': 40.758896, 'Longitude': -73.985130, 'Name': 'Place_6'},
    {'Place_ID': 7, 'Latitude': 40.730610, 'Longitude': -73.999669, 'Name': 'Place_7'},
    {'Place_ID': 8, 'Latitude': 40.692202, 'Longitude': -73.974375, 'Name': 'Place_8'},
    {'Place_ID': 9, 'Latitude': 40.7580, 'Longitude': -73.9855, 'Name': 'Place_9'},
    {'Place_ID': 10, 'Latitude': 40.7500, 'Longitude': -73.9800, 'Name': 'Place_10'},
    {'Place_ID': 11, 'Latitude': 40.6895, 'Longitude': -74.0445, 'Name': 'Place_11'},
    {'Place_ID': 12, 'Latitude': 40.705627, 'Longitude': -73.9783, 'Name': 'Place_12'}
]

# Distance Matrix (in kilometers)
distances = np.array([
    [0, 2.1, 1.8, 3.0, 4.5, 1.2, 1.0, 3.5, 2.5, 2.8, 3.2, 4.0],
    [2.1, 0, 1.5, 2.5, 3.8, 1.0, 1.2, 3.0, 1.7, 2.3, 2.8, 3.6],
    [1.8, 1.5, 0, 1.0, 2.2, 0.5, 0.7, 2.2, 1.2, 1.5, 1.9, 2.7],
    [3.0, 2.5, 1.0, 0, 1.5, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5, 3.0],
    [4.5, 3.8, 2.2, 1.5, 0, 1.5, 1.8, 2.2, 3.0, 3.5, 3.8, 4.2],
    [1.2, 1.0, 0.5, 1.0, 1.5, 0, 0.5, 1.8, 1.0, 1.2, 1.5, 2.0],
    [1.0, 1.2, 0.7, 1.2, 1.8, 0.5, 0, 1.5, 0.9, 1.0, 1.4, 1.8],
    [3.5, 3.0, 2.2, 1.5, 2.2, 1.8, 1.5, 0, 1.3, 1.5, 2.0, 2.4],
    [2.5, 1.7, 1.2, 2.0, 3.0, 1.0, 0.9, 1.3, 0, 1.4, 1.7, 2.2],
    [2.8, 2.3, 1.5, 2.3, 3.5, 1.2, 1.4, 2.0, 1.7, 1.9, 0, 1.8],
    [3.2, 2.8, 1.9, 2.5, 3.8, 1.5, 1.4, 2.4, 2.2, 2.5, 1.8, 0]
])

# Function to calculate ad engagement for a route
def calculate_ad_engagement(route, model, scaler, df_template):
    total_engagement = 0
    for place in route:
        # Predict engagement for this place
        X_new = df_template.loc[place].values.reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)
        engagement = model.predict(X_new_scaled)
        total_engagement += engagement
    return total_engagement

# Example: Find the best route between source and destination
source = 0  # Place_1 (index 0)
destination = 11  # Place_12 (index 11)
places_to_visit = list(set(range(len(places))) - {source, destination})

max_ad_engagement = -float('inf')
best_route = None
best_total_distance = None

# Generate all permutations and evaluate
from itertools import permutations
for perm in permutations(places_to_visit):
    route = [source] + list(perm) + [destination]
    total_distance = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
    
    ad_engagement = calculate_ad_engagement(route, rf_model, scaler, df)
    
    if ad_engagement > max_ad_engagement:
        max_ad_engagement = ad_engagement
        best_route = route
        best_total_distance = total_distance

# Print results
print(f"Best route: {[places[i]['Name'] for i in best_route]}")
print(f"Total distance: {best_total_distance} km")
print(f"Max Ad Engagement: {max_ad_engagement}")

# Visualize the best route
route_coordinates = [places[i] for i in best_route]
latitudes = [place['Latitude'] for place in route_coordinates]
longitudes = [place['Longitude'] for place in route_coordinates]

plt.figure(figsize=(10, 6))
plt.plot(longitudes, latitudes, marker='o', color='b', linestyle='-', markersize=5)
plt.title('Optimized Route for Vehicle Campaign')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
for i, place in enumerate(route_coordinates):
    plt.text(longitudes[i] + 0.01, latitudes[i] + 0.01, place['Name'], fontsize=9)
plt.grid(True)
plt.show()






import numpy as np
from itertools import permutations
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
rf_model = joblib.load('ad_engagement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Custom places and distances (distances are in km)
places = [
    {'Place_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974, 'Name': 'Place_1'},
    {'Place_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242, 'Name': 'Place_2'},
    {'Place_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428, 'Name': 'Place_3'},
    {'Place_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604, 'Name': 'Place_4'},
    {'Place_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535, 'Name': 'Place_5'},
    {'Place_ID': 6, 'Latitude': 40.758896, 'Longitude': -73.985130, 'Name': 'Place_6'},
    {'Place_ID': 7, 'Latitude': 40.730610, 'Longitude': -73.999669, 'Name': 'Place_7'},
    {'Place_ID': 8, 'Latitude': 40.692202, 'Longitude': -73.974375, 'Name': 'Place_8'},
    {'Place_ID': 9, 'Latitude': 40.7580, 'Longitude': -73.9855, 'Name': 'Place_9'},
    {'Place_ID': 10, 'Latitude': 40.7500, 'Longitude': -73.9800, 'Name': 'Place_10'},
    {'Place_ID': 11, 'Latitude': 40.6895, 'Longitude': -74.0445, 'Name': 'Place_11'},
    {'Place_ID': 12, 'Latitude': 40.705627, 'Longitude': -73.9783, 'Name': 'Place_12'}
]

# Distance Matrix (in kilometers)
distances = np.array([
    [0, 2.1, 1.8, 3.0, 4.5, 1.2, 1.0, 3.5, 2.5, 2.8, 3.2, 4.0],
    [2.1, 0, 1.5, 2.5, 3.8, 1.0, 1.2, 3.0, 1.7, 2.3, 2.8, 3.6],
    [1.8, 1.5, 0, 1.0, 2.2, 0.5, 0.7, 2.2, 1.2, 1.5, 1.9, 2.7],
    [3.0, 2.5, 1.0, 0, 1.5, 1.0, 1.2, 1.5, 2.0, 2.3, 2.5, 3.0],
    [4.5, 3.8, 2.2, 1.5, 0, 1.5, 1.8, 2.2, 3.0, 3.5, 3.8, 4.2],
    [1.2, 1.0, 0.5, 1.0, 1.5, 0, 0.5, 1.8, 1.0, 1.2, 1.5, 2.0],
    [1.0, 1.2, 0.7, 1.2, 1.8, 0.5, 0, 1.5, 0.9, 1.0, 1.4, 1.8],
    [3.5, 3.0, 2.2, 1.5, 2.2, 1.8, 1.5, 0, 1.3, 1.5, 2.0, 2.4],
    [2.5, 1.7, 1.2, 2.0, 3.0, 1.0, 0.9, 1.3, 0, 1.4, 1.7, 2.2],
    [2.8, 2.3, 1.5, 2.3, 3.5, 1.2, 1.4, 2.0, 1.7, 1.9, 0, 1.8],
    [3.2, 2.8, 1.9, 2.5, 3.8, 1.5, 1.4, 2.4, 2.2, 2.5, 1.8, 0]
])

# Function to calculate ad engagement for a route
def calculate_ad_engagement(route, model, scaler, df_template):
    total_engagement = 0
    for place in route:
        # Predict engagement for this place
        X_new = df_template.loc[place].values.reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)
        engagement = model.predict(X_new_scaled)
        total_engagement += engagement
    return total_engagement

# Example: Find the best route between source and destination
source = 0  # Place_1 (index 0)
destination = 11  # Place_12 (index 11)
places_to_visit = list(set(range(len(places))) - {source, destination})

max_ad_engagement = -float('inf')
best_route = None
best_total_distance = None

# Generate all permutations and evaluate
from itertools import permutations
for perm in permutations(places_to_visit):
    route = [source] + list(perm) + [destination]
    total_distance = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
    
    ad_engagement = calculate_ad_engagement(route, rf_model, scaler, df)
    
    if ad_engagement > max_ad_engagement:
        max_ad_engagement = ad_engagement
        best_route = route
        best_total_distance = total_distance

# Print results
print(f"Best route: {[places[i]['Name'] for i in best_route]}")
print(f"Total distance: {best_total_distance} km")
print(f"Max Ad Engagement: {max_ad_engagement}")

# Visualize the best route
route_coordinates = [places[i] for i in best_route]
latitudes = [place['Latitude'] for place in route_coordinates]
longitudes = [place['Longitude'] for place in route_coordinates]

plt.figure(figsize=(10, 6))
plt.plot(longitudes, latitudes, marker='o', color='b', linestyle='-', markersize=5)
plt.title('Optimized Route for Vehicle Campaign')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
for i, place in enumerate(route_coordinates):
    plt.text(longitudes[i] + 0.01, latitudes[i] + 0.01, place['Name'], fontsize=9)
plt.grid(True)
plt.show()
    

