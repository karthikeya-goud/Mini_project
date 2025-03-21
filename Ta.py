import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Custom 12 places with latitude, longitude, and distance (in float)
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

# Distance Matrix (example, in kilometers)
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

# Time simulation (9 AM to 7 PM)
start_time = datetime(2025, 3, 21, 9, 0)
end_time = datetime(2025, 3, 21, 19, 0)
time_interval = timedelta(hours=1)

# Data simulation (for N days)
N = 3
timestamps = []
node_data = []

for day in range(N):
    current_day = start_time + timedelta(days=day)
    current_time = current_day

    while current_time <= end_time:
        for node in places:
            # Simulate number of people, traffic, weather, ad type, and engagement
            people_count = random.randint(50, 200)
            traffic_condition = random.choice(['Low', 'Medium', 'High'])
            weather_condition = random.choice(['Clear', 'Cloudy', 'Rainy'])
            temperature = random.randint(10, 35)
            ad_type = random.choice(['Billboard', 'Digital Screen', 'Hologram'])
            ad_engagement = random.choice([0.5, 1.0, 1.5, 2.0])
            clicks = random.randint(0, 50)
            impressions = random.randint(50, 200)
            engagement_rate = (clicks / impressions) * 100 if impressions != 0 else 0

            node_data.append({
                'Timestamp': current_time,
                'Place_ID': node['Place_ID'],
                'Latitude': node['Latitude'],
                'Longitude': node['Longitude'],
                'People_Count': people_count,
                'Traffic_Condition': traffic_condition,
                'Weather_Condition': weather_condition,
                'Temperature': temperature,
                'Ad_Type': ad_type,
                'Ad_Engagement': ad_engagement,
                'Clicks': clicks,
                'Impressions': impressions,
                'Engagement_Rate (%)': engagement_rate
            })

        current_time += time_interval

df = pd.DataFrame(node_data)
df.to_csv('simulated_node_data.csv', index=False)

# Feature engineering for modeling
X = df[['People_Count', 'Traffic_Condition', 'Temperature', 'Clicks', 'Impressions']]
X = pd.get_dummies(X, drop_first=True)
y = df['Ad_Engagement']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Model evaluation
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Visualization of predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest: True vs Predicted Ad Engagement')
plt.grid(True)
plt.show()

# OR-Tools: Route Optimization (Traveling Salesman Problem)
def create_data_model():
    data = {}
    data['distance_matrix'] = distances
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def main():
    data = create_data_model()
    
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(manager, routing, solution)

def print_solution(manager, routing, solution):
    print('Route:')
    index = routing.Start(0)
    route_distance = 0
    route = []
    
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index)
    
    route.append(manager.IndexToNode(index))
    print(f"Route distance: {route_distance} km")
    print(f"Route: {route}")

# Running the route optimization
main()
