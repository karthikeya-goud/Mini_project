txt="""
import random
import pandas as pd
from datetime import datetime, timedelta

# Define nodes (latitude, longitude) - Your chosen locations (nodes)
nodes = [
    {'Node_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974},  # Node 1
    {'Node_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242},  # Node 2
    {'Node_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428},  # Node 3
    {'Node_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604},  # Node 4
    {'Node_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535},  # Node 5
]

# Define traffic levels (Low, Medium, High)
traffic_levels = ['Low', 'Medium', 'High']
weather_conditions = ['Clear', 'Rainy', 'Cloudy']

# Define time window for simulation (9 AM to 10 PM)
start_time = datetime(2025, 3, 21, 9, 0)
end_time = datetime(2025, 3, 21, 22, 0)
time_interval = timedelta(minutes=30)

# Prepare to generate data
timestamps = []
node_data = []

# Simulate data for each time slot (each 30 minutes)
current_time = start_time
while current_time <= end_time:
    for node in nodes:
        people_count = random.randint(50, 200)  # Random people count in node (simulated)
        traffic_condition = random.choice(traffic_levels)  # Random traffic level
        weather_condition = random.choice(weather_conditions)  # Random weather condition
        
        # Simulated ad performance
        ad_type = random.choice(['Banner', 'Video', 'Interactive'])
        clicks = random.randint(0, 50)
        impressions = random.randint(100, 500)
        engagement_rate = (clicks / impressions) * 100 if impressions > 0 else 0
        
        node_data.append({
            'Timestamp': current_time,
            'Node_ID': node['Node_ID'],
            'Latitude': node['Latitude'],
            'Longitude': node['Longitude'],
            'People_Count': people_count,
            'Traffic_Condition': traffic_condition,
            'Weather_Condition': weather_condition,
            'Ad_Type': ad_type,
            'Clicks': clicks,
            'Impressions': impressions,
            'Engagement_Rate (%)': engagement_rate
        })
    
    # Increment to the next timestamp (next 30 minutes)
    current_time += time_interval

# Convert to a DataFrame
df = pd.DataFrame(node_data)

# Save to CSV (optional)
df.to_csv('simulated_node_data.csv', index=False)

# Display the first few rows of the dataset
print(df.head())















import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Define a simple graph (nodes and edges) for route optimization
# Example: node_id -> [(neighbor_node_id, edge_weight)]
edges = {
    1: [(2, 2), (3, 5), (4, 3)],
    2: [(1, 2), (3, 4), (5, 6)],
    3: [(1, 5), (2, 4), (4, 2)],
    4: [(1, 3), (3, 2), (5, 7)],
    5: [(2, 6), (4, 7)]
}

# Create graph
G = nx.Graph()
for node, neighbors in edges.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# Implementing Dijkstra's algorithm for route optimization
def get_optimal_route(source, destination, traffic_conditions):
    traffic_weight = {'Low': 1, 'Medium': 2, 'High': 3}
    
    # Adjust edge weights based on traffic condition
    for u, v, data in G.edges(data=True):
        traffic_condition = traffic_conditions[(u, v)]
        data['weight'] = data['weight'] * traffic_weight[traffic_condition]

    # Find the shortest path using Dijkstra's algorithm
    optimal_path = nx.dijkstra_path(G, source, destination, weight='weight')
    return optimal_path

# Example: Find the optimal route from node 1 to node 5
traffic_conditions = {
    (1, 2): 'Medium', (1, 3): 'High', (1, 4): 'Low',
    (2, 1): 'Medium', (2, 3): 'Low', (2, 5): 'High',
    (3, 1): 'High', (3, 2): 'Low', (3, 4): 'Medium',
    (4, 1): 'Low', (4, 3): 'Medium', (4, 5): 'High',
    (5, 2): 'High', (5, 4): 'High'
}

# Get optimal route
optimal_route = get_optimal_route(1, 5, traffic_conditions)
print("Optimal route from node 1 to node 5:", optimal_route)
















from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Feature engineering
df['Traffic_Condition'] = df['Traffic_Condition'].map({'Low': 1, 'Medium': 2, 'High': 3})
df['Weather_Condition'] = df['Weather_Condition'].map({'Clear': 1, 'Cloudy': 2, 'Rainy': 3})
df['Ad_Type'] = df['Ad_Type'].map({'Banner': 1, 'Video': 2, 'Interactive': 3})

# Features and target
features = ['People_Count', 'Traffic_Condition', 'Weather_Condition', 'Ad_Type']
target = 'Engagement_Rate (%)'

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict ad engagement on test data
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Visualize predicted vs actual engagement rates
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Engagement Rate (%)')
plt.ylabel('Predicted Engagement Rate (%)')
plt.title('Actual vs Predicted Engagement Rate')
plt.show()


















# Graph for Ad Engagement vs Traffic Level
plt.figure(figsize=(10, 6))
df.groupby('Traffic_Condition')['Engagement_Rate (%)'].mean().plot(kind='bar')
plt.title('Average Ad Engagement Rate by Traffic Condition')
plt.ylabel('Average Engagement Rate (%)')
plt.xlabel('Traffic Condition')
plt.show()

# Graph for Route Visualization
plt.figure(figsize=(10, 6))
nodes_for_graph = list(G.nodes())
edges_for_graph = list(G.edges())

nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, font_size=12)
plt.title('Network Graph of Nodes and Routes')
plt.show()







import seaborn as sns

# Pivot the DataFrame to create a heatmap for traffic conditions over time
df_pivot = df.pivot_table(index='Timestamp', columns='Node_ID', values='Traffic_Condition', aggfunc='first')

# Convert traffic conditions to numeric (Low = 1, Medium = 2, High = 3)
df_pivot = df_pivot.applymap(lambda x: {'Low': 1, 'Medium': 2, 'High': 3}.get(x))

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_pivot, cmap='YlGnBu', cbar_kws={'label': 'Traffic Condition Level'}, annot=True, fmt="d")
plt.title('Traffic Condition Heatmap Over Time')
plt.xlabel('Node ID')
plt.ylabel('Timestamp')
plt.show()












# Box plot to show the distribution of ad engagement by weather condition
plt.figure(figsize=(10, 6))
sns.boxplot(x='Weather_Condition', y='Engagement_Rate (%)', data=df)
plt.title('Ad Engagement Rate by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Engagement Rate (%)')
plt.show()












# Count the traffic condition distribution per node
plt.figure(figsize=(10, 6))
sns.countplot(x='Node_ID', hue='Traffic_Condition', data=df)
plt.title('Traffic Condition Distribution Across Nodes')
plt.xlabel('Node ID')
plt.ylabel('Count')
plt.show()












# Plot the average ad engagement rate by timestamp
df_time_series = df.groupby('Timestamp')['Engagement_Rate (%)'].mean()

plt.figure(figsize=(12, 6))
df_time_series.plot()
plt.title('Average Ad Engagement Rate Over Time')
plt.xlabel('Time of Day')
plt.ylabel('Average Engagement Rate (%)')
plt.xticks(rotation=45)
plt.show()











# Visualizing the optimal route on the graph
optimal_path = get_optimal_route(1, 5, traffic_conditions)

# Visualize the path in the graph with custom color
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
node_labels = nx.get_node_attributes(G, 'node_id')
edge_labels = nx.get_edge_attributes(G, 'weight')

# Plot graph
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
nx.draw_networkx_edges(G, pos, edgelist=[(optimal_path[i], optimal_path[i+1]) for i in range(len(optimal_path)-1)], edge_color='r', width=3)
plt.title('Optimal Route (Shortest Path) with Traffic Impact')
plt.show()












import geopandas as gpd
from shapely.geometry import Point

# Create a GeoDataFrame with latitude and longitude for the nodes
geometry = [Point(xy[1], xy[0]) for xy in [(40.712776, -74.005974), (40.730610, -73.935242), (40.748817, -73.985428), (40.764351, -73.973604), (40.679356, -73.974535)]]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Plot the map with ad engagement size/color based on 'Engagement_Rate (%)'
gdf.plot(column='Engagement_Rate (%)', cmap='coolwarm', legend=True, markersize=100, figsize=(12, 8))
plt.title('Ad Engagement Across Nodes')
plt.show()












# Histogram for ad engagement distribution based on traffic condition
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Engagement_Rate (%)', hue='Traffic_Condition', multiple='stack', kde=True)
plt.title('Distribution of Ad Engagement Rate by Traffic Condition')
plt.xlabel('Engagement Rate (%)')
plt.ylabel('Frequency')
plt.show()












# Create a DataFrame for route traffic load and ad visibility
route_traffic = pd.DataFrame({
    'Route': ['1-2', '1-3', '1-4', '2-3', '2-5', '3-4', '3-5', '4-5'],
    'Traffic_Level': ['Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Medium'],
    'Ad_Visibility': [50, 80, 60, 40, 70, 50, 90, 60]  # Simulated ad visibility (1-100 scale)
})

# Plot route traffic load vs. ad visibility
plt.figure(figsize=(12, 6))
sns.barplot(x='Route', y='Ad_Visibility', hue='Traffic_Level', data=route_traffic)
plt.title('Route Traffic Load and Ad Visibility Comparison')
plt.ylabel('Ad Visibility (%)')
plt.xlabel('Route')
plt.show()











import random
import pandas as pd
from datetime import datetime, timedelta

# Define nodes (latitude, longitude)
nodes = [
    {'Node_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974},  # New York City
    {'Node_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242},  # Brooklyn, NYC
    {'Node_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428},  # Manhattan, NYC
    {'Node_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604},  # Midtown, NYC
    {'Node_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535},  # Park Slope, Brooklyn
]

# Define traffic levels
traffic_levels = ['Low', 'Medium', 'High']

# Define ad types
ad_types = ['Billboard', 'Mobile', 'Digital', 'Social']

# Define weather conditions
weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Windy']

# Number of days to simulate
n_days = 7  # You can change this to any number of days
time_interval = timedelta(minutes=30)

# Generate data for each day
node_data = []
start_time = datetime(2025, 3, 21, 9, 0)  # Start time (9 AM)
end_time = datetime(2025, 3, 21, 22, 0)  # End time (10 PM)

for day in range(n_days):
    current_time = start_time + timedelta(days=day)  # Shift by one day each time
    while current_time <= end_time:
        for node in nodes:
            # Random number of people (between 50 and 200)
            people_count = random.randint(50, 200)
            
            # Random traffic condition
            traffic_condition = random.choice(traffic_levels)
            
            # Random weather condition
            weather_condition = random.choice(weather_conditions)
            
            # Random ad type
            ad_type = random.choice(ad_types)
            
            # Simulating ad engagement and metrics (clicks, impressions)
            clicks = random.randint(5, 30)
            impressions = random.randint(50, 300)
            engagement_rate = (clicks / impressions) * 100 if impressions > 0 else 0
            
            # Append data for the current timestamp and node
            node_data.append({
                'Timestamp': current_time,
                'Node_ID': node['Node_ID'],
                'Latitude': node['Latitude'],
                'Longitude': node['Longitude'],
                'People_Count': people_count,
                'Traffic_Condition': traffic_condition,
                'Weather_Condition': weather_condition,
                'Ad_Type': ad_type,
                'Clicks': clicks,
                'Impressions': impressions,
                'Engagement_Rate (%)': engagement_rate
            })
        
        # Move to the next timestamp
        current_time += time_interval

# Convert to a DataFrame for easy analysis
df = pd.DataFrame(node_data)

# Save to CSV (optional)
df.to_csv('simulated_node_data_multiple_days.csv', index=False)

# Display the first few rows of the dataset
print(df.head())





import folium

# Create a base map centered on New York City
map_center = [40.730610, -73.935242]  # Brooklyn coordinates
m = folium.Map(location=map_center, zoom_start=12)

# Add nodes as markers on the map
for index, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Node_Name']} - {row['Traffic_Condition']} - {row['Ad_Type']}",
        tooltip=row['Node_Name']
    ).add_to(m)

# Save the map to an HTML file
m.save('custom_locations_map.html')




import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize random seed
random.seed(42)

# Define nodes (latitude, longitude)
nodes = [
    {'Node_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974},  # New York City
    {'Node_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242},  # Brooklyn, NYC
    {'Node_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428},  # Manhattan, NYC
    {'Node_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604},  # Midtown, NYC
    {'Node_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535},  # Park Slope, Brooklyn
]

# Define traffic levels and ad engagement
traffic_levels = ['Low', 'Medium', 'High']
ad_types = ['Banner Ad', 'Video Ad', 'Interactive Ad']

# Generate data for each node in 24 hours with 30-minute intervals
start_time = datetime(2025, 3, 21, 0, 0)
end_time = datetime(2025, 3, 21, 23, 30)
time_interval = timedelta(minutes=30)

timestamps = []
node_data = []

current_time = start_time
while current_time <= end_time:
    for node in nodes:
        # Random number of people (between 50 and 200)
        people_count = random.randint(50, 200)
        
        # Random traffic condition
        traffic_condition = random.choice(traffic_levels)
        
        # Random ad performance data
        ad_type = random.choice(ad_types)
        clicks = random.randint(0, 50)
        impressions = random.randint(100, 500)
        engagement_rate = (clicks / impressions) * 100 if impressions != 0 else 0
        
        # Append data for the current timestamp and node
        node_data.append({
            'Timestamp': current_time,
            'Node_ID': node['Node_ID'],
            'Latitude': node['Latitude'],
            'Longitude': node['Longitude'],
            'People_Count': people_count,
            'Traffic_Condition': traffic_condition,
            'Ad_Type': ad_type,
            'Clicks': clicks,
            'Impressions': impressions,
            'Engagement_Rate (%)': engagement_rate
        })
    
    current_time += time_interval

# Convert to DataFrame
df = pd.DataFrame(node_data)

# Route Optimization using OR-Tools
def create_data_model():
    """Creates the data for the routing problem."""
    data = {}
    data['distance_matrix'] = [
        # Simulated distance matrix (in km)
        [0, 5, 10, 8, 7],
        [5, 0, 6, 4, 3],
        [10, 6, 0, 5, 6],
        [8, 4, 5, 0, 2],
        [7, 3, 6, 2, 0]
    ]
    data['num_vehicles'] = 1  # Single vehicle for simplicity
    data['depot'] = 0  # Start at node 0
    return data

def compute_solution(data):
    """Solves the TSP problem using OR-Tools."""
    # Instantiate the routing model
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Create a distance matrix callback
    def distance_callback(from_index, to_index):
        # Returns the distance between the two nodes.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set up the search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Extract the route
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Add the depot as the last node
        return route
    else:
        return None

# Create data model and solve the problem
data = create_data_model()
route = compute_solution(data)

# Display the route
print("Route for the vehicle:", route)

# Visualization: Plot the route on a map
node_coords = {node['Node_ID']: (node['Latitude'], node['Longitude']) for node in nodes}
route_coords = [node_coords[node_id] for node_id in route]

# Plot the route
latitudes, longitudes = zip(*route_coords)
plt.figure(figsize=(8, 6))
plt.plot(longitudes, latitudes, marker='o', color='b', label='Route Path')
plt.title("Optimized Vehicle Route with Ad Visibility")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.legend()
plt.show()

# Ad Engagement Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Node_ID', y='Engagement_Rate (%)', data=df, ci=None, palette='viridis')
plt.title('Ad Engagement Rate across Nodes')
plt.xlabel('Node ID')
plt.ylabel('Engagement Rate (%)')
plt.show()

# Traffic Condition Distribution
traffic_condition_counts = df['Traffic_Condition'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=traffic_condition_counts.index, y=traffic_condition_counts.values, palette='coolwarm')
plt.title('Traffic Condition Distribution across Nodes')
plt.xlabel('Traffic Condition')
plt.ylabel('Frequency')
plt.show()

# Show some data for validation
print(df.head())


"""


"""import random
import pandas as pd
from datetime import datetime, timedelta
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import matplotlib.pyplot as plt
import seaborn as sns

# Custom 12 Locations (Latitudes, Longitudes)
nodes = [
    {'Node_ID': 1, 'Latitude': 40.712776, 'Longitude': -74.005974},  # Location 1
    {'Node_ID': 2, 'Latitude': 40.730610, 'Longitude': -73.935242},  # Location 2
    {'Node_ID': 3, 'Latitude': 40.748817, 'Longitude': -73.985428},  # Location 3
    {'Node_ID': 4, 'Latitude': 40.764351, 'Longitude': -73.973604},  # Location 4
    {'Node_ID': 5, 'Latitude': 40.679356, 'Longitude': -73.974535},  # Location 5
    {'Node_ID': 6, 'Latitude': 40.712800, 'Longitude': -74.006000},  # Location 6
    {'Node_ID': 7, 'Latitude': 40.750000, 'Longitude': -73.980000},  # Location 7
    {'Node_ID': 8, 'Latitude': 40.760000, 'Longitude': -73.970000},  # Location 8
    {'Node_ID': 9, 'Latitude': 40.710000, 'Longitude': -74.000000},  # Location 9
    {'Node_ID': 10, 'Latitude': 40.740000, 'Longitude': -73.990000}, # Location 10
    {'Node_ID': 11, 'Latitude': 40.780000, 'Longitude': -73.940000}, # Location 11
    {'Node_ID': 12, 'Latitude': 40.790000, 'Longitude': -73.960000}  # Location 12
]

# Define a custom distance matrix (in float, distances between each node in km)
distance_matrix = [
    [0, 5.0, 10.0, 8.0, 7.0, 4.5, 6.2, 8.5, 9.0, 7.8, 6.4, 7.2],
    [5.0, 0, 6.0, 4.5, 3.0, 5.2, 4.8, 6.5, 7.0, 5.3, 4.0, 5.8],
    [10.0, 6.0, 0, 5.5, 6.5, 9.0, 7.0, 8.0, 6.0, 6.5, 5.8, 7.0],
    [8.0, 4.5, 5.5, 0, 2.0, 5.5, 4.5, 6.0, 6.5, 4.0, 3.5, 4.0],
    [7.0, 3.0, 6.5, 2.0, 0, 4.0, 3.5, 5.0, 5.5, 4.2, 3.0, 4.5],
    [4.5, 5.2, 9.0, 5.5, 4.0, 0, 3.0, 4.0, 6.0, 5.5, 4.2, 5.0],
    [6.2, 4.8, 7.0, 4.5, 3.5, 3.0, 0, 2.5, 4.0, 3.5, 3.0, 4.2],
    [8.5, 6.5, 8.0, 6.0, 5.0, 4.0, 2.5, 0, 3.5, 5.0, 4.5, 5.0],
    [9.0, 7.0, 6.0, 6.5, 5.5, 6.0, 4.0, 3.5, 0, 5.5, 4.8, 5.5],
    [7.8, 5.3, 6.5, 4.0, 4.2, 5.5, 3.5, 5.0, 5.5, 0, 2.5, 4.5],
    [6.4, 4.0, 5.8, 3.5, 3.0, 4.2, 3.0, 4.5, 4.8, 2.5, 0, 3.2],
    [7.2, 5.8, 7.0, 4.0, 4.5, 5.0, 4.2, 5.0, 5.5, 4.5, 3.2, 0]
]

# Define traffic levels and ad engagement
traffic_levels = ['Low', 'Medium', 'High']
ad_types = ['Banner Ad', 'Video Ad', 'Interactive Ad']

# Function to create data model for OR-Tools
def create_data_model():
    """Creates the data for the routing problem."""
    data = {}
    data['distance_matrix'] = distance_matrix  # Your custom distance matrix
    data['num_vehicles'] = 1  # One vehicle for simplicity
    data['depot'] = 0  # Starting node (can change)
    return data

# Function to solve the TSP problem using OR-Tools
def compute_solution(data):
    """Solves the TSP problem using OR-Tools."""
    # Create routing model
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # Create a distance callback
    def distance_callback(from_index, to_index):
        # Get the nodes corresponding to the indices
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define arc cost (minimize distance)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Extract the route
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Add depot as the last node
        return route
    else:
        return None

# Create data model and solve the problem
data = create_data_model()
route = compute_solution(data)

# Display the optimized route
print("Optimized Route (Node IDs):", route)

# Visualize the route (Node IDs mapped to coordinates)
node_coords = {node['Node_ID']: (node['Latitude'], node['Longitude']) for node in nodes}
route_coords = [node_coords[node_id] for node_id in route]

# Plot the route on a map
latitudes, longitudes = zip(*route_coords)
plt.figure(figsize=(8, 6))
plt.plot(longitudes, latitudes, marker='o', color='b', label='Route Path')
plt.title("Optimized Vehicle Route")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.legend()
plt.show()

# Ad Engagement Visualization for each node (simulated)
df = pd.DataFrame(columns=["Node_ID", "Traffic_Condition", "Ad_Type", "Engagement_Rate (%)"])
for node in nodes:
    # Random traffic condition and ad type assignment
    traffic_condition = random.choice(traffic_levels)
    ad_type = random.choice(ad_types)
    clicks = random.randint(0, 50)
    impressions = random.randint(100, 500)
    engagement_rate = (clicks / impressions) * 100 if impressions != 0 else 0

    # Add simulated ad data to dataframe
    df = df.append({
        "Node_ID": node['Node_ID'],
        "Traffic_Condition": traffic_condition,
        "Ad_Type": ad_type,
        "Engagement_Rate (%)": engagement_rate
    }, ignore_index=True)

# Plot Ad Engagement Rate
plt.figure(figsize=(10, 6))
sns.barplot(x='Node_ID', y='Engagement_Rate (%)', data=df, ci=None, palette='viridis')
plt.title('Ad Engagement Rate across Nodes')
plt.xlabel('Node ID')
plt.ylabel('Engagement Rate (%)')
plt.show()

# Traffic Condition Distribution
traffic_condition_counts = df['Traffic_Condition'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=traffic_condition_counts.index, y=traffic_condition_counts.values, palette='coolwarm')
plt.title('Traffic Condition Distribution across Nodes')
plt.xlabel('Traffic Condition')
plt.ylabel('Frequency')
plt.show()
"""