import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#dataframe
df = pd.read_csv("top3_draft_picks_done.csv")

#Columns that define player similarity (need to be pre draft features)
graph_features_columns = ["College PPG", "College RPG", "College APG", "College FG%", "Age at Draft", "Final_height"]

#functiuon to Build networkX graph (node=player / edges=connect 2 players if they're close k matches)
def build_player_similarity_graph(df, feature_columns, k_neighbors=5):

    #Make matrix of dataframe features
    X = df[feature_columns].values

    #Make features be on the same scale (standardization)
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)

    #K-nearest neighbors in order to find most similar players to each other based on the features, using Euclidean as similarity measure
    #+1 represents closest "neighbor" of a player (technically himself)
    near_neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    #Feeding matrix; now the model can calculate the distances between the players
    near_neighbors.fit(X_scale)

    #Distances=how close players are / indices=which players are the closest (as their row numbers)
    distances, indices = near_neighbors.kneighbors(X_scale)

    #Start graph and add a node per player
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx, name=row["Player"])

    #Now add the edges between each player and the player's k nearest neighbors, skipping the 1st one because it's the player himself
    for i in range(len(df)):
        neighbor_ind = indices[i][1:]
        neighbor_dist = distances[i][1:]

        #Weight how close a player is similar to another (improves the model and helps for precision)
        for j, dist in zip(neighbor_ind, neighbor_dist):
            sim_weight = 1.0 / (1.0 + dist)

            #Avoid duplicate edges
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=sim_weight)

    #Return the graph done & the scaler (need the scaler to convert new players too)
    return G, scaler

#Build graph w/ df
G, std_scaler = build_player_similarity_graph(df, graph_features_columns, k_neighbors=5)
print("# of nodes:", G.number_of_nodes())
print("# of edges:", G.number_of_edges())

#Graph feature: degree centrality: for each node, provides a value between 0 and 1 (close t 1 meaning connected to many and close to 0 meaning connected to few)
degree_centrality_feature = nx.degree_centrality(G)
#Graph feature: clustering: if high clustered meaning his neighbors are too connected; otherwise, are not that connected
clustering_feature = nx.clustering(G)

#Put the values of the graph back into the dataframe rows. We use the row index to match the values of the graph to the rows of the df
df["degree_centrality"] = df.index.to_series().map(degree_centrality_feature)
df["clustering"] = df.index.to_series().map(clustering_feature)

#Hot encoding in the column of positions (because there's more than a position in some players; split each into list
df["position_list"] = df["Position"].astype(str).str.split("/")
#now build list but of all the positions
every_positions = []
for position_list in df["position_list"]:
    for position in position_list:
        if position not in every_positions:
            every_positions.append(position)

every_positions = sorted(every_positions)

#Initialize column per each of the positions
for position in every_positions:
    df[f"Pos_{position}"] = 0

#Marks each of the player's actual positions with 1 in the hot encoded columns; this makes the model understand each player's role
for idx, position_list in df["position_list"].items():
    for position in position_list:
        df.loc[idx, f"Pos_{position}"] = 1

print("Encoded positions added: ")
print(every_positions)

#Feature column selection
basic_cols = ["College PPG", "College RPG", "College APG", "College FG%", "Age at Draft", "Final_height", "degree_centrality", "clustering",]

#Take the new position columns
positions_cols = [col for col in df.columns if col.startswith("Pos_")]
feature_cols = basic_cols + positions_cols

#x = feature columns / y = regression target
X = df[feature_cols]
y = df["Top3Likelihood"]

print("X shape:", X.shape)
print("Feature columns:", feature_cols)

#Scale numeric features so all stats have equal influence on predictions
numeric_cols_to_scale = ["College PPG", "College RPG", "College APG", "College FG%", "Age at Draft", "Final_height"]
model_scaler = StandardScaler()

#Create scaled copy of X
X_scaled = X.copy()
X_scaled[numeric_cols_to_scale] = model_scaler.fit_transform(X[numeric_cols_to_scale])

print("Features scaled for model training.")

#Train and test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Create linear regression model
l_model = LinearRegression()

#Train it
l_model.fit(X_train, y_train)

#Prediction on set
y_linear_predict = l_model.predict(X_test)

#Now evaluates it with the metrics
mae = mean_absolute_error(y_test, y_linear_predict)
r2 = r2_score(y_test, y_linear_predict)
mse = mean_squared_error(y_test, y_linear_predict)
rmse = np.sqrt(mse)

print("\n>--------> Linear Regression Model <--------<")
print("MAE:", mae)
print("R2 score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)

#Create Random Forest regression model
forest_model = RandomForestRegressor(n_estimators=200, random_state=42)

#Train it
forest_model.fit(X_train, y_train)

#Prediction on set
y_forest_predict = forest_model.predict(X_test)

#Now evaluates it with the same metrics as the linear regression one
mae_forest = mean_absolute_error(y_test, y_forest_predict)
mse_forest = mean_squared_error(y_test, y_forest_predict)
rmse_forest = np.sqrt(mse_forest)
r2_forest = r2_score(y_test, y_forest_predict)

print("\n>--------> Random Forest Model <--------<")
print("MAE:", mae_forest)
print("MSE:", mse_forest)
print("RMSE:", rmse_forest)
print("R2 score:", r2_forest)

#Retrain linear regression on entire dataset for final model
final_model = LinearRegression()
final_model.fit(X_scaled, y)

#Save feature columns order for Streamlit
exact_and_final_feature_cols = feature_cols

print("\nLinear Regression Model ready for real prediction usage")
print("Final feature columns:", exact_and_final_feature_cols)

#correlation heatmap: graph 1 (shows how all the numeric features correlates to each other)
correlating_columns = ["College PPG", "College RPG", "College APG", "College FG%", "Age at Draft", "Final_height",]
correlating_columns_matrix = df[correlating_columns].corr()

#plot the correlation heatmap graph
plt.figure(figsize=(10,7))
sns.heatmap(correlating_columns_matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, square=True, cbar_kws={"shrink": .8})
plt.title("Player Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

#Actual vs predicted values plot: graph 2 (shows how close the model's predictions are to the real values)
#Reuse the linear regression model's predictions
y_plot_linear_pred = l_model.predict(X_test)
plt.figure(figsize=(7,7))

#scatter plot
plt.scatter(y_test, y_plot_linear_pred, color="red", s=125, edgecolors="white", linewidths=1.2)

#diagonal line
minimum_value = min(y_test.min(), y_plot_linear_pred.min())
maximum_value = max(y_test.max(), y_plot_linear_pred.max())
plt.plot([minimum_value, maximum_value], [minimum_value, maximum_value], color="lime", linestyle="dashdot")

#show graph
plt.xlabel("Actual Top3Likelihood")
plt.ylabel("Predicted Top3Likelihood")
plt.title("Linear Regression: Actual Top3Likelihood vs. Predicted Top3Likelihood")
plt.tight_layout()
plt.show()

#Save final trained model
joblib.dump(final_model, "top3_model.pkl")

#Save model scaler for scaling input features
joblib.dump(model_scaler, "scaler.pkl")

#Save graph scaler for calculating graph features
joblib.dump(std_scaler, "graph_scaler.pkl")

#Save feature columns list
joblib.dump(exact_and_final_feature_cols, "feature_columns.pkl")

#Save visualization data for Streamlit graphs
test_visualization_data = {
    "y_test": y_test.values,
    "y_pred": y_linear_predict,
    "correlation_matrix": correlating_columns_matrix
}
joblib.dump(test_visualization_data, "visualization_data.pkl")

print("Model, Scalers, Feature Columns, and Visualization Data saved successfully.")







