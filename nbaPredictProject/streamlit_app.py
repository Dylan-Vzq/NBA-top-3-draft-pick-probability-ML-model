import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.neighbors import NearestNeighbors

#Load trained model, scalers, and feature columns
model = joblib.load("top3_model.pkl")
scaler = joblib.load("scaler.pkl")
graph_scaler = joblib.load("graph_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
visualization_data = joblib.load("visualization_data.pkl")

#Load player data for similarity graph
player_data = pd.read_csv("top3_draft_picks_done.csv")

#Features used for similarity calculation
graph_feature_cols = ["College PPG", "College RPG", "College APG", "College FG%", "Age at Draft", "Final_height"]


#Function to calculate graph features for a new player
def calculate_graph_features(new_player_stats, existing_data, graph_scaler, k_neighbors=5):
    
    #Get existing player features
    existing_features = existing_data[graph_feature_cols].values
    
    #Create array with new player's stats
    new_features = np.array([[
        new_player_stats["College PPG"],
        new_player_stats["College RPG"],
        new_player_stats["College APG"],
        new_player_stats["College FG%"],
        new_player_stats["Age at Draft"],
        new_player_stats["Final_height"]
    ]])
    
    #Combine existing players with new player
    all_features = np.vstack([existing_features, new_features])
    
    #Scale all features
    all_scaled = graph_scaler.transform(all_features)
    
    #Fit k-NN to find similar players
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    knn.fit(all_scaled)
    distances, indices = knn.kneighbors(all_scaled)
    
    #Build similarity graph
    G = nx.Graph()
    for i in range(len(all_features)):
        G.add_node(i)
    
    #Add edges between similar players
    for i in range(len(all_features)):
        for j_idx in range(1, len(indices[i])):
            j = indices[i][j_idx]
            dist = distances[i][j_idx]
            weight = 1.0 / (1.0 + dist)
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=weight)
    
    #Calculate graph features for new player (last index)
    new_player_idx = len(all_features) - 1
    degree_cent = nx.degree_centrality(G)[new_player_idx]
    clustering_coef = nx.clustering(G)[new_player_idx]
    
    return degree_cent, clustering_coef


#Sidebar title
st.sidebar.title("Top-3 Draft Pick Predictor")

#Player input section
st.sidebar.subheader("Enter Player Stats")

player_name = st.sidebar.text_input("Player Name", "  ")
position_input = st.sidebar.multiselect("Position(s)", ["PG", "SG", "SF", "PF", "C"], default=["PG"])
age = st.sidebar.slider("Age at Draft", 17, 30, 19)
height_ft = st.sidebar.slider("Height - Feet", 5, 7, 6)
height_in = st.sidebar.slider("Height - Inches", 0, 11, 6)
ppg = st.sidebar.number_input("College PPG", 0.0, 50.0, 15.0)
rpg = st.sidebar.number_input("College RPG", 0.0, 20.0, 5.0)
apg = st.sidebar.number_input("College APG", 0.0, 15.0, 3.0)
fg_percent = st.sidebar.slider("College FG%", 30.0, 80.0, 50.0)

#Page title
st.title("NBA Draft Analysis")
st.write("Enter player stats in the sidebar to predict the chance of being a Top-3 pick.")

#Prediction logic
if st.sidebar.button("Predict Top-3 Likelihood"):
    height_total_inches = height_ft * 12 + height_in

    #Prepare stats for graph feature calculation
    new_player_stats = {
        "College PPG": ppg,
        "College RPG": rpg,
        "College APG": apg,
        "College FG%": fg_percent,
        "Age at Draft": age,
        "Final_height": height_total_inches
    }
    
    #Auto-calculate graph features based on player similarity
    degree_centrality, clustering = calculate_graph_features(
        new_player_stats, player_data, graph_scaler, k_neighbors=5
    )

    #Create input dictionary with all features
    input_dict = {
        "College PPG": ppg,
        "College RPG": rpg,
        "College APG": apg,
        "College FG%": fg_percent,
        "Age at Draft": age,
        "Final_height": height_total_inches,
        "degree_centrality": degree_centrality,
        "clustering": clustering
    }

    #One-hot encode positions
    for pos in [col for col in feature_columns if col.startswith("Pos_")]:
        input_dict[pos] = 1 if pos.split("_")[1] in position_input else 0

    #Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    #Scale numeric features
    scaled_cols = ["College PPG", "College RPG", "College APG", "College FG%", "Age at Draft", "Final_height"]
    input_df_scaled = input_df.copy()
    input_df_scaled[scaled_cols] = scaler.transform(input_df[scaled_cols])

    #Predict and clip result to [0, 1]
    prediction = model.predict(input_df_scaled)[0]
    prediction = max(0, min(1, prediction))
    percent = round(prediction * 100, 2)

    #Show prediction result
    st.success(f"Prediction for {player_name}: {percent}% chance of being a Top-3 pick!")

    #Show input summary
    st.write("### Input Summary")
    st.write(f"**Auto-calculated Graph Features:** Degree Centrality = {degree_centrality:.3f}, Clustering = {clustering:.3f}")
    st.dataframe(input_df)

    #Bar chart comparison
    st.write("### Top-3 Pick Likelihood Comparison")

    #Calculate average likelihood for each pick position
    avg_first_pick = player_data[player_data["Top3Likelihood"] == 1.0]["Top3Likelihood"].mean()
    avg_second_pick = player_data[player_data["Top3Likelihood"] == 0.85]["Top3Likelihood"].mean()
    avg_third_pick = player_data[player_data["Top3Likelihood"] == 0.70]["Top3Likelihood"].mean()

    comparison_data = {
        "This Player": prediction,
        "Avg #1 Pick": avg_first_pick,
        "Avg #2 Pick": avg_second_pick,
        "Avg #3 Pick": avg_third_pick
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(comparison_data.keys(), [v * 100 for v in comparison_data.values()], 
                  color=['#4CAF50', '#2196F3', '#FF9800', '#E91E63'])
    ax.set_ylabel("Top-3 Likelihood (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Your Player vs. Historical Draft Picks")
    
    #Add value labels on bars
    for bar, val in zip(bars, comparison_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val*100:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("""
    **Bar Chart Explanation:**
    - **This Player**: Your prediction based on entered stats
    - **Avg #1 Pick**: Average likelihood for #1 overall picks (100%)
    - **Avg #2 Pick**: Average likelihood for #2 overall picks (85%)
    - **Avg #3 Pick**: Average likelihood for #3 overall picks (70%)
    """)

#Model visualizations section
st.write("---")
st.write("## Model Visualizations")

#Create two columns for graphs
col1, col2 = st.columns(2)

with col1:
    st.write("### Correlation Heatmap")
    st.write("Shows how player features correlate with each other.")
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(visualization_data["correlation_matrix"], annot=True, fmt=".2f", 
                cmap="viridis", linewidths=.5, square=True, cbar_kws={"shrink": .8}, ax=ax1)
    ax1.set_title("Player Feature Correlation Heatmap", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.write("### Actual vs Predicted")
    st.write("Shows how well the model predicts on test data.")
    
    y_test = visualization_data["y_test"]
    y_pred = visualization_data["y_pred"]
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(y_test, y_pred, color="red", s=100, edgecolors="white", linewidths=1.2)
    
    #Diagonal line for perfect predictions
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], color="lime", linestyle="dashdot", linewidth=2)
    
    ax2.set_xlabel("Actual Top3Likelihood")
    ax2.set_ylabel("Predicted Top3Likelihood")
    ax2.set_title("Linear Regression: Actual vs Predicted", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig2)

st.write("---")
st.caption("NBA Draft Predictor")
