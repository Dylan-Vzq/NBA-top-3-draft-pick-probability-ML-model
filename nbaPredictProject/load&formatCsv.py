import pandas as pd
import numpy as np

#File loading
csv_path = "top3_draft_picks.csv"

#Read height as string to avoid to convert it to a date
df = pd.read_csv(csv_path, dtype={"Height (ft)": "string"})

print("Columns:", df.columns.tolist())
print("Rows:", len(df))

#Remove % and convert the percentage to numeric
percent_columns = ["Rookie NBA FG%", "College FG%"]

for col in percent_columns:
    # remove %
    df[col] = df[col].str.replace("%", "", regex=False)
    # numeric conversion
    df[col] = pd.to_numeric(df[col], errors="coerce")

#Convert height to feet (it will stay like this[5-5] for the display but the model will deal with a float
height = df["Height (ft)"].astype(str).str.split("-", expand=True)

#Conversion
df["Height_feet"] = pd.to_numeric(height[0], errors="coerce")
df["Height_inches"] = pd.to_numeric(height[1], errors="coerce")

#conversion (to decimal feet)
df["Final_height"] = df["Height_inches"] + (df["Height_feet"] * 12)

#Create regression target (Top3Likelihood)
df["Overall Pick"] = pd.to_numeric(df["Overall Pick"], errors="coerce")
draft_pick = {1: 1.00, 2: 0.85, 3: 0.70}
#transform values
df["Top3Likelihood"] = df["Overall Pick"].map(draft_pick)

#Preview and outputs a clean dataframe for the model
print("\nSample:")
print(df.head())

print("\nColumns done:")
print(df.columns.tolist())

#save new csv
df.to_csv("top3_draft_picks_done.csv", index=False)
print("\nSaved new csv: top3_draft_picks_done.csv")