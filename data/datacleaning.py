import pandas as pd

df = pd.read_csv('./data/comparisondata.csv')

# Print exact column names
print("CSV Columns:", df.columns.tolist())



# Define columns you want to drop
columns_to_drop = ["Type of Report","Performance Measure","Hospital Ratings","Latitude","Longitude"]

# Drop only the columns that actually exist in the DataFrame
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Save the cleaned data to a new CSV (or overwrite the existing one)
df.to_csv("./data/comparisondata.csv", index=False)