import pandas as pd
# Read the JSON file and convert it to a Pandas DataFrame 
# pd.read_json:  a method that reads a JSON and converts it to a DataFrame (df)
# df: a variable that holds the DataFrame
df = pd.read_csv('./data/hospitaldatamodified.csv')

# Keep only rows where county is 'San Diego'
df = df[df["County"] == "San Diego"]

# Save the filtered data back to a CSV file (optional)
df.to_csv("./data/hospitaldatamodified.csv", index=False)

# Print the DataFrame
# Additional print statements to understand the DataFrame:
# print(df.info()) # prints a summary of the DataFrame, simmilar to database schema
# print(df.describe()) # prints statistics of the DataFrame
# print(df.head()) # prints the first 5 rows of the DataFrame
# print(df.tail()) # prints the last 5 rows of the DataFrame
# print(df.columns) # prints the columns of the DataFrame
# print(df.index) # prints the index of the DataFrame

# Questions:
# What part of the data set needs to be cleaned?
# From PBL learning, what is a good time to clean data?  
# Could you hav Garbage in, Garbage out problem if you don't clean the data?