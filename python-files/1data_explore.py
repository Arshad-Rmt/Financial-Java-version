import pandas as pd
data = pd.read_csv("data.csv\data_2.csv")  
print(data.head())
# print("\nColumns:\n", data.columns.tolist())
print("DATA CLEANING ")
print("\nMissing Values:\n", data.isnull().sum())

