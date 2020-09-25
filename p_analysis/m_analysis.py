import pandas as pd

# analysis functions

def analyze(df):
    grouped = df.groupby('Make').agg({'Combined MPG':'mean'}).reset_index()
    results = grouped.sort_values('Combined MPG', ascending=False).head(10)
    return results