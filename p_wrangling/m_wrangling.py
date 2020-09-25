import pandas as pd

# wrangling functions

def wrangle(df,year):
    filtered = df[df['Year']==year]
    return filtered