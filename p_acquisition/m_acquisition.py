import pandas as pd

# acquisition functions

def acquire():
    data = pd.read_csv('./data/raw/vehicles.csv')
    return data