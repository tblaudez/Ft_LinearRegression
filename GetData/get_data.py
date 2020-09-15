import pandas as pd
from pathlib import Path


def get_data() -> tuple:
    filename = Path(__file__).parent / 'data.csv'
    data = pd.read_csv(filename)
    return data['km'].tolist(), data['price'].tolist()
