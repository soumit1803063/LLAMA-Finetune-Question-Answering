import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    # Load data from a CSV file
    data = pd.read_csv(file_path)
    if 'syntheses' in data.columns:
        data.drop("syntheses", axis=1, inplace=True)
    return data

def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data
