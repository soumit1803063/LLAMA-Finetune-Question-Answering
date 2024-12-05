from .utils import load_data, split_data

def data_preparation_pipeline(data_file_path:str,test_size:float,random_state:int):
    dataframe = load_data(file_path=data_file_path)
    train_dataframe, test_dataframe = split_data(data=dataframe,test_size=test_size,random_state=random_state)
    return train_dataframe,test_dataframe