import pandas as pd
from sklearn.impute import SimpleImputer

def data_preprocessing(data_url):
    # Load and preprocess the dataset
    Data = pd.read_csv(data_url, header=None, na_values=["?"])
    Data.dropna(subset=Data.columns[Data.columns != 1], inplace=True)
    imputer = SimpleImputer()
    Data[1] = imputer.fit_transform(Data[[1]])
    
    return Data
