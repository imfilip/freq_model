from copy import deepcopy
from typing import Union, List
import pandas as pd


def count_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.isnull().sum().reset_index()

def get_null_records(df):
    """
    Zwraca rekordy z DataFrame, które mają co najmniej jedną wartość null.
    
    Parametry:
        df (pd.DataFrame): DataFrame do przeanalizowania
        
    Zwraca:
        pd.DataFrame: DataFrame zawierający tylko rekordy z wartościami null
    """
    return df[df.isnull().any(axis=1)]

def drop_duplicates(df, subset):
    return df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)


def dropna(df, subset):
    return df.dropna(subset=subset).reset_index(drop=True)


def count_duplicates(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Liczy duplikaty w zadanych kolumnach DataFrame.
    
    Parametry:
        df (pd.DataFrame): DataFrame do przeanalizowania
        columns (str lub List[str]): Nazwa kolumny lub lista kolumn, w których chcemy policzyć duplikaty
        
    Zwraca:
        pd.DataFrame: DataFrame zawierający wartości i liczbę ich wystąpień,
                     posortowany malejąco według liczby wystąpień
    """
    if isinstance(columns, str):
        columns = [columns]
        
    duplicates = (df[columns]
                 .value_counts()
                 .to_frame('occurrences')
                 .reset_index()
                 .sort_values('occurrences', ascending=False))
    
    return duplicates


def ceildiv(a, b):
    return -(a // -b)


def create_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy unikalne identyfikatory grup dla rekordów w DataFrame.

    Parametry:
        df (pd.DataFrame): DataFrame zawierający dane ubezpieczeniowe.

    Zwraca:
        pd.DataFrame: DataFrame z dodaną kolumną 'GroupID', która zawiera unikalne identyfikatory grup.
    """
    data = deepcopy(df)
    df_with_group_id = (
        data
        .iloc[data.drop(['IDpol','Exposure','ClaimNb'], axis=1).drop_duplicates().index]
        .reset_index(drop=True)
        .assign(GroupID=lambda x: x.index+1)
        .merge(data, how='right')
        .assign(GroupID=lambda x: x['GroupID'].ffill())
    )
    return df_with_group_id

def prepare_target_and_weights(train, test):
    """
    Prepares target variables and weights for frequency modeling.
    
    Parameters:
    -----------
    train : DataFrame
        Training dataset containing ClaimNb and Exposure columns
    test : DataFrame
        Test dataset containing ClaimNb and Exposure columns
        
    Returns:
    --------
    tuple
        (y_train, weights_train, y_freq_train, y_test, weights_test, y_freq_test)
    """
    y_train = train['ClaimNb']
    weights_train = train['Exposure']
    y_freq_train = y_train / weights_train

    y_test = test['ClaimNb'] 
    weights_test = test['Exposure']
    y_freq_test = y_test / weights_test
    
    return y_train, weights_train, y_freq_train, y_test, weights_test, y_freq_test


if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4, 5],
            "Exposure": [1, 2, 3, 4, 5],
            "ClaimNb": [1, 2, 3, 4, 5],
            "Area": ["A", "A", "B", "B", "C"],
            "VehPower": [1, 1, 3, 4, 5],
            "VehAge": [1, 1, 3, 4, 5],
            "DrivAge": [1, 1, 3, 4, 5],
            "BonusMalus": [1, 1, 3, 4, 5],
            "VehBrand": ["A", "A", "B", "B", "C"],
            "VehGas": ["A", "A", "B", "B", "C"],
        }
    )
    df_with_group_id = create_group_id(df)
    print(df_with_group_id)
