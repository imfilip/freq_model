from typing import Union

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def calculate_frequency(df: pd.DataFrame, y_col: str = 'ClaimNb', exposure_col: str = 'Exposure') -> Union[float, int]:
    """
    Oblicza częstotliwość roszczeń na podstawie sumy roszczeń i ekspozycji.

    Parametry:
    - df: pd.DataFrame - DataFrame zawierający kolumny 'ClaimNb' i 'Exposure'.

    Zwraca:
    - float lub int - Częstotliwość roszczeń.
    """
    return df[y_col].sum() / df[exposure_col].sum()

def split_data_by_group_and_stratify(df, group_col='GroupID', stratify_col='ClaimNb', test_size=0.2, random_state=42):
    """
    Splits the DataFrame into train and test sets, ensuring that all records with the same group_col value
    are in the same set, and that the stratify_col has the same distribution in both sets.

    Parameters:
    - df: pd.DataFrame - The DataFrame to split.
    - group_col: str - The column name to group by.
    - stratify_col: str - The column name to stratify by.
    - test_size: float - The proportion of the dataset to include in the test split.
    - random_state: int - Random seed for reproducibility.

    Returns:
    - train_df: pd.DataFrame - The training set.
    - test_df: pd.DataFrame - The test set.
    """
    # Group by the specified column and calculate the mean of the stratify column for stratification
    grouped = df.groupby(group_col).agg({stratify_col: 'mean'}).reset_index()

    # Use GroupShuffleSplit to split the groups
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(grouped, groups=grouped[group_col]))

    # Get the GroupIDs for train and test
    train_groups = grouped.iloc[train_idx][group_col]
    test_groups = grouped.iloc[test_idx][group_col]

    # Split the original DataFrame based on these GroupIDs
    train_df = df[df[group_col].isin(train_groups)]
    test_df = df[df[group_col].isin(test_groups)]

    print(f"Train frequency: {calculate_frequency(train_df) * 100:.3f}%")
    print(f"Test frequency: {calculate_frequency(test_df) * 100:.3f}%")
    absolute_difference = abs(calculate_frequency(train_df) - calculate_frequency(test_df))
    relative_difference = absolute_difference / calculate_frequency(train_df)
    print(f"Train-test frequency difference: {absolute_difference * 100:.3f}% (Relative difference: {relative_difference * 100:.3f}%)")

    return train_df, test_df

