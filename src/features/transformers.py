import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder
)

def drop_null_rows(X):
    return X.dropna(subset=['ClaimNb', 'Exposure'])
drop_null_transformer = FunctionTransformer(drop_null_rows)

def drop_duplicates(X):
    return X.drop_duplicates(subset=['IDpol'])
drop_duplicates_transformer = FunctionTransformer(drop_duplicates)

def add_frequency_column(X):
    X = X.copy()
    X['Frequency'] = X['ClaimNb'] / X['Exposure']
    return X
add_frequency_column_transformer = FunctionTransformer(add_frequency_column)


class CapTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, cap_value, else_behaviour):
        """
        Custom transformer do ograniczania wartości w kolumnie.

        Parameters:
        - column: Nazwa kolumny, której wartości mają być ograniczone.
        - cap_value: Maksymalna wartość, którą można przypisać.
        """
        self.column = column
        self.cap_value = cap_value
        self.else_behaviour = else_behaviour

    def fit(self, X, y=None):
        try:
            self._impute_value = X[self.column].mean(axis=0)
        except:
            self._impute_value = X[self.column].mode()[0]
        return self

    def else_func(self):
        if self.else_behaviour == 'identity':
            return lambda x: x
        elif self.else_behaviour == 'round':
            return lambda x: round(x)
        elif self.else_behaviour == 'ceil':
            return lambda x: math.ceil(x)
        elif self.else_behaviour == 'int':
            return int
        elif self.else_behaviour == 'log':
            return lambda x: round(math.log(x), 2)
        else:
            raise ValueError(f"Unknown else_behaviour: {self.else_behaviour}")

    def transform(self, X):
        """
        Ogranicza wartości w wybranej kolumnie do maksymalnej wartości.

        Parameters:
        - X: DataFrame wejściowy.

        Returns:
        - DataFrame z ograniczonymi wartościami w wybranej kolumnie.
        """
        X = X.copy()  # Tworzenie kopii danych wejściowych
        X[self.column] = (
            X[self.column]
            .fillna(self._impute_value)
            .apply(lambda x: self.cap_value if x > self.cap_value else self.else_func()(x))
        )
        return X
    

def create_feature_pipeline(
    numerical_features,
    categorical_features,
    n_bins_drivage=20, 
    n_bins_vehage=8, 
    cap_values={
        'VehPower': 13,
        'VehAge': 21,
        'DrivAge': 90,
        'BonusMalus': 100,
        'Density': 100_000
    }
):
    """
    Tworzy pipeline do przetwarzania cech z konfigurowalnymi parametrami.
    
    Parametry:
    - n_bins_drivage: int - liczba przedziałów dla dyskretyzacji wieku kierowcy
    - n_bins_vehage: int - liczba przedziałów dla dyskretyzacji wieku pojazdu
    - cap_values: dict - słownik z wartościami granicznymi dla poszczególnych cech
    
    Zwraca:
    - Pipeline - skonfigurowany pipeline do przetwarzania cech
    """
    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer_features = ColumnTransformer(transformers=[
        ("binned_numeric_drivage", KBinsDiscretizer(n_bins=n_bins_drivage, random_state=0), ["DrivAge"]),
        ("binned_numeric_vehage", KBinsDiscretizer(n_bins=n_bins_vehage, random_state=0), ["VehAge"]),
        # ('target_encoding', TargetEncoder(random_state=0, cv=3), ['Region']),
        ('number', numeric_pipeline, numerical_features),
        ('category', categorical_pipeline, categorical_features),
    ],
    remainder='drop', 
    force_int_remainder_cols=False,
    verbose_feature_names_out=False
    )

    pipeline_features = Pipeline([
        ('cap_veh_power', CapTransformer(column='VehPower', cap_value=cap_values['VehPower'], else_behaviour='identity')),
        ('cap_veh_age', CapTransformer(column='VehAge', cap_value=cap_values['VehAge'], else_behaviour='identity')),
        ('cap_driv_age', CapTransformer(column='DrivAge', cap_value=cap_values['DrivAge'], else_behaviour='identity')),
        ('cap_bonus_malus', CapTransformer(column='BonusMalus', cap_value=cap_values['BonusMalus'], else_behaviour='int')),
        ('cap_density', CapTransformer(column='Density', cap_value=cap_values['Density'], else_behaviour='log')),
        ('transformer', transformer_features),
    ])
    
    return pipeline_features

def create_simple_feature_pipeline(
    numerical_features,
    categorical_features,
    cap_values={
        'VehPower': 13,
        'VehAge': 21,
        'DrivAge': 90,
        'BonusMalus': 100,
        'Density': 100_000
    }
):
    """
    Tworzy pipeline do przetwarzania cech z konfigurowalnymi parametrami.
    
    Parametry:
    - n_bins_drivage: int - liczba przedziałów dla dyskretyzacji wieku kierowcy
    - n_bins_vehage: int - liczba przedziałów dla dyskretyzacji wieku pojazdu
    - cap_values: dict - słownik z wartościami granicznymi dla poszczególnych cech
    
    Zwraca:
    - Pipeline - skonfigurowany pipeline do przetwarzania cech
    """
    numeric_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer_features = ColumnTransformer(transformers=[
        ('number', numeric_pipeline, numerical_features),
        ('category', categorical_pipeline, categorical_features),
    ],
    remainder='drop', 
    force_int_remainder_cols=False,
    verbose_feature_names_out=False
    )

    pipeline_features = Pipeline([
        ('cap_veh_power', CapTransformer(column='VehPower', cap_value=cap_values['VehPower'], else_behaviour='identity')),
        ('cap_veh_age', CapTransformer(column='VehAge', cap_value=cap_values['VehAge'], else_behaviour='identity')),
        ('cap_driv_age', CapTransformer(column='DrivAge', cap_value=cap_values['DrivAge'], else_behaviour='identity')),
        ('cap_bonus_malus', CapTransformer(column='BonusMalus', cap_value=cap_values['BonusMalus'], else_behaviour='int')),
        ('cap_density', CapTransformer(column='Density', cap_value=cap_values['Density'], else_behaviour='log')),
        ('transformer', transformer_features),
    ])
    
    return pipeline_features

if __name__ == "__main__":
    pass