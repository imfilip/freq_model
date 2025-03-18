from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import math

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