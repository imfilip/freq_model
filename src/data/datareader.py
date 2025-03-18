import csv
from collections import Counter
from functools import partial
from typing import Generator, List, Optional

import pandas as pd

class CleanedDataReader:
    """
    Klasa odpowiedzialna za odczytanie danych z pliku CSV, 
    identyfikację wadliwych linii i ich ewentualne naprawienie.
    """

    def __init__(self, path: str, separator: str = ","):
        """
        Inicjalizacja obiektu.

        :param path: Ścieżka do pliku CSV.
        :param separator: Separator użyty w pliku CSV (domyślnie przecinek).
        """

        self.path = path
        self.separator = separator
        self.invalid_lines, self.correct_line_length, self.original_data_size = self._get_invalid_lines(self._generate_csv_lines())
        self.show_invalid_lines = self.show_invalid_lines()

    def _generate_csv_lines(self) -> Generator[List[str], None, None]:
        """
        Generator, który zwraca linie z pliku CSV.

        :return: Generator linii z pliku CSV.
        """

        with open(self.path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                yield row

    def _get_invalid_lines(self, lines_generator: Generator[List[str], None, None]) -> tuple[List[int], int, int]:
        """
        Metoda identyfikująca wadliwe linie na podstawie ich długości.

        :param lines_generator: Generator linii z pliku CSV.
        :return: Lista indeksów wadliwych linii, długość poprawnych linii i maksymalna długość linii.
        """

        lines_length = {idx: len(line) for idx, line in enumerate(lines_generator)}
        c = Counter(lines_length.values()).most_common()[0][0]
        invalid_lines = [idx for idx, length in lines_length.items() if length != c]
        return invalid_lines, c, max(lines_length)

    def show_invalid_lines(self) -> List[str]:
        """
        Metoda zwracająca wadliwe linie w postaci ciągów znaków.

        :return: Lista wadliwych linii jako ciągi znaków.
        """

        return [self.separator.join(line) 
            for idx, line in enumerate(self._generate_csv_lines()) 
                if idx in self.invalid_lines]

    def handle_bad_line(self, bad_line: List[str], sep: str) -> Optional[List[str]]:
        """
        Metoda obsługująca wadliwe linie. Jeśli linia jest za długa, 
        usuwa puste kolumny i zwraca naprawioną linię.

        :param bad_line: Wadliwa linia.
        :param sep: Separator użyty w pliku CSV.
        :return: Naprawiona linia lub None, jeśli nie ma potrzeby jej naprawiania.
        """

        if len(bad_line) > self.correct_line_length:
            enhanced_line = [col for col in bad_line if col]
            print(f"Bad line:\n{bad_line}\nwas converted to:\n{enhanced_line}\n")
            return enhanced_line 
        return None
    
    def get_enhanced_data(self, **kwargs) -> pd.DataFrame:
        """
        Metoda zwracająca ramkę danych Pandas z naprawionymi wadliwymi liniami.

        :param kwargs: Dodatkowe parametry przekazywane do pd.read_csv.
        :return: Ramka danych Pandas.
        """

        df = pd.read_csv(
            self.path,
            on_bad_lines=partial(self.handle_bad_line, sep=self.separator), 
            engine='python',
            **kwargs
        )
        return df 


if __name__ == "__main__":
    DR = CleanedDataReader('/content/drive/MyDrive/EH/claims_data.csv')
    print(DR.invalid_lines)
    print(DR.correct_line_length)
    print(DR.original_data_size)
    print(DR.show_invalid_lines)
    df = DR.get_enhanced_data(index_col=0)
    print(df.head)
    print(df.shape)