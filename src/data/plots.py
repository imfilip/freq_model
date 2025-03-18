from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from src.data.utils import ceildiv

# warnings.filterwarnings("ignore", module="matplotlib\\..*", category=UserWarning)


def plot_hists(df, columns: List[str] = ['Exposure', 'ClaimNb'], plot_cols: int = 2):
    # Close any existing plots and set style
    plt.close("all")
    plt.style.use("dark_background")

    hists_to_plot = len(columns)
    nrows = ceildiv(hists_to_plot, plot_cols)
    figsize = (12 * plot_cols, 8 * nrows)

    plt.subplots(nrows=nrows, ncols=plot_cols, figsize=figsize)
    
    for idx, col in enumerate(columns, 1):
        plt.subplot(ceildiv(hists_to_plot, plot_cols), plot_cols, idx)
        sns.histplot(df, x=col)
        plt.xlabel(col, fontsize=20)
        plt.ylabel('Count', fontsize=20)

    return None


def freq_over_attribute_plot(
    df: pd.DataFrame,
    attribute: str,
    exp: str = "Exposure",
    claims: str = "ClaimNb"
) -> None:
    """
    Tworzy wykres częstotliwości szkód w zależności od określonego atrybutu, 
    porównując częstotliwości marginalne z częstotliwościami portfelowymi.

    Parametry:
        df (pd.DataFrame): DataFrame zawierający dane ubezpieczeniowe.
        attribute (str): Nazwa kolumny atrybutu, który ma być analizowany.
        exposure_col (str): Nazwa kolumny reprezentującej ekspozycję. Domyślnie 'Exposure'.
        claims_col (str): Nazwa kolumny reprezentującej liczbę szkód. Domyślnie 'ClaimNb'.

    Zwraca:
        None: Wyświetla wykres.
    """
    # Zamknięcie istniejących wykresów i ustawienie stylu
    plt.close("all")
    plt.style.use("dark_background")

    # Agregacja danych i obliczenie częstotliwości portfelowej
    data_to_plot = (
        df.groupby(by=[attribute])
        .agg({claims: "sum", exp: "sum"})
        .reset_index()
        .assign(MarginalFreq=lambda x: x[claims] / x[exp])
        .assign(PortfolioFreq=lambda _: df[claims].sum() / df[exp].sum())
        .assign(Rank=lambda x: x[attribute].rank(method="dense") - 1)
    )

    sorted_attributes = sorted(data_to_plot[attribute].unique())

    # Inicjalizacja wykresu i osi
    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor("#1f1f1f")
    ax1.set_facecolor("#2a2a2a")

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # Wykres kolumnowy dla ekspozycji
    sns.barplot(
        x=attribute,
        y=exp,
        data=data_to_plot,
        estimator=sum,
        order=sorted_attributes,
        alpha=0.8,
        ax=ax1,
        color=colors[0]
    )
    
    # Dodanie etykiet do słupków
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.0f", fontsize=11, fontweight="bold", color="white")

    ax1.set_xticks(range(len(sorted_attributes)))
    ax1.set_xticklabels(sorted_attributes, rotation=45, ha="right", fontsize=12)
    ax1.set_ylabel("Exposure", fontsize=14, fontweight="bold", color="white")
    ax1.tick_params(axis="both", colors="white", labelsize=12)

    # Wykres liniowy dla częstotliwości
    ax2 = ax1.twinx()

    # Linia dla częstotliwości w ramach atrybutów
    line_marginal = sns.lineplot(
        x="Rank",
        y="MarginalFreq",
        data=data_to_plot,
        label=f"{attribute} Frequency",
        marker="o",
        markersize=10,
        linewidth=3,
        color=colors[1],
        ax=ax2
    )

    # Linia dla częstotliwości portfelowej
    line_portfolio = sns.lineplot(
        x="Rank",
        y="PortfolioFreq",
        data=data_to_plot,
        label="Portfolio Frequency",
        marker="o",
        markersize=5,
        linewidth=3,
        color="#ffff99",
        ax=ax2
    )

    # Formatowanie osi Y jako procenty
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("Frequency", fontsize=14, fontweight="bold", color="white")
    ax2.tick_params(axis="y", colors="white", labelsize=12)

    # Dodanie legendy
    leg = ax2.legend(loc="upper left", fontsize=12, frameon=True, framealpha=0.8)
    leg.get_frame().set_facecolor("#3a3a3a")

    # Tytuł
    fig.suptitle(f'Frequency over {attribute}', fontsize=20, fontweight='bold', y=0.98, color='white')
    ax1.set_title('Comparison to portfolio frequency', fontsize=16, pad=20, color='white')
    
    # Dodatkowe ustawienia
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

    return fig, ax1, ax2

def corr_plot(df: pd.DataFrame, method: str = 'pearson') -> None:
    """
    Tworzy profesjonalny wykres korelacji (heatmapę) dla danych w DataFrame.

    Parametry:
        df (pd.DataFrame): DataFrame zawierający dane numeryczne, dla których mają być obliczone korelacje.
        method (str): Metoda obliczania korelacji. Dostępne opcje to:
                      - 'pearson' (domyślnie): Korelacja liniowa.
                      - 'kendall': Korelacja rangowa Kendalla.
                      - 'spearman': Korelacja rangowa Spearmana.

    Zwraca:
        None: Wyświetla wykres korelacji.
    
    Przykład użycia:
        corr_plot(df=my_dataframe, method='spearman')
    """
    # Zamknięcie istniejących wykresów i ustawienie stylu
    plt.close("all")
    plt.style.use("dark_background")

    # Obliczenie macierzy korelacji z wybraną metodą
    correlation_matrix = df.corr(method=method)

    # Tworzenie maski dla górnego trójkąta macierzy korelacji
    mask = np.triu(np.ones(correlation_matrix.shape), k=0).astype(bool)

    # Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(15, 12))
    fig.patch.set_facecolor("#1f1f1f")  # Kolor tła wykresu

    # Heatmapa korelacji
    sns.heatmap(
        correlation_matrix,
        mask=mask,  # Maskowanie górnego trójkąta
        annot=True,  # Wyświetlanie wartości korelacji
        fmt=".2f",  # Format wartości (do dwóch miejsc po przecinku)
        cmap="viridis",  # Mapa kolorów
        linewidths=0.5,  # Linie oddzielające komórki
        linecolor="gray",  # Kolor linii oddzielających komórki
        cbar_kws={"shrink": 0.8},  # Formatowanie paska kolorów (cbar)
        ax=ax  # Użycie określonej osi
    )

    # Ustawienia osi i tytułu
    ax.set_title(f"Macierz korelacji ({method.capitalize()} method)", fontsize=16, fontweight="bold", color="white", pad=20)
    ax.tick_params(axis="both", colors="white", labelsize=12)

    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()

def scatter_plot(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    hue: str = None, 
    size: str = None, 
    title: str = None, 
    trendline: bool = False,
    ci: int = 95
) -> None:
    """
    Tworzy profesjonalny wykres punktowy (scatterplot) dla dwóch zmiennych z opcjonalną linią trendu.

    Parametry:
        df (pd.DataFrame): DataFrame zawierający dane do wizualizacji.
        x (str): Nazwa kolumny dla osi X.
        y (str): Nazwa kolumny dla osi Y.
        hue (str, opcjonalne): Nazwa kolumny określającej kolor punktów. Domyślnie None.
        size (str, opcjonalne): Nazwa kolumny określającej rozmiar punktów. Domyślnie None.
        title (str, opcjonalne): Tytuł wykresu. Jeśli nie podano, zostanie wygenerowany automatycznie.
        trendline (bool, opcjonalne): Czy dodać linię trendu do wykresu. Domyślnie False.
        ci (int, opcjonalne): Poziom ufności dla przedziału wokół linii trendu (dotyczy Seaborn). Domyślnie 95.

    Zwraca:
        None: Wyświetla wykres punktowy.

    Przykład użycia:
        scatter_plot(df=my_dataframe, x='Variable1', y='Variable2', hue='Category', size='Weight', trendline=True)
    """
    # Zamknięcie istniejących wykresów i ustawienie stylu
    plt.close("all")
    plt.style.use("dark_background")

    # Inicjalizacja wykresu
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#1f1f1f")  # Kolor tła wykresu

    if trendline:
        # Wykres punktowy z linią trendu za pomocą Seaborn regplot
        sns.regplot(
            data=df,
            x=x,
            y=y,
            scatter_kws={"alpha": 0.3, "s": 100},  # Styl punktów
            line_kws={"color": "red", "linewidth": 1},  # Styl linii trendu
            ci=ci,
            ax=ax
        )
        if hue or size:
            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                size=size,
                palette="viridis",
                sizes=(20, 200),
                alpha=0.05,
                ax=ax
            )
    else:
        # Wykres punktowy bez linii trendu za pomocą Seaborn scatterplot
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            size=size,
            palette="viridis",
            sizes=(20, 200),
            alpha=0.8,
            ax=ax
        )

    # Ustawienia osi i tytułu
    ax.set_xlabel(x, fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel(y, fontsize=14, fontweight="bold", color="white")
    ax.tick_params(axis="both", colors="white", labelsize=12)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=20)
    else:
        ax.set_title(f"Scatterplot: {x} vs {y}", fontsize=16, fontweight="bold", color="white", pad=20)

    # Legenda (jeśli istnieje)
    if hue or size:
        leg = ax.legend(loc="best", fontsize=12, frameon=True)
        leg.get_frame().set_facecolor("#3a3a3a")
        leg.get_frame().set_edgecolor("gray")
        for text in leg.get_texts():
            text.set_color("white")

    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()

def plot_heatmap(
    data: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    lower_bound: float = 0.05,
    upper_bound: float = 0.95,
    colour: Optional[str] = None
) -> Union[None, Tuple[float, float]]:
    """
    Generuje mapę cieplną (heatmap) na podstawie dostarczonych danych.

    Parametry:
        data (pd.DataFrame): Dane wejściowe w formacie tabelarycznym.
        x (str): Nazwa kolumny używanej jako oś X na mapie cieplnej.
        y (str): Nazwa kolumny używanej jako oś Y na mapie cieplnej.
        value (str): Nazwa kolumny reprezentującej wartości do agregacji.
                     Dozwolone wartości to 'Exposure', 'ClaimNb' lub 'Freq'.
        lower_bound (float): Dolny próg kwantylowy dla filtrowania danych.
                             Domyślnie 0.05.
        upper_bound (float): Górny próg kwantylowy dla filtrowania danych.
                             Domyślnie 0.95.
        colour (Optional[str]): Kolorystyka mapy cieplnej. Domyślnie None.

    Zwraca:
        Union[None, Tuple[float, float]]:
            - None, jeśli wartość `value` to 'Exposure' lub 'ClaimNb'.
            - Krotka zawierająca dolny i górny próg kwantylowy, jeśli wartość `value` to 'Freq'.

    Wyjątki:
        ValueError: Jeśli wartość parametru `value` jest niepoprawna.
    """
    def filter_data(data: pd.DataFrame, lower_bound: float, upper_bound: float) -> Tuple[pd.DataFrame, float, float]:
        """
        Filtruje dane na podstawie określonych progów kwantylowych.

        Parametry:
            data (pd.DataFrame): Dane do filtrowania.
            lower_bound (float): Dolny próg kwantylowy.
            upper_bound (float): Górny próg kwantylowy.

        Zwraca:
            Tuple[pd.DataFrame, float, float]: Przefiltrowane dane oraz wartości dolnego i górnego progu kwantylowego.
        """
        lower_quantile = np.nanquantile(data.values.flatten(), lower_bound)
        upper_quantile = np.nanquantile(data.values.flatten(), upper_bound)
        filtered_data = data.map(lambda x: np.nan if x < lower_quantile or x > upper_quantile else x)
        return filtered_data, lower_quantile, upper_quantile

    if value in ('Exposure', 'ClaimNb'):
        # Tworzenie tabeli przestawnej
        aggregated_data = data.pivot_table(values=value, index=x, columns=y, aggfunc='sum')
        
        # Filtrowanie danych na podstawie kwantyli
        filtered_data, lower_quantile, upper_quantile = filter_data(aggregated_data, lower_bound, upper_bound)
        
        # Rysowanie heatmapy
        plt.figure(figsize=(15, 12))
        sns.heatmap(filtered_data.sort_index(ascending=False), cmap=colour)

    elif value == 'Freq':
        # Tworzenie tabel przestawnych dla ClaimNb i Exposure
        claim_data = data.pivot_table(values='ClaimNb', index=x, columns=y, aggfunc='sum')
        exposure_data = data.pivot_table(values='Exposure', index=x, columns=y, aggfunc='sum')
        
        # Obliczanie częstości (frequency)
        freq_data = claim_data / exposure_data
        
        # Filtrowanie danych częstości na podstawie kwantyli
        filtered_freq_data, lower_quantile, upper_quantile = filter_data(freq_data, lower_bound, upper_bound)
        
        # Rysowanie heatmapy
        plt.figure(figsize=(15, 12))
        sns.heatmap(filtered_freq_data.sort_index(ascending=False), cmap=colour)

    else:
        raise ValueError(f"Niepoprawna wartość '{value}'. Oczekiwano jednej z ['Exposure', 'ClaimNb', 'Freq'].")

    return lower_quantile, upper_quantile

if __name__ == "__main__":
    pass
