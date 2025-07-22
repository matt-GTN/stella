# src/preprocess.py

import pandas as pd

def preprocess_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Préprocesse les données financières brutes.
    Le DataFrame retourné contient les features pour le modèle ET des colonnes supplémentaires
    comme 'calendarYear' pour la visualisation.
    """
    df_processed = df.copy()
    
    # On garde une trace de l'année pour la visualisation
    df_processed['calendarYear'] = df_processed['calendarYear'].astype(str)

    # On crée l'index pour la cohérence
    df_processed = df_processed.set_index(df_processed['symbol'] + '_' + df_processed['calendarYear'])

    # Calculs
    df_processed['marginProfit'] = df_processed['netIncomePerShare'] / df_processed['revenuePerShare']
    df_processed = df_processed.sort_values(by='calendarYear')
    df_processed['revenuePerShare_YoY_Growth'] = ((df_processed['revenuePerShare'] / df_processed['revenuePerShare'].shift(1)) - 1) * 100
    
    # Sélection des colonnes finales "riches"
    final_cols = [
        'calendarYear', 'marketCap', 'marginProfit', 'roe', 'roic', 'revenuePerShare', 
        'debtToEquity', 'revenuePerShare_YoY_Growth', 'earningsYield'
    ]
    
    # On s'assure de ne pas sélectionner des colonnes qui n'existeraient pas dans les données brutes
    available_cols = [col for col in final_cols if col in df_processed.columns]
    
    df_processed = df_processed[available_cols].dropna()

    print(f"Données preprocess :\n{df_processed.head()}")
    return df_processed