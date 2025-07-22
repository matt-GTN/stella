# agent/src/compare_fundamentals.py

import pandas as pd

# On importe les logiques existantes pour les réutiliser
from .fetch_data import fetch_fundamental_data
from .preprocess import preprocess_financial_data

def compare_fundamental_metrics(tickers: list[str], metric: str) -> pd.DataFrame:
    """
    Récupère l'historique d'une métrique fondamentale pour plusieurs tickers
    et les combine dans un seul DataFrame pour une comparaison temporelle.
    
    Returns:
        pd.DataFrame: Un DataFrame où l'index est 'calendarYear' et chaque colonne
                      est un ticker, contenant les valeurs de la métrique.
    """
    all_metrics_series = []
    
    for ticker in tickers:
        try:
            print(f"Comparaison (Évolution): Récupération des données pour {ticker}...")
            raw_df = fetch_fundamental_data(ticker)
            processed_df = preprocess_financial_data(raw_df)
            
            # On vérifie que les colonnes nécessaires sont présentes
            if metric not in processed_df.columns or 'calendarYear' not in processed_df.columns:
                print(f"Avertissement: Données insuffisantes pour '{metric}' chez {ticker}.")
                continue
            
            # On sélectionne l'évolution de la métrique pour ce ticker
            metric_series = processed_df.set_index('calendarYear')[metric]
            metric_series.name = ticker.upper() # Le nom de la série devient le ticker
            
            all_metrics_series.append(metric_series)

        except Exception as e:
            print(f"Erreur lors du traitement de {ticker} pour la comparaison d'évolution: {e}")
            continue
            
    if not all_metrics_series:
        raise ValueError(f"Impossible de récupérer l'historique de la métrique '{metric}' pour les tickers fournis.")
        
    # On combine toutes les séries en un seul DataFrame
    # L'index (calendarYear) permet d'aligner les données automatiquement
    combined_df = pd.concat(all_metrics_series, axis=1)
    
    # On peut trier par l'index (années) pour s'assurer de l'ordre
    return combined_df.sort_index()