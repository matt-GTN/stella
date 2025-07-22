# agent/src/compare_prices.py

import pandas as pd
from .fetch_price import fetch_price_history

def compare_price_histories(tickers: list[str], period_days: int = 252) -> pd.DataFrame:
    """
    Récupère et normalise les historiques de prix pour plusieurs tickers afin de les comparer.
    La normalisation est essentielle pour comparer sur une base de 100.
    """
    all_normalized_prices = []
    
    for ticker in tickers:
        try:
            print(f"Comparaison de prix: Récupération pour {ticker}...")
            price_df = fetch_price_history(ticker, period_days)
            
            # Normalisation : (prix actuel / premier prix) * 100
            normalized_price = (price_df['close'] / price_df['close'].iloc[0]) * 100
            normalized_price.name = ticker.upper() # On renomme la série avec le nom du ticker
            all_normalized_prices.append(normalized_price)
            
        except Exception as e:
            print(f"Erreur lors de la récupération des prix pour {ticker}: {e}")
            continue
    
    if not all_normalized_prices:
        raise ValueError("Impossible de récupérer les données de prix pour la comparaison.")
    
    # Concatène toutes les séries normalisées en un seul DataFrame
    combined_df = pd.concat(all_normalized_prices, axis=1)
    # Remplit les valeurs manquantes (si les jours de bourse diffèrent)
    combined_df = combined_df.fillna(method='ffill')
    
    return combined_df