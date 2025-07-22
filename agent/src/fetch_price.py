# agent/src/fetch_price.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_price_history(ticker: str, period_days: int = 252) -> pd.DataFrame:
    """
    Récupère l'historique des prix de clôture pour un ticker sur une période donnée
    en utilisant la librairie yfinance pour une couverture internationale.
    
    Args:
        ticker (str): Le ticker de l'action (ex: 'AAPL', '005930.KS', 'AIR.PA').
        period_days (int): Le nombre de jours dans le passé à récupérer.
        
    Returns:
        pd.DataFrame: Un DataFrame avec 'date' en index et 'close' en colonne simple.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        price_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if price_df.empty:
            raise ValueError(f"Aucun historique de prix trouvé pour le ticker '{ticker}'. Il est peut-être invalide ou non listé sur Yahoo Finance.")
            
        df_close = price_df[['Close']].copy()
        
        # Si les colonnes sont un MultiIndex (ex: [('Close', 'TICKER')]), on l'aplatit.
        if isinstance(df_close.columns, pd.MultiIndex):
            df_close.columns = df_close.columns.droplevel(1)

        # Rename the column to 'close' to match the rest of the agent's expectations
        df_close.rename(columns={'Close': 'close'}, inplace=True)
        
        print(f"yfinance: Historique de prix récupéré avec succès pour {ticker}.")
        return df_close

    except Exception as e:
        raise ValueError(f"Impossible de traiter les données de prix de yfinance pour {ticker}: {e}")

if __name__ == '__main__':
    try:
        samsung_prices = fetch_price_history("005930.KS", period_days=90)
        print("\nHistorique des prix pour Samsung (005930.KS):")
        print(samsung_prices.head())
        # Test the column name
        print(f"Column name for Samsung: {samsung_prices.columns[0]}")

        airbus_prices = fetch_price_history("AIR.PA", period_days=90)
        print("\nHistorique des prix pour Airbus (AIR.PA):")
        print(airbus_prices.head())
        print(f"Column name for Airbus: {airbus_prices.columns[0]}")
        
    except Exception as e:
        print(f"Erreur: {e}")