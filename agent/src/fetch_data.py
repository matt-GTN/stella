# src/fetch_data.py

import requests
import pandas as pd
import os

FMP_API_KEY = os.getenv("FMP_API_KEY")

# --- NOUVEAU : Définition d'une exception personnalisée ---
class APILimitError(Exception):
    """Exception levée lorsque la clé API est invalide, expirée ou a atteint sa limite."""
    pass

def fetch_fundamental_data(ticker: str) -> pd.DataFrame:
    """
    Récupère les données fondamentales d'une action.
    Lève une APILimitError si la clé API a un problème ou si la limite est atteinte.
    Lève une ValueError pour les autres erreurs d'API.
    """
    if not FMP_API_KEY:
        raise ValueError("La clé API FMP_API_KEY n'est pas configurée dans les variables d'environnement.")

    BASE_URL = "https://financialmodelingprep.com/api/v3/key-metrics/"
    url = f"{BASE_URL}{ticker}?period=annual&apikey={FMP_API_KEY}"

    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if not data: # Si la réponse est OK mais vide (ex: ticker invalide)
            raise ValueError(f"Aucune donnée retournée pour le ticker '{ticker}'. Il est peut-être invalide.")
        return pd.DataFrame(data)
    else:
        # --- MODIFICATION CLÉ : Gérer les erreurs spécifiques ---
        # 401: Unauthorized (clé invalide), 429: Too Many Requests (limite atteinte)
        if response.status_code in [401, 429]:
            try:
                # Essayer de récupérer le message d'erreur de l'API
                error_message = response.json().get('error', "La limite d'utilisation de la clé API a été atteinte ou la clé est invalide.")
            except:
                error_message = "La limite d'utilisation de la clé API a été atteinte ou la clé est invalide."
            raise APILimitError(error_message)
        else:
            # Pour toutes les autres erreurs HTTP
            raise ValueError(f"Erreur de l'API pour {ticker}: Status {response.status_code}, Réponse: {response.text}")

if __name__ == '__main__':
    # Example usage for testing
    try:
        aapl_data = fetch_fundamental_data("AAPL")
        print("\nAAPL Data Fetched Successfully!")
    except ValueError as e:
        print(f"Error fetching AAPL data: {e}")

    try:
        xyz_data = fetch_fundamental_data("XYZ")
        print("\nXYZ Data Fetched Successfully!")
    except ValueError as e:
        print(f"Error fetching XYZ data: {e}")