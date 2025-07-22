# src/fetch_news.py

import requests
import os
import json
from datetime import datetime, timedelta
# On peut garder notre exception personnalisée pour la cohérence
from .fetch_data import APILimitError 

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_recent_news(ticker: str, company_name: str, limit: int = 3) -> str:
    """
    Récupère les dernières actualités pour une entreprise en utilisant NewsAPI.
    Retourne une chaîne de caractères JSON contenant une liste d'articles.
    """
    if not NEWS_API_KEY:
        raise ValueError("La clé API NEWS_API_KEY n'est pas configurée.")

    # NewsAPI préfère les noms d'entreprise aux tickers pour la recherche générale
    # On nettoie le nom pour de meilleurs résultats (ex: "McDonald's Corporation" -> "McDonald's")
    search_query = company_name.split(' ')[0].replace(',', '')

    BASE_URL = "https://newsapi.org/v2/everything"
    
    # On cherche les nouvelles des 30 derniers jours
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    params = {
        'q': search_query,          # Le terme de recherche (le nom de l'entreprise)
        'language': 'fr',           # On peut chercher en français !
        'from': one_month_ago,
        'sortBy': 'relevancy',      # On trie par pertinence
        'apiKey': NEWS_API_KEY,
        'pageSize': limit           # Le nombre d'articles à retourner
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return json.dumps([]) # Retourne une liste vide si rien n'est trouvé

        # --- On adapte le formatage à la structure de NewsAPI ---
        articles_to_return = []
        for article in articles:
            articles_to_return.append({
                "title": article.get('title'),
                "site": article.get('source', {}).get('name'), # La source est dans un sous-dictionnaire
                "url": article.get('url'),
                "image": article.get('urlToImage') # Le champ s'appelle urlToImage
            })
        
        return json.dumps(articles_to_return)

    except requests.exceptions.HTTPError as http_err:
        # NewsAPI renvoie des messages d'erreur clairs en cas de problème
        error_details = http_err.response.json()
        raise APILimitError(f"Erreur de l'API d'actualités : {error_details.get('message')}")
    except requests.exceptions.RequestException as req_err:
        raise APILimitError(f"Impossible de contacter le service d'actualités. Erreur: {req_err}")