# tools.py

import pandas as pd
import plotly.express as px
import plotly.io as pio
from langchain_core.tools import tool
from io import StringIO
from typing import List

# --- Import des logiques de src  ---
from src.search_ticker import search_ticker as _search_ticker_logic
from src.fetch_data import fetch_fundamental_data as _fetch_data_logic
from src.preprocess import preprocess_financial_data as _preprocess_data_logic
from src.analyze import analyse_risks as _analyze_risks_logic
from src.fetch_news import fetch_recent_news as _fetch_recent_news_logic
from src.fetch_profile import fetch_company_profile as _fetch_profile_logic
from src.fetch_price import fetch_price_history as _fetch_price_history_logic
from src.compare_fundamentals import compare_fundamental_metrics as _compare_fundamental_metrics_logic
from src.compare_prices import compare_price_histories as _compare_price_histories_logic
# PDF research is imported lazily to avoid initialization delays
# from src.pdf_research import query_research_document as _query_research_document_logic
from src.chart_theme import stella_theme


# --- Définition des outils ---
@tool
def search_ticker(company_name: str) -> str:
    """
    Utilise cet outil en PREMIER si l'utilisateur fournit un nom de société (comme 'Apple', 'Microsoft', 'Airbus') 
    au lieu d'un ticker (comme 'AAPL', 'MSFT', 'AIR.PA').
    Cet outil trouve le ticker boursier le plus probable pour un nom d'entreprise.
    
    Args:
        company_name (str): Le nom de l'entreprise à rechercher.
    """
    # La logique réelle est appelée depuis execute_tool_node, ceci est une coquille pour le LLM.
    return "[Le ticker est prêt à être recherché par le système.]"


@tool
def fetch_data(ticker: str) -> str:
    """Récupère les données financières fondamentales pour un ticker boursier donné."""
    return f"[Les données pour {ticker} sont prêtes à être récupérées par le système.]"

@tool
def preprocess_data() -> str:
    """Prépare les données financières récupérées pour la prédiction."""
    return "[L'étape de preprocessing est prête à être exécutée.]"

@tool
def analyze_risks() -> str:
    """Prédit la performance d'une action par rapport au marché en se basant sur les données prétraitées.
    Utilise un modèle de machine learning pour détecter les risques de sous-performance."""
    return "[L'étape de prédiction est prête à être exécutée.]"

@tool
def display_raw_data() -> str:
    """Affiche le tableau de données financières brutes qui ont été initialement récupérées."""
    return "[Le tableau de données brutes est prêt à être affiché.]"

@tool
def display_processed_data() -> str:
    """Affiche le tableau de données financières traitées et nettoyées, prêtes pour l'analyse."""
    return "[Le tableau de données traitées est prêt à être affiché.]"


def _create_dynamic_chart_logic(
    data: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str = None
) -> str:
    """Contient la logique de création de graphique, sans être un outil LangChain."""
    try:
        df = data.copy() # On travaille sur une copie
        if 'calendarYear' in df.columns:
            df['calendarYear'] = df['calendarYear'].astype(str)

        common_args = {
            'title': title,
            'color': color_column,
            'color_discrete_sequence': stella_theme['colors'] # Appliquer la palette
        }

        if chart_type == 'line':
            fig = px.line(df, x=x_column, y=y_column, markers=True, **common_args)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_column, y=y_column, **common_args)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_column, y=y_column, **common_args)
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_column, values=y_column, title=title, color_discrete_sequence=stella_theme['colors'])
        else:
            return f"Erreur : Le type de graphique '{chart_type}' n'est pas supporté."

        fig.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif"))
        return pio.to_json(fig)

    except Exception as e:
        # Il est utile de savoir quelle colonne a posé problème
        if isinstance(e, KeyError):
            return f"Erreur: La colonne '{e.args[0]}' est introuvable. Colonnes disponibles: {list(df.columns)}"
        return f"Erreur lors de la création du graphique : {str(e)}"

# L'outil LangChain qui est "vu" par le LLM
@tool
def create_dynamic_chart(
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str = None
) -> str:
    """
    Crée un graphique dynamique et interactif. Les données sont fournies automatiquement.
    Tu DOIS utiliser les noms de colonnes exacts qui te sont fournis dans le contexte actuel.

    Args:
        chart_type (str): Le type de graphique. Supportés : 'line', 'bar', 'scatter', 'pie'.
        x_column (str): Nom exact de la colonne pour l'axe X.
        y_column (str): Nom exact de la colonne pour l'axe Y.
        title (str): Un titre descriptif pour le graphique.
        color_column (str, optional): Nom exact de la colonne pour la couleur.
    """
    return "[L'outil de création de graphique est prêt à être exécuté.]"

@tool
def get_stock_news(ticker: str, company_name: str = None) -> str:
    """
    Utilise cet outil pour trouver les dernières actualités pour une entreprise.
    Tu peux l'utiliser si on te demande "les nouvelles", "les actualités", "que se passe-t-il avec...".
    
    Args:
        ticker (str): Le ticker de l'action (ex: 'AAPL'). Tu dois le trouver avec search_ticker si besoin.
        company_name (str, optional): Le nom complet de l'entreprise (ex: 'Apple Inc.'). Peut améliorer la pertinence de la recherche.
    """
    return "[Les actualités sont prêtes à être récupérées par le système.]"

@tool
def get_company_profile(ticker: str) -> str:
    """
    Utilise cet outil pour obtenir une description générale d'une entreprise.
    Fournit des informations comme le secteur, le CEO, une description de l'activité, le site web et beaucoup d'autres.
    C'est l'outil parfait si l'utilisateur demande "parle-moi de...", "que fait...", ou "qui est..." une entreprise.
    
    Args:
        ticker (str): Le ticker de l'action à rechercher (ex: 'AAPL').
    """
    return "[Le profil de l'entreprise est prêt à être récupéré par le système.]"

@tool
def display_price_chart(ticker: str, period_days: int = 252) -> str:
    """
    Affiche un graphique de l'évolution du prix (cours) d'une action sur une période.
    Utilise cet outil lorsque l'utilisateur demande "le prix", "le cours", "le graphique de l'action", "la performance de l'action", ou "l'évolution de l'action".
    Args:
        ticker (str): Le ticker de l'action (ex: 'AAPL'). L'agent doit le trouver si besoin.
        period_days (int): Le nombre de jours à afficher. 30 pour 1 mois, 90 pour 3 mois, 252 pour 1 an, 1260 pour 5 ans. La valeur par défaut est 252 (1 an).
    """
    return "[Le graphique de prix est prêt à être généré.]"

@tool
def compare_stocks(tickers: List[str], metric: str, comparison_type: str = 'fundamental', period_days: int = 252):
    """
    Compare plusieurs actions sur une métrique spécifique. Pour une métrique fondamentale,
    cela montre l'évolution sur plusieurs années. Pour le prix, cela montre la performance sur une période donnée.
    C'est l'outil principal pour toute demande contenant "compare", "vs", "versus", "par rapport à".

    Args:
        tickers (List[str]): La liste des tickers à comparer (ex: ['AAPL', 'MSFT', 'GOOGL']).
        metric (str): La métrique à comparer. 
                      - Pour les données fondamentales, utilise le nom exact (ex: 'roe', 'marketCap').
                      - Pour le prix, utilise TOUJOURS la valeur 'price'.
        comparison_type (str): Le type de comparaison. 'fundamental' ou 'price'. Le LLM doit déduire
                               le type en fonction de la métrique demandée ('price' vs autre chose).
        period_days (int): Pour une comparaison de prix, spécifie la période. 30 (1 mois), 90 (3 mois), 252 (1 an), etc. La valeur par défaut est 252.
    """
    return "[La comparaison est prête à être exécutée par le système.]"

@tool
def query_research(query: str) -> str:
    """
    Recherche dans le document de recherche interne de l'équipe pour obtenir des informations
    sur les méthodologies, analyses, conclusions de recherche, ta stack technique, etc.
    Utilise cet outil quand l'utilisateur demande des informations sur:
    - Les recherches de l'équipe
    - Les méthodologies utilisées
    - Les conclusions d'études
    - Des explications théoriques ou techniques
    
    Args:
        query (str): La question ou le sujet à rechercher dans le document de recherche
    """
    return "[La recherche dans le document est prête à être exécutée.]"

# --- La liste complète des outils disponibles pour l'agent ---
available_tools = [
    search_ticker,
    fetch_data,
    get_stock_news,
    get_company_profile,
    preprocess_data,
    analyze_risks,
    display_raw_data,
    display_processed_data,
    create_dynamic_chart,
    display_price_chart,
    compare_stocks,
    query_research
]