# agent.py
from dotenv import load_dotenv
load_dotenv()

# Variables d'environnement
import os

# Variables et données
import json
from typing import TypedDict, List, Annotated, Any
import pandas as pd
from io import StringIO
import textwrap

# Graphiques
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import graphviz

# Numéro de session unique
import uuid

# Import de scripts
from src.fetch_data import APILimitError 
from src.chart_theme import stella_theme 

# LangGraph et LangChain
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client


# --- Import des tools ---
from tools import (
    available_tools,
    _fetch_recent_news_logic,
    _search_ticker_logic,
    _fetch_data_logic, 
    _preprocess_data_logic, 
    _analyze_risks_logic, 
    _create_dynamic_chart_logic,
    _fetch_profile_logic,
    _fetch_price_history_logic,
    _compare_fundamental_metrics_logic,
    _compare_price_histories_logic,
    _query_research_document_logic
)

# Environment variables and constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "moonshotai/kimi-k2-instruct"
LANGSMITH_TRACING = True
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "stella"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY n'a pas été enregistrée comme variable d'environnement.")

# Initialize the LLM
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0
)

# Objet AgentState pour stocker et modifier l'état de l'agent entre les nœuds
class AgentState(TypedDict):
    input: str
    ticker: str
    tickers: List[str]
    company_name: str
    fetched_df_json: str
    processed_df_json: str
    analysis: str
    plotly_json: str  
    messages: Annotated[List[AnyMessage], add_messages]
    error: str

# --- Prompt système (définition du rôle de l'agent) ---
system_prompt = """Ton nom est Stella. Tu es une assistante experte financière. Ton but principal est d'aider les utilisateurs en analysant des actions. Tu as été créée par une équipe de recherche dans le cadre du **Projet OPA**.

**Structure des réponses**
Tu répondras toujours de manière structurée et claire, en utilisant des balises strong, puces, etc en markdown pour organiser l'information.

**Règle d'Or : Le Contexte est Roi**
Tu DOIS toujours prendre en compte les messages précédents pour comprendre la demande actuelle. 
Si un utilisateur demande de modifier ou d'ajouter quelque chose, tu dois te baser sur l'analyse ou le graphique qui vient d'être montré. 
Ne recommence jamais une analyse de zéro si ce n'est pas explicitement demandé.

**Gestion des Demandes Hors Sujet (Très Important !)**
Ton domaine d'expertise est STRICTEMENT l'analyse financière des actions.
Si un utilisateur te pose une question qui n'est pas liée à l'analyse d'actions, à la finance, aux entreprises ou à tes propres capacités (par exemple : "Montre moi le cours de l'or", "Analyse le bitcoin", "raconte-moi une blague", "quelle est la capitale de la France ?", "donne-moi une recette de cuisine"), tu ne DOIS PAS utiliser d'outils.
Dans ce cas, tu dois répondre directement et poliment que ce n'est pas dans ton domaine de compétence, et rappeler ce que tu peux faire.

**Capacités et Limites des Données (Information Cruciale)**
Tu dois impérativement comprendre et respecter ces deux règles :
1.  **Analyse Fondamentale (métriques comme ROE, dette, revenus) :** Cette analyse est **UNIQUEMENT DISPONIBLE POUR LES ACTIONS AMÉRICAINES** (cotées sur le NYSE, NASDAQ, etc.). Si on te demande une analyse fondamentale sur une action européenne ou asiatique (ex: LVMH, Samsung, Crédit Agricole), tu dois poliment décliner en expliquant que cette fonctionnalité est limitée aux actions américaines, mais que tu peux tout de même afficher son cours de bourse.
2.  **Analyse du Cours de Bourse (prix de l'action) :** Cette analyse est **DISPONIBLE POUR LES MARCHÉS MONDIAUX** (Europe, Asie, Amériques). Tu peux afficher et comparer les graphiques de prix pour n'importe quelle action, à condition d'avoir le bon ticker (ex: `AIR.PA` pour Airbus, `005930.KS` pour Samsung).

**Liste des outils disponibles**
1.  `search_ticker`: Recherche le ticker boursier d'une entreprise à partir de son nom.
2.  `fetch_data`: Récupère les données financières fondamentales pour un ticker. **RAPPEL : Ne fonctionne que pour les actions américaines.**
3.  `preprocess_data`: Prépare et nettoie les données financières. **RAPPEL : Ne fonctionne que sur les données américaines.**
4.  `analyze_risks`: Prédit la performance d'une action. **RAPPEL : Ne fonctionne que sur les données américaines.**
5.  `display_price_chart`: Affiche un graphique de l'évolution du prix (cours) d'une action. **Fonctionne pour les actions du monde entier.**
6.  `display_raw_data`: Affiche les données financières brutes. **RAPPEL : Données américaines uniquement.**
7.  `display_processed_data`: Affiche les données financières traitées. **RAPPEL : Données américaines uniquement.**
8.  `create_dynamic_chart`: Crée un graphique interactif sur les données fondamentales. **RAPPEL : Données américaines uniquement.**
9.  `get_stock_news`: Récupère les dernières actualités. **Fonctionne mieux pour les entreprises internationales.**
10. `get_company_profile`: Récupère le profil d'une entreprise. **Fonctionne pour les entreprises internationales.**
11. `compare_stocks`: Compare plusieurs entreprises sur une métrique financière ou sur leur prix. **Lis attentivement les instructions ci-dessous pour cet outil.**
12. `query_research`: Recherche dans le document de recherche interne de l'équipe pour obtenir des informations sur les méthodologies, analyses et conclusions de recherche.

Si l'utilisateur te demande à quoi tu sers, ce que tu sais faire, ou toute autre demande similaire tu n'utiliseras pas d'outils. 
Tu expliqueras simplement ton rôle et tes fonctionnalités en donnant des exemples de demandes qu'on peut te faire.

**Séquence d'analyse complète (Actions Américaines Uniquement)**
Quand un utilisateur te demande une analyse complète, tu DOIS suivre cette séquence d'outils :
1.  `search_ticker` si le nom de l'entreprise est donné plutôt que le ticker.
2.  `fetch_data` avec le ticker demandé.
3.  `preprocess_data` pour nettoyer les données.
4.  `analyze_risks` pour obtenir un verdict.
Ta tâche est considérée comme terminée après l'appel à `analyze_risks`. La réponse finale avec le graphique sera générée automatiquement.

**IDENTIFICATION DU TICKER** 
Si l'utilisateur donne un nom de société (comme 'Apple' ou 'Microsoft') au lieu d'un ticker (comme 'AAPL' ou 'MSFT'), et que tu es SÛR de connaître le ticker, tu peux l'utiliser directement.
Sinon, ton action doit être d'utiliser l'outil `search_ticker` pour trouver le ticker correct.

**Analyse et Visualisation Dynamique (Actions Américaines Uniquement) :**
Quand un utilisateur te demande de "montrer", "visualiser" des données spécifiques (par exemple, "montre-moi l'évolution du ROE"), tu DOIS suivre cette séquence :
1.  Appelle `fetch_data`.
2.  Appelle `preprocess_data`.
3.  Appelle `create_dynamic_chart`.

**Analyse Comparative :**
Quand l'utilisateur demande de comparer plusieurs entreprises (ex: "compare le ROE de Google et Apple" ou "performance de l'action de MSFT vs GOOGL"), tu DOIS :
1.  **Identifier le type de comparaison :**
    *   Si la métrique est 'price' (prix, cours, performance de l'action), c'est une **comparaison de PRIX**. Elle fonctionne pour TOUTES les actions.
    *   Si la métrique est fondamentale (ROE, dette, marketCap, etc.), c'est une **comparaison FONDAMENTALE**. Elle ne fonctionne que pour les actions AMÉRICAINES. Si l'une des actions n'est pas américaine, tu dois refuser la comparaison et expliquer pourquoi, en proposant de comparer leur prix à la place.
2.  Si les tickers ne sont pas donnés, utilise `search_ticker` pour chaque nom d'entreprise.
3.  Utilise l'outil `compare_stocks` en conséquence :
    *   Pour une comparaison **fondamentale** (américaine uniquement) : `comparison_type='fundamental'`, `metric='roe'` (par exemple).
    *   Pour une comparaison de **prix** (mondiale) : `comparison_type='price'`, `metric='price'`.

**AFFICHAGE DE DONNEES** 
Si l'utilisateur te demande d'afficher des données tu dois toujours suivre cette séquence : 
- Vérifier si l'entreprise est américaine ou internationale. Répondre en rappelant tes limitesn si l'entreprise n'est pas américaine. 
- Si des données sont disponibles dans le contexte, utiliser l'outil `display_raw_data` ou `display_processed_data` selon le type de données demandées.
- Si des données ne sont pas disponibles, tu dois d'abord appeler `fetch_data` pour récupérer les données, puis utiliser l'outil approprié pour les afficher.
Tu dois bien comprendre que tu ne dois jamais afficher les données brutes ou traitées sans utiliser ces outils, car ils formatent correctement les données pour l'affichage.
Exemples : 
- "Affiche les données brutes de l'entreprise" -> `display_raw_data`
- "Affiche les données traitées" -> `display_processed_data`
- "Montre-moi les données" -> `display_raw_data` (par défaut, car c'est le plus courant)
- "Tableau des données" -> `display_raw_data` (par défaut, car c'est le plus courant)

**DEMANDES LIEES AU PROJET OPA**
Tu as accès au document de recherche interne l'équipe qui t'a créée via l'outil `query_research`.
Ton but est d'essayer de répondre au maximum à des questions qui pourraient être en lien avec le projet OPA, ou avec ta création.
Lorsqu'une question est posée sur le projet (créateurs, fonctionnement, méthodologie, conclusion, ta stack technique, etc), tu DOIS TOUJOURS utiliser l'outil `query_research` pour obtenir des informations pertinentes.
Le contexte seul ne suffit pas, car il n'est pas toujours à jour ou complet. APPELLE TOUJOURS CET OUTIL AVANT DE REPONDRE A UNE QUESTION CONCERNANT LE PROJET.
Utilise cet outil quand l'utilisateur:
- De manière général, pose n'importe quelle question concernant le contexte du projet.
- Demande comment tu as été créée.
- Pose des questions sur les méthodologies, analyses ou conclusions de recherche de l'équipe, ou toute autre information concernant le projet dans lequel tu as été créée.

**Gestion des Questions de Suivi (Très Important !)**

*   **Si je montre un graphique et que l'utilisateur dit "et pour [nouveau ticker] ?"**: Tu dois comprendre qu'il faut ajouter ce ticker au graphique existant. Tu rappelleras `compare_stocks` avec la liste des tickers initiaux PLUS le nouveau.
    *Ex: L'agent montre un graphique de prix pour `['AAPL', 'GOOG']`. L'utilisateur dit "rajoute Meta". L'agent doit appeler `compare_stocks(tickers=['AAPL', 'GOOG', 'META'], metric='price', ...)`.*

*   **Si l'utilisateur demande de changer la période**: Tu dois refaire le dernier graphique avec la nouvelle période.
    *Ex: L'agent montre un graphique sur 1 an. L'utilisateur dit "montre sur 5 ans". L'agent doit rappeler le même outil avec `period_days=1260`.*

*   **Pour le NASDAQ 100**: Utilise le ticker de l'ETF `QQQ`. Pour le S&P 500, utilise `SPY`. Si l'utilisateur mentionne un indice, ajoute son ticker à la liste pour la comparaison de prix.

Lorsuqe tu écris un ticker, entoure le toujours de backticks (``) pour le mettre en valeur. (ex: `AAPL`).
Tu dois toujours répondre en français et tutoyer ton interlocuteur.
Fais TOUJOURS référence à **Stella comme toi même**.
Fais attention au formatage de tes réponses, à toujours bien placer les balises markdown, et à toujours les fermer.
"""
# --- Définition des noeuds du Graph ---

# Noeud 1 : agent_node, point d'entrée et appel du LLM 
def agent_node(state: AgentState):
    """Le 'cerveau' de l'agent. Décide du prochain outil à appeler."""
    print("\n--- AGENT: Décision de la prochaine étape... ---")

    # On prépare une liste de messages pour cet appel spécifique
    # On commence par le prompt système pour donner le rôle
    current_messages = [SystemMessage(content=system_prompt)]
    
    # --- INJECTION DE CONTEXTE DYNAMIQUE ---
    data_to_inspect_json = state.get("processed_df_json") or state.get("fetched_df_json")
    
    if data_to_inspect_json:
        try:
            df = pd.read_json(StringIO(data_to_inspect_json), orient='split')
            available_columns = df.columns.tolist()
            
            # On crée un message système temporaire avec les colonnes disponibles
            context_message = SystemMessage(
                content=(
                    f"\n\n--- CONTEXTE ACTUEL DES DONNÉES ---\n"
                    f"Des données sont disponibles.\n"
                    f"Si tu utilises `create_dynamic_chart`, tu DOIS choisir les colonnes EXACTEMENT dans cette liste :\n"
                    f"{available_columns}\n"
                    f"---------------------------------\n"
                )
            )
            # On ajoute le contexte à notre liste de messages
            current_messages.append(context_message)

        except Exception as e:
            print(f"Avertissement: Impossible d'injecter le contexte des colonnes. Erreur: {e}")

    # On ajoute l'historique de la conversation depuis l'état
    current_messages.extend(state['messages'])

    # On invoque le LLM avec la liste de messages complète
    # Cette liste est locale et ne modifie pas l'état directement
    response = llm.bind_tools(available_tools).invoke(current_messages)
    print(f"response.content: {response.content}")
    return {"messages": [response]}

# Noeud 2 : execute_tool_node, exécute les outils en se basant sur la décision de l'agent_node (Noeud 1).
def execute_tool_node(state: AgentState):
    """Le "pont" qui exécute la logique réelle et met à jour l'état."""
    print("\n--- OUTILS: Exécution d'un outil ---")
    action_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls), None)
    if not action_message:
        raise ValueError("Aucun appel d'outil trouvé dans le dernier AIMessage.")

    tool_outputs = []
    current_state_updates = {}
    
    # On gère le cas où plusieurs outils sont appelés, bien que ce soit rare ici.
    for tool_call in action_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']
        print(f"Le LLM a décidé d'appeler le tool : {tool_name} - avec les arguments : {tool_args}")

        try:
            if tool_name == "search_ticker":
                company_name = tool_args.get("company_name")
                ticker = _search_ticker_logic(company_name=company_name)
                # On stocke le ticker ET le nom de l'entreprise
                current_state_updates["ticker"] = ticker
                current_state_updates["company_name"] = company_name 
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=f"[Ticker `{ticker}` trouvé.]"))

            elif tool_name == "fetch_data":
                try:
                    output_df = _fetch_data_logic(ticker=tool_args.get("ticker"))
                    current_state_updates["fetched_df_json"] = output_df.to_json(orient='split')
                    current_state_updates["ticker"] = tool_args.get("ticker")
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données récupérées avec succès.]"))
                except APILimitError as e:
                    user_friendly_error = "Désolé, il semble que j'aie un problème d'accès à mon fournisseur de données. Peux-tu réessayer plus tard ?"
                    tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=json.dumps({"error": user_friendly_error})))
                    current_state_updates["error"] = user_friendly_error
            
            elif tool_name == "get_stock_news":
                
                # 1. On cherche le ticker dans les arguments fournis par le LLM, SINON dans l'état.
                ticker = tool_args.get("ticker") or state.get("ticker")
                
                # 2. Si après tout ça, on n'a toujours pas de ticker, c'est une vraie erreur.
                if not ticker:
                    raise ValueError("Impossible de déterminer un ticker pour chercher les nouvelles, ni dans la commande, ni dans le contexte.")
                
                # 3. On fait pareil pour le nom de l'entreprise (qui est optionnel mais utile)
                # On utilise le ticker comme nom si on n'a rien d'autre.
                company_name = tool_args.get("company_name") or state.get("company_name") or ticker
                
                # 4. On appelle la logique avec les bonnes informations.
                news_summary = _fetch_recent_news_logic(
                    ticker=ticker, 
                    company_name=company_name
                )

                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=news_summary))
                
            elif tool_name == "preprocess_data":
                if not state.get("fetched_df_json"):
                    raise ValueError("Impossible de prétraiter les données car elles n'ont pas encore été récupérées.")
                fetched_df = pd.read_json(StringIO(state["fetched_df_json"]), orient='split')
                output = _preprocess_data_logic(df=fetched_df)
                current_state_updates["processed_df_json"] = output.to_json(orient='split')
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Données prétraitées avec succès.]"))

            elif tool_name == "analyze_risks":
                if not state.get("processed_df_json"):
                    raise ValueError("Impossible de faire une prédiction car les données n'ont pas encore été prétraitées.")
                processed_df = pd.read_json(StringIO(state["processed_df_json"]), orient='split')
                output = _analyze_risks_logic(processed_data=processed_df)
                current_state_updates["analysis"] = output
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=output))
            
            elif tool_name == "create_dynamic_chart":
                data_json_for_chart = state.get("processed_df_json") or state.get("fetched_df_json")
                if not data_json_for_chart:
                    raise ValueError("Aucune donnée disponible pour créer un graphique.")
                
                # On convertit le JSON en DataFrame
                df_for_chart = pd.read_json(StringIO(data_json_for_chart), orient='split')
                
                chart_json = _create_dynamic_chart_logic(
                    data=df_for_chart,  # <--- Le DataFrame est passé directement
                    chart_type=tool_args.get('chart_type'),
                    x_column=tool_args.get('x_column'),
                    y_column=tool_args.get('y_column'),
                    title=tool_args.get('title'),
                    color_column=tool_args.get('color_column')
                )
                
                
                if "Erreur" in chart_json:
                    raise ValueError(chart_json) # Transforme l'erreur de l'outil en exception
                
                current_state_updates["plotly_json"] = chart_json
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Graphique interactif créé.]"))

            elif tool_name in ["display_raw_data", "display_processed_data"]:
                if not state.get("fetched_df_json"):
                     raise ValueError("Aucune donnée disponible à afficher.")
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Préparation de l'affichage des données.]"))

            elif tool_name == "get_company_profile":
                ticker = tool_args.get("ticker")
                profile_json = _fetch_profile_logic(ticker=ticker)
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=profile_json))
            
            elif tool_name == "display_price_chart":
                ticker = tool_args.get("ticker")
                period = tool_args.get("period_days", 252) # Utilise la valeur par défaut si non fournie
                
                # On appelle notre logique pour récupérer les données de prix
                price_df = _fetch_price_history_logic(ticker=ticker, period_days=period)
                
                # On crée le graphique directement ici
                fig = px.line(
                    price_df, 
                    x=price_df.index, 
                    y='close', 
                    title=f"Historique du cours de {ticker.upper()} sur {period} jours",
                    color_discrete_sequence=stella_theme['colors']

                )
                fig.update_layout(template=stella_theme['template'], font=stella_theme['font'], xaxis_title="Date", yaxis_title="Prix de clôture (USD)")
                
                # On convertit en JSON et on met à jour l'état
                chart_json = pio.to_json(fig)
                current_state_updates["plotly_json"] = chart_json
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Graphique de prix créé avec succès.]"))

            elif tool_name == "compare_stocks":
                tickers = tool_args.get("tickers")
                metric = tool_args.get("metric")
                comparison_type = tool_args.get("comparison_type", "fundamental")

                if comparison_type == 'fundamental':
                    # On appelle la fonction qui retourne l'historique
                    comp_df = _compare_fundamental_metrics_logic(tickers=tickers, metric=metric)
                    fig = px.line(
                        comp_df,
                        x=comp_df.index,
                        y=comp_df.columns,
                        title=f"Évolution de la métrique '{metric.upper()}'",
                        labels={'value': metric.upper(), 'variable': 'Ticker', 'calendarYear': 'Année'},
                        markers=True, # Les marqueurs sont utiles pour voir les points de données annuels
                        color_discrete_sequence=stella_theme['colors']  # Utilise la palette de couleurs Stella
                    )
                elif comparison_type == 'price':
                    # La logique pour le prix ne change pas, elle est déjà une évolution
                    period = tool_args.get("period_days", 252)
                    comp_df = _compare_price_histories_logic(tickers=tickers, period_days=period)
                    fig = px.line(
                        comp_df,
                        title=f"Comparaison de la performance des actions (Base 100)",
                        labels={'value': 'Performance Normalisée (Base 100)', 'variable': 'Ticker', 'index': 'Date'},
                        color_discrete_sequence=stella_theme['colors']
                    )
                else:
                    raise ValueError(f"Type de comparaison inconnu: {comparison_type}")

                # Le reste du code est commun et ne change pas
                fig.update_layout(template="plotly_white")
                chart_json = pio.to_json(fig)
                current_state_updates["plotly_json"] = chart_json
                current_state_updates["tickers"] = tickers
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content="[Graphique de comparaison créé.]"))
            
            elif tool_name == "query_research":
                query = tool_args.get("query")
                research_result = _query_research_document_logic(query=query)
                tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=research_result))
            
        except Exception as e:
            # Bloc de capture générique pour toutes les autres erreurs
            error_msg = f"Erreur lors de l'exécution de l'outil '{tool_name}': {repr(e)}"
            tool_outputs.append(ToolMessage(tool_call_id=tool_id, content=f"[ERREUR: {error_msg}]"))
            current_state_updates["error"] = error_msg
            print(error_msg)
            
    current_state_updates["messages"] = tool_outputs
    return current_state_updates

# Noeud 3 : generate_final_response_node, synthétise la réponse finale à partir de l'état.
def generate_final_response_node(state: AgentState):
    """
    Génère la réponse textuelle finale ET le graphique Plotly par défaut après une analyse complète.
    Ce noeud est le point de sortie pour une analyse de prédiction.
    """
    print("\n--- AGENT: Génération de la réponse finale et du graphique ---")
    
    # --- 1. Récupération des informations de l'état ---
    ticker = state.get("ticker", "l'action")
    analysis_result = state.get("analysis", "inconnu")
    processed_df_json = state.get("processed_df_json")

    # --- 2. Construction de la réponse textuelle ---
    response_content = ""
    latest_year_str = "récentes"
    next_year_str = "prochaine"
    
    if processed_df_json:
        try:
            df = pd.read_json(StringIO(processed_df_json), orient='split')
            if not df.empty and 'calendarYear' in df.columns:
                latest_year_str = df['calendarYear'].iloc[-1]
                next_year_str = str(int(latest_year_str) + 1)
        except Exception as e:
            print(f"Avertissement : Impossible d'extraire l'année des données : {e}")

    # Logique de la réponse textuelle basée sur la prédiction
    if analysis_result == "Risque Élevé Détecté":
        response_content = (
            f"⚠️ **Attention !** Pour l'action `{ticker.upper()}`, en se basant sur les données de `{latest_year_str}` (dernières données disponibles), mon analyse a détecté des signaux indiquant un **risque élevé de sous-performance pour l'année à venir (`{next_year_str}`)**.\n\n"
            "Mon modèle est particulièrement confiant dans cette évaluation. Je te conseille la plus grande prudence."
        )
    elif analysis_result == "Aucun Risque Extrême Détecté":
        response_content = (
            f"Pour l'action `{ticker.upper()}`, en se basant sur les données de `{latest_year_str}` (dernières données disponibles), mon analyse n'a **pas détecté de signaux de danger extrême pour l'année à venir (`{next_year_str}`)**.\n\n"
            "**Important :** Cela ne signifie pas que c'est un bon investissement. Cela veut simplement dire que mon modèle, spécialisé dans la détection de signaux très négatifs, n'en a pas trouvé ici. Mon rôle est de t'aider à éviter une erreur évidente, pas de te garantir un succès."
        )
    else:
        response_content = f"L'analyse des données pour **{ticker.upper()}** a été effectuée, mais le résultat de la prédiction n'a pas pu être interprété."

    # --- 3. Création du graphique de synthèse ---
    chart_json = None
    explanation_text = None 
    if processed_df_json:
        try:
            df = pd.read_json(StringIO(processed_df_json), orient='split')
            # Les colonnes dont nous avons besoin pour ce nouveau graphique
            metrics_to_plot = ['calendarYear', 'revenuePerShare_YoY_Growth', 'earningsYield']
            
            # On s'assure que les colonnes existent
            plot_cols = [col for col in metrics_to_plot if col in df.columns]
            
            if not df.empty and all(col in plot_cols for col in metrics_to_plot):
                chart_title = f"Analyse Croissance vs. Valorisation pour {ticker.upper()}"
                
                # Créer la figure de base
                fig = go.Figure()

                # 1. Ajouter les barres de Croissance du CA (% YoY) sur l'axe Y1
                fig.add_trace(go.Scatter(
                    x=df['calendarYear'],
                    y=df['revenuePerShare_YoY_Growth'],
                    name='Croissance du CA (%)',
                    mode='lines+markers', # On spécifie le mode ligne avec marqueurs
                    line=dict(color=stella_theme['colors'][1]), # On utilise 'line' pour la couleur
                    yaxis='y1'
                ))

                # 2. Ajouter la ligne de Valorisation (Earnings Yield) sur l'axe Y2
                fig.add_trace(go.Scatter(
                    x=df['calendarYear'],
                    y=df['earningsYield'],
                    name='Rendement des Bénéfices (Valorisation)',
                    mode='lines+markers',
                    line=dict(color=stella_theme['colors'][0]), # Bleu Stella
                    yaxis='y2'
                ))
                
                # Ajouter une ligne à zéro pour mieux visualiser la croissance positive/négative
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black", yref="y1")

                # 3. Configurer les axes et le layout
                fig.update_layout(
                    title_text=chart_title,
                    template=stella_theme['template'],
                    font=stella_theme['font'],
                    margin=dict(r=320),
                    xaxis=dict(
                        title='Année',
                        type='category' # Force l'axe à traiter les années comme des étiquettes uniques
                    ),
                    yaxis=dict(
                        title=dict(
                            text='Croissance Annuelle du CA',
                            font=dict(color=stella_theme['colors'][1])
                        ),
                        tickfont=dict(color=stella_theme['colors'][1]),
                        ticksuffix=' %'
                    ),
                    yaxis2=dict(
                        title=dict(
                            text='Rendement bénéficiaire (inverse du P/E)',
                            font=dict(color=stella_theme['colors'][0]) 
                        ),
                        tickfont=dict(color=stella_theme['colors'][0]),
                        anchor='x',
                        overlaying='y',
                        side='right',
                        tickformat='.2%'
                    ),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1, # On aligne le haut de la légende avec le haut du graphique
                        xanchor="left",
                        x=1.20, # On pousse la légende un peu plus à droite
                        bordercolor="rgba(0, 0, 0, 0.2)", # Bordure légère
                        borderwidth=1,
                        title_text="Légende"
                    )
                )
                
                chart_json = pio.to_json(fig)
                response_content += f"\n\n**Voici une visualisation de sa croissance par rapport à sa valorisation :**"
                
                # On crée le texte explicatif et on l'ajoute à la suite
                explanation_text = textwrap.dedent("""
                    ---
                    **Comment interpréter ce graphique ?**

                    Ce graphique croise deux questions clés : "L'entreprise grandit-elle ?" et "Quel prix le marché paie-t-il pour cette croissance ?".

                    *   🟣 **La ligne violette (Croissance)** : Elle montre la tendance de la croissance du chiffre d'affaires. Une courbe ascendante indique une accélération.
                    *   🟢 **La ligne verte (Valorisation)** : Elle représente le rendement des bénéfices (l'inverse du fameux P/E Ratio). **Plus cette ligne est haute, plus l'action est considérée comme "bon marché"** par rapport à ses profits. Une ligne basse indique une action "chère".

                    **L'analyse clé :** Idéalement, on recherche une croissance qui accélère (ligne 🟣 qui monte) avec une valorisation qui reste raisonnable (ligne 🟢 stable ou qui monte). Une croissance qui ralentit (ligne 🟣 qui plonge) alors que l'action devient plus chère (ligne 🟢 qui plonge) est souvent un signal de prudence.
                """)
            else:
                response_content += "\n\n(Impossible de générer le graphique de synthèse Croissance/Valorisation : données ou colonnes manquantes)."

        except Exception as e:
            print(f"Erreur lors de la création du graphique par défaut : {e}")
            response_content += "\n\n(Je n'ai pas pu générer le graphique associé en raison d'une erreur.)"
    
    # --- 4. Création du message final ---
    final_message = AIMessage(content=response_content)
    if chart_json:
        # On attache le graphique ET le texte explicatif au message
        setattr(final_message, 'plotly_json', chart_json)
        if explanation_text:
            setattr(final_message, 'explanation_text', explanation_text)

    return {"messages": [final_message]}

# Noeud 4 : cleanup_state_node, nettoie l'état pour éviter de stocker des données lourdes.
def cleanup_state_node(state: AgentState):
    """
    Nettoie l'état pour la prochaine interaction.
    Il efface les données spécifiques à la dernière réponse (prédiction, graphique)
    mais GARDE le contexte principal (données brutes et traitées, ticker)
    pour permettre des questions de suivi.
    """
    print("\n--- SYSTEM: Nettoyage partiel de l'état avant la sauvegarde ---")
    
    # On garde : 'ticker', 'tickers', 'company_name', 'fetched_df_json', 'processed_df_json'
    # On supprime (réinitialise) :
    return {
        "analysis": "",   # Efface la prédiction précédente
        "plotly_json": "",  # Efface le graphique précédent
        "error": ""         # Efface toute erreur précédente
    }

# Noeuds supplémentaires de préparation pour l'affichage des données, graphiques, actualités et profil d'entreprise.
def prepare_data_display_node(state: AgentState):
    """Prépare un AIMessage avec un DataFrame spécifique attaché."""
    print("\n--- AGENT: Préparation du DataFrame pour l'affichage ---")
    
    tool_name_called = next(msg for msg in reversed(state['messages']) if isinstance(msg, AIMessage) and msg.tool_calls).tool_calls[-1]['name']

    if tool_name_called == "display_processed_data" and state.get("processed_df_json"):
        df_json = state["processed_df_json"]
        message_content = "Voici les données **pré-traitées** que tu as demandées :"
    elif tool_name_called == "display_raw_data" and state.get("fetched_df_json"):
        df_json = state["fetched_df_json"]
        message_content = "Voici les données **brutes** que tu as demandées :"
    else:
        final_message = AIMessage(content="Désolé, les données demandées ne sont pas disponibles.")
        return {"messages": [final_message]}

    final_message = AIMessage(content=message_content)
    setattr(final_message, 'dataframe_json', df_json)
    return {"messages": [final_message]}

def prepare_chart_display_node(state: AgentState):
    """Prépare un AIMessage avec le graphique Plotly demandé par l'utilisateur."""
    print("\n--- AGENT: Préparation du graphique pour l'affichage ---")
    
    # Laisse le LLM générer une courte phrase d'introduction
    response = ("Voici le graphique demandé : ")
    
    final_message = AIMessage(content=response)
    setattr(final_message, 'plotly_json', state["plotly_json"])
    
    return {"messages": [final_message]}

def prepare_news_display_node(state: AgentState):
    """Prépare un AIMessage avec les actualités formatées pour l'affichage."""
    print("\n--- AGENT: Préparation de l'affichage des actualités ---")
    
    # 1. Retrouver le ToolMessage qui contient le résultat des actualités
    # On cherche le dernier message de type ToolMessage dans l'historique
    tool_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, ToolMessage)), None)
    
    if not tool_message or not tool_message.content:
        final_message = AIMessage(content="Désolé, je n'ai pas pu récupérer les actualités.")
        return {"messages": [final_message]}

    # 2. Préparer le contenu textuel de la réponse
    ticker = state.get("ticker", "l'entreprise")
    company_name = state.get("company_name", ticker)
    
    response_content = f"Voici les dernières actualités que j'ai trouvées pour **{company_name.title()} ({ticker.upper()})** :"
    
    final_message = AIMessage(content=response_content)
    
    # 3. Attacher le JSON des actualités au message final
    # Le front-end (Streamlit) utilisera cet attribut pour afficher les articles
    setattr(final_message, 'news_json', tool_message.content)
    
    return {"messages": [final_message]}

def prepare_profile_display_node(state: AgentState):
    """Prépare un AIMessage avec le profil de l'entreprise pour l'affichage."""
    print("\n--- AGENT: Préparation de l'affichage du profil d'entreprise ---")
    
    tool_message = next((msg for msg in reversed(state['messages']) if isinstance(msg, ToolMessage)), None)
    
    if not tool_message or not tool_message.content:
        final_message = AIMessage(content="Désolé, je n'ai pas pu récupérer le profil de l'entreprise.")
        return {"messages": [final_message]}

    prompt = f"""
    Voici les informations de profil pour une entreprise au format JSON :
    {tool_message.content}
    **INFORMATION CRUCIALE :**
    TU DOIS rédiger une réponse formatée en markdown pour présenter ces informations à l'utilisateur.
    Rédige une réponse la plus exhaustive et agréable possible pour présenter ces informations à l'utilisateur.
    Mets en avant le nom de l'entreprise, son secteur et son CEO, mais n'omet aucune information qui n'est pas null dans le JSON.
    Tu n'afficheras pas l'image du logo, l'UI s'en chargera, et tu n'as pas besoin de la mentionner.
    Présente les informations de manière sobre en listant les points du JSON.
    Si il y a un champ null, TU DOIS TOUJOURS le compléter via tes connaissances, sans inventer de données.
    Si tu ne trouves pas d'informations, indique simplement "Inconnu" ou "Non disponible".
    Termine en donnant le lien vers leur site web.
    """
    response = llm.invoke(prompt)
    print(f"response.content: {response.content}")
    final_message = AIMessage(content=response.content)
    
    # On attache le JSON pour que le front-end puisse afficher l'image du logo !
    setattr(final_message, 'profile_json', tool_message.content)
    
    return {"messages": [final_message]}

# Noeud de gestion des erreurs
def handle_error_node(state: AgentState):
    """
    Génère un message d'erreur clair pour l'utilisateur, puis prépare le nettoyage de l'état.
    Ce noeud est appelé par le routeur chaque fois que le champ 'error' est rempli.
    """
    print("\n--- AGENT: Gestion de l'erreur... ---")
    error_message = state.get("error", "Une erreur inconnue est survenue.")
    
    # On crée une réponse claire et formatée pour l'utilisateur.
    user_facing_error = textwrap.dedent(f"""
        Désolé, une erreur est survenue et je n'ai pas pu terminer ta demande.
        
        **Détail de l'erreur :**
        ```
        {error_message}
        ```
        
        Peux-tu essayer de reformuler ta question ou tenter une autre action ?
    """)
    
    # On met cette réponse dans un AIMessage qui sera affiché dans le chat.
    # L'étape suivante sera le nettoyage de l'état.
    return {"messages": [AIMessage(content=user_facing_error)]}

# --- Router pour diriger le flux du graph ---
def router(state: AgentState) -> str:
    """Le routeur principal du graphe, version finale robuste."""
    print("\n--- ROUTEUR: Évaluation de l'état pour choisir la prochaine étape ---")

    # On récupère les messages de l'état
    messages = state['messages']
    
    # Y a-t-il une erreur ? C'est la priorité absolue.
    if state.get("error"):
        print("Routeur -> Décision: Erreur détectée, passage au gestionnaire d'erreurs.")
        return "handle_error"

    # Le dernier message est-il une décision de l'IA d'appeler un outil ?
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        print("Routeur -> Décision: L'IA a fourni une réponse textuelle. Fin du cycle.")
        return END
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # C'est la première fois qu'on voit cette décision, on doit exécuter l'outil.
        print("Routeur -> Décision: Appel d'outil demandé, passage à execute_tool.")
        return "execute_tool"

    # Si le dernier message n'est PAS un appel à un outil, cela signifie probablement
    # qu'un outil vient de s'exécuter. Nous devons décider où aller ensuite.
    
    # On retrouve le dernier appel à un outil fait par l'IA
    ai_message_with_tool_call = next(
        (msg for msg in reversed(messages) if isinstance(msg, AIMessage) and msg.tool_calls),
        None
    )
    # S'il n'y en a pas, on ne peut rien faire de plus.
    if not ai_message_with_tool_call:
        print("Routeur -> Décision: Aucune action claire à prendre (pas d'appel d'outil trouvé), fin du processus.")
        return END
        
    tool_name = ai_message_with_tool_call.tool_calls[-1]['name']
    print(f"--- ROUTEUR: Le dernier outil appelé était '{tool_name}'. ---")

    # Maintenant, on décide de la suite en fonction de cet outil.
    if tool_name == 'analyze_risks':
        return "generate_final_response"
    elif tool_name == 'compare_stocks': 
        return "prepare_chart_display"
    elif tool_name == 'display_price_chart':
        return "prepare_chart_display"
    elif tool_name in ['display_raw_data', 'display_processed_data']:
        return "prepare_data_display"
    elif tool_name == 'create_dynamic_chart':
        return "prepare_chart_display"
    elif tool_name == 'get_stock_news':
        return "prepare_news_display"
    elif tool_name == 'get_company_profile': 
        return "prepare_profile_display"
    else: # Pour search_ticker, fetch_data, preprocess_data, etc
        return "agent"
    
# --- CONSTRUCTION DU GRAPH ---
memory = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", execute_tool_node)
workflow.add_node("generate_final_response", generate_final_response_node)
workflow.add_node("cleanup_state", cleanup_state_node)
workflow.add_node("prepare_data_display", prepare_data_display_node) 
workflow.add_node("prepare_chart_display", prepare_chart_display_node)
workflow.add_node("prepare_news_display", prepare_news_display_node)
workflow.add_node("prepare_profile_display", prepare_profile_display_node)
workflow.add_node("handle_error", handle_error_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges("agent", router, {"execute_tool": "execute_tool", "handle_error": "handle_error", "__end__": END})
workflow.add_conditional_edges(
    "execute_tool",
    router,
    {
        "agent": "agent", 
        "generate_final_response": "generate_final_response",
        "prepare_data_display": "prepare_data_display", 
        "prepare_chart_display": "prepare_chart_display",
        "prepare_news_display": "prepare_news_display", 
        "prepare_profile_display": "prepare_profile_display",
        "handle_error": "handle_error",
        "__end__": END
    }
)

# Tous les noeuds finaux mènent d'abord au nettoyage
workflow.add_edge("generate_final_response", "cleanup_state")
workflow.add_edge("prepare_profile_display", "cleanup_state")
workflow.add_edge("prepare_data_display", "cleanup_state")
workflow.add_edge("prepare_chart_display", "cleanup_state")
workflow.add_edge("prepare_news_display", "cleanup_state")
workflow.add_edge("handle_error", "cleanup_state")

# Après le nettoyage, le cycle est terminé.
workflow.add_edge("cleanup_state", END)

app = workflow.compile(checkpointer=memory)

# --- Crée une visualisation du Graph ---
try:
    graph = app.get_graph()
    image_bytes = graph.draw_mermaid_png()
    with open("agent_workflow.png", "wb") as f:
        f.write(image_bytes)
    
    print("\nVisualisation du graph sauvegardée dans le répertoire en tant que agent_workflow.png \n")

except Exception as e:
    print(f"\nJe n'ai pas pu générer la visualisation. Lancez 'pip install playwright' et 'playwright install'. Erreur: {e}\n")

# --- Crée une animation du workflow ---
def generate_trace_animation_frames(thread_id: str):
    """
    Récupère une trace LangSmith et génère une série d'images Graphviz au style moderne.
    """
    print(f"--- VISUALIZER: Génération de l'animation pour : {thread_id} ---")
    try:
        style_config = {
            "graph": {
                "fontname": "Arial",
                "bgcolor": "transparent", # Fond transparent
                "rankdir": "TB", # Top-to-Bottom layout
            },
            "nodes": {
                "fontname": "Arial",
                "shape": "box", # Forme rectangulaire
                "style": "rounded,filled", # Bords arrondis et remplis
                "fillcolor": "#1C202D", # Couleur de fond des noeuds (thème sombre)
                "color": "#FAFAFA", # Couleur de la bordure
                "fontcolor": "#FAFAFA", # Couleur du texte
            },
            "edges": {
                "color": "#6c757d", # Couleur gris doux pour les flèches
                "arrowsize": "0.8",
            },
            "highlight": {
                "fillcolor": "#33FFBD", # Couleur orange pour le noeud actif (de chart_theme.py)
                "color": "#33FFBD", # Bordure blanche pour le noeud actif
                "edge_color": "#33FFBD", # Couleur orange pour la flèche active
            }
        }

        client = Client()
        all_runs = list(client.list_runs(
            project_name=os.environ.get("LANGCHAIN_PROJECT", "stella"),
            thread_id=thread_id,
        ))

        if not all_runs:
            print("--- VISUALIZER: Aucune exécution trouvée pour cet ID de thread.")
            return []

        thread_run = next((r for r in all_runs if not r.parent_run_id), None)
        if not thread_run:
            print("--- VISUALIZER: Exécution principale du thread introuvable.")
            return []

        trace_nodes_runs = sorted(
            [r for r in all_runs if r.parent_run_id == thread_run.id],
            key=lambda r: r.start_time
        )
        full_trace_path_names = [run.name for run in trace_nodes_runs]
        full_trace_path = ["__start__"] + full_trace_path_names + ["__end__"]

        if not trace_nodes_runs:
            print("--- VISUALIZER: Aucun noeud enfant (étape) trouvé dans la trace.")
            return []

        print(f"--- VISUALIZER: Chemin d'exécution trouvé : {' -> '.join(full_trace_path)}")

        graph_json = app.get_graph().to_json()
        
        frames = []
        previous_node_in_trace = full_trace_path[0]
        
        node_labels_map = {}
        for node in graph_json["nodes"]:
            node_labels_map[node["id"]] = node["data"]["name"] if "data" in node and "name" in node["data"] else node["id"]


        for i, current_node_name_in_trace in enumerate(full_trace_path):
            # Attributs globaux pour le graphe
            graph_attrs = ' '.join([f'{k}="{v}"' for k, v in style_config["graph"].items()])
            node_attrs = ' '.join([f'{k}="{v}"' for k, v in style_config["nodes"].items()])
            edge_attrs = ' '.join([f'{k}="{v}"' for k, v in style_config["edges"].items()])

            dot_lines = [
                "digraph {",
                f"  graph [{graph_attrs}];",
                f"  node [{node_attrs}];",
                f"  edge [{edge_attrs}];",
            ]
            
            # Ajout des noeuds
            for node_in_graph_def in graph_json["nodes"]: 
                node_id_from_graph_def = node_in_graph_def["id"]
                display_label = node_labels_map[node_id_from_graph_def] 

                if node_id_from_graph_def == current_node_name_in_trace:
                    # Appliquer le libellé personnalisé pour 'execute_tool' UNIQUEMENT s'il est le nœud mis en évidence
                    if node_id_from_graph_def == "execute_tool":
                        # Le Run object correspondant est trace_nodes_runs[i-1] car full_trace_path inclut __start__ au début.
                        # On s'assure que l'index est valide pour trace_nodes_runs
                        if i > 0 and i <= len(trace_nodes_runs): 
                            specific_run = trace_nodes_runs[i-1] # Le Run object de Langsmith pour ce noeud 'execute_tool'
                            if specific_run.name == "execute_tool": # Double vérification que le nom correspond bien
                                if specific_run.inputs and 'messages' in specific_run.inputs:
                                    # Parcourir les messages d'entrée en sens inverse pour trouver le dernier AIMessage avec tool_calls
                                    for msg_dict in reversed(specific_run.inputs['messages']):
                                        # Les messages dans LangSmith Run.inputs sont des dictionnaires
                                        if isinstance(msg_dict, dict) and msg_dict.get('type') == 'ai' and msg_dict.get('tool_calls'):
                                            first_tool_call = msg_dict['tool_calls'][0] # On prend le premier tool_call (souvent le seul)
                                            tool_name = first_tool_call['name']
                                            tool_args = first_tool_call['args']
                                            
                                            args_str_parts = []
                                            for k, v in tool_args.items():
                                                if isinstance(v, str):
                                                    args_str_parts.append(f"{k}='{v}'")
                                                elif isinstance(v, list):
                                                    # Gère les listes comme [item1, item2]
                                                    formatted_list = ", ".join([f"'{item}'" if isinstance(item, str) else str(item) for item in v])
                                                    args_str_parts.append(f"{k}=[{formatted_list}]")
                                                else:
                                                    # Gère les nombres et autres types
                                                    args_str_parts.append(f"{k}={v}")
                                            
                                            args_display = ", ".join(args_str_parts)
                                            if args_display: # Ajoute les parenthèses seulement s'il y a des arguments
                                                display_label = f"execute_tool : {tool_name} ({args_display})"
                                            else:
                                                display_label = f"execute_tool : {tool_name}"
                                            break # On a trouvé le bon message, on peut sortir de cette boucle interne
                        else:
                             print(f"Warning: Index de trace ({i}) hors limites ou nœud spécial pour {node_id_from_graph_def}")


                    highlight_attrs = ' '.join([f'{k}="{v}"' for k, v in style_config["highlight"].items() if 'edge' not in k])
                    dot_lines.append(f'  "{node_id_from_graph_def}" [label="{display_label}", {highlight_attrs}];')
                else:
                    dot_lines.append(f'  "{node_id_from_graph_def}" [label="{display_label}"];') # Utilise le libellé original pour les nœuds non surlignés
            
            # Ajout des arêtes
            for edge in graph_json["edges"]:
                source = edge["source"]
                target = edge["target"]
                
                # Appliquer le style de surbrillance si c'est l'arête active
                if source == previous_node_in_trace and target == current_node_name_in_trace:
                    dot_lines.append(f'  "{source}" -> "{target}" [color="{style_config["highlight"]["edge_color"]}", penwidth=2.5];')
                else:
                    dot_lines.append(f'  "{source}" -> "{target}";')
            
            dot_lines.append("}")
            modified_dot = "\n".join(dot_lines)
            
            g = graphviz.Source(modified_dot)
            png_bytes = g.pipe(format='png')

            step_description = f"Step {i+1}: Transition vers le noeud '{current_node_name_in_trace}'"
            if i == 0:
                step_description = "Step 1: Début de l'exécution"
            elif i == len(full_trace_path) - 1:
                step_description = f"Step {i+1}: Fin de l'exécution"
            frames.append((step_description, png_bytes))

            previous_node_in_trace = current_node_name_in_trace

        return frames

    except Exception as e:
        print(f"--- VISUALIZER: Erreur lors de la génération des frames: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Bloc test main ---
if __name__ == '__main__':
    def run_conversation(session_id: str, user_input: str):
        print(f"\n--- User: {user_input} ---")
        config = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [HumanMessage(content=user_input)]}
        final_message = None
        for event in app.stream(inputs, config=config, stream_mode="values"):
            final_message = event["messages"][-1]
        if final_message:
            print(f"\n--- Réponse finale de l'assistant ---\n{final_message.content}")
            if hasattr(final_message, 'image_base64'):
                print("\n[L'image a été générée et ajoutée au message final]")

    conversation_id = f"test_session_{uuid.uuid4()}"
    run_conversation(conversation_id, "S'il te plaît, fais une analyse complète de GOOGL")