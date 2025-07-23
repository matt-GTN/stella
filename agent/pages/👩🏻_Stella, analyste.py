# app.py
import streamlit as st
import os
import uuid
import base64
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from io import StringIO
import json
import textwrap


from agent import app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import base64
import os

# Fonction pour encoder une image locale en Base64
def get_image_as_base64(path):
    # Vérifie si le fichier existe
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

STELLA_AVATAR = "agent/assets/avatar_stella.png" # Chemin vers l'avatar de Stella

st.set_page_config(page_title="Assistant financier IA", page_icon="📈", layout="wide")
st.title("📈 Analyste financier IA")

st.markdown("""
    <style>
        /* Cible les éléments de message de chat dans Streamlit */
        .stChatMessage .st-emotion-cache-1w7qfeb {
            font-size: 18px; /* Valeur à modifier pour changer la taille de la police */
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialisation du session_state pour les messages et d'un ID de session unique ---
if "messages" not in st.session_state:
    welcome_message = textwrap.dedent("""
    Hello ! Je suis Stella. Je peux t'aider à analyser le potentiel d'une action. Que souhaites-tu faire ?
    
    *(Si tu ne sais pas par où démarrer, tu peux me demander de t'expliquer comment je peux t'aider.)*
    """)
    st.session_state.messages = [AIMessage(content=welcome_message)]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Affichage des messages existant depuis l'historique---
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=STELLA_AVATAR):
            st.markdown(msg.content)

            # Logique pour le DataFrame 
            if hasattr(msg, 'dataframe_json') and msg.dataframe_json:
                try:
                    df = pd.read_json(StringIO(msg.dataframe_json), orient='split')
                    st.dataframe(df, key=f"df_{i}") 
                except Exception as e:
                    st.error(f"Impossible d'afficher le DataFrame : {e}")

            # --- Logique pour les graphiques Plotly ---
            if hasattr(msg, 'plotly_json') and msg.plotly_json:
                try:
                    fig = go.Figure(pio.from_json(msg.plotly_json))
                    st.plotly_chart(fig, use_container_width=True, key=f"df_{i}")
                except Exception as e:
                    st.error(f"Impossible d'afficher le graphique : {e}")

            # --- Logique pour le texte explicatif ---
            if hasattr(msg, 'explanation_text') and msg.explanation_text:
                st.markdown(msg.explanation_text)

            # --- Logique pour le profil d'entreprise ---
            if hasattr(msg, 'profile_json') and msg.profile_json:
                try:
                    profile_data = json.loads(msg.profile_json)
                    if profile_data.get("image"):
                        # On peut afficher le logo à côté du titre pour un effet pro
                        st.image(profile_data["image"], width=60)
                except Exception as e:
                    print(f"Erreur affichage logo: {e}")

            # --- Logique pour les News ---
            if hasattr(msg, 'news_json') and msg.news_json:
                try:
                    news_articles = json.loads(msg.news_json)
                    if not news_articles:
                        st.info("Je n'ai trouvé aucune actualité récente.")
                    else:
                        # On ajoute un peu d'espace avant les articles
                        st.write("---") 
                        
                        for article in news_articles:
                            # On crée deux colonnes : une petite pour l'image, une grande pour le texte
                            col1, col2 = st.columns([1, 4]) # Ratio 1:4

                            with col1:
                                # On affiche l'image si elle existe
                                if article.get('image'):
                                    st.image(
                                        article['image'], 
                                        width=180, # On fixe une largeur pour que les images soient uniformes
                                        use_container_width='never' # Important pour respecter la largeur fixée
                                    )
                                else:
                                    # Placeholder si pas d'image, pour garder l'alignement
                                    st.text(" ") 

                            with col2:
                                # On affiche le titre, la source et le lien
                                st.markdown(f"**{article['title']}**")
                                st.caption(f"Source : {article.get('site', 'N/A')}")
                                st.markdown(f"<small><a href='{article['url']}' target='_blank'>Lire l'article</a></small>", unsafe_allow_html=True)
                            
                            # On ajoute un séparateur horizontal entre chaque article pour la clarté
                            st.divider()

                except Exception as e:
                    st.error(f"Impossible d'afficher les actualités : {e}")

    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

# --- Gestion de l'input utilisateur ---
if prompt := st.chat_input("Qu'est ce que je peux faire pour toi aujourd'hui ? 😊​"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant", avatar=STELLA_AVATAR):
        thinking_placeholder = st.empty()
        thinking_placeholder.write("🧠 Hmm, laisse-moi réfléchir une seconde...")

        inputs = {"messages": st.session_state.messages}
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        
        final_response = None
        
        try:
            # On streame les events pour afficher les étapes en temps réel
            for event in app.stream(inputs, config=config, stream_mode="values"):
                last_message = event["messages"][-1]
                
                # On vérifie si l'IA a décidé d'appeler un outil
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    tool_call = last_message.tool_calls[0] # On se concentre sur le premier appel
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # --- Outils de recherche initiaux ---
                    if tool_name == 'search_ticker':
                        company_name = tool_args.get('company_name', 'l\'entreprise demandée')
                        thinking_placeholder.write(f"🔍 Parfait, je commence par chercher l'identifiant boursier pour **{company_name}**...")
                    
                    elif tool_name == 'get_company_profile':
                        ticker = tool_args.get('ticker', 'l\'action')
                        thinking_placeholder.write(f"ℹ️ D'accord, je rassemble les informations générales (secteur, activité...) pour `{ticker.upper()}`.")
                    
                    # --- Outils de récupération de données ---
                    elif tool_name == 'fetch_data':
                        ticker = tool_args.get('ticker', 'l\'action')
                        thinking_placeholder.write(f"📊 Je récupère maintenant les données fondamentales pour `{ticker.upper()}`. Un instant...")
                        
                    elif tool_name == 'get_stock_news':
                        ticker = tool_args.get('ticker', 'l\'action')
                        thinking_placeholder.write(f"📰 Je consulte les dernières news pour voir ce qui se dit sur `{ticker.upper()}`.")

                    # --- Outils d'analyse complète ---
                    elif tool_name == 'preprocess_data':
                        thinking_placeholder.write("⚙️ Les données sont là ! Je les nettoie et calcule quelques indicateurs clés pour mon analyse...")
                    
                    elif tool_name == 'analyze_risks':
                        thinking_placeholder.write("🔮 Je soumets les données à mon modèle de prédiction pour évaluer les risques...")

                    # --- Outils de visualisation (demandés par l'utilisateur) ---
                    elif tool_name == 'display_price_chart':
                        ticker = tool_args.get('ticker', 'l\'action')
                        thinking_placeholder.write(f"📈 Préparation du graphique de l'évolution du prix pour `{ticker.upper()}`...")
                    
                    elif tool_name == 'create_dynamic_chart':
                        ticker = tool_args.get('ticker', 'l\'action')
                        metric = tool_args.get('y_column', 'la métrique demandée')
                        thinking_placeholder.write(f"🎨 Je construis le graphique personnalisé pour visualiser `{ticker.upper()}`.")
                        
                    elif tool_name == 'compare_stocks':
                        tickers = tool_args.get('tickers', [])
                        metric = tool_args.get('metric', 'la métrique')
                        if metric == 'price':
                             thinking_placeholder.write(f"🚀 Comparaison des performances de `{', '.join(tickers)}`... Je normalise les prix pour un graphique équitable.")
                        else:
                             thinking_placeholder.write(f"🔬 Analyse comparative de la métrique **'{metric}'** pour `{', '.join(tickers)}`. Cela peut prendre un moment, je récupère les données pour chaque entreprise.")

                # La réponse finale est la dernière AIMessage SANS appel d'outil
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    final_response = last_message

            thinking_placeholder.empty()

            if final_response:
                st.session_state.messages.append(final_response)

                # On enregistre l'ID de cette conversation pour que la page de visualisation puisse l'utiliser
                st.session_state.last_run_id = st.session_state.session_id
                st.toast("✅ Exécution terminée ! Vous pouvez maintenant la visualiser sur la page 'Visualize Run'.")
            else:
                fallback_response = AIMessage(content="Désolée, je semble avoir rencontré une erreur en cours de route. Peux-tu réessayer ou reformuler ta demande ?")
                st.session_state.messages.append(fallback_response)
        
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"Oups ! Une erreur inattendue et un peu technique s'est produite. Voici le détail pour les curieux : {e}"
            st.error(error_msg)
            st.session_state.messages.append(AIMessage(content=error_msg))
            import traceback
            traceback.print_exc()

        # Rafraîchit la page pour afficher le nouveau message ajouté à l'historique
        st.rerun()
