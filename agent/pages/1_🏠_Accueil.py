import streamlit as st

st.set_page_config(page_title="Accueil", page_icon="🏠", layout="wide")

st.title("🏠 Accueil – Assistant Financier IA")

st.info("""
Bienvenue dans l'interface de présentation de notre projet : **Stella**, votre assistante IA dédiée à l'analyse d'actions.
Voici un aperçu des différentes pages de l'application :
""")

with st.container(border=True):
    st.markdown("## 👩🏻 Stella, analyste")
    st.markdown("""
    Discutez avec **Stella**, l'assistante IA. Elle peut :
    - Analyser les données fondamentales d'une entreprise (USA uniquement)
    - Prédire son risque de sous-performance
    - Comparer avec d'autres entreprises ou indices
    - Et plus encore !
    """)
    st.page_link("pages/2_👩🏻_Stella,_analyste.py", label="Accéder à Stella", icon="👉")

with st.container(border=True):
    st.markdown("## 🧠 Modélisation")
    st.markdown("""
    Explorez et ajustez les paramètres d'un **Random Forest Classifier** pour comprendre :
    - Quels facteurs influencent le plus le risque
    - Comment améliorer la précision du modèle
    - Les erreurs types et leur analyse via **SHAP**
    """)
    st.page_link("pages/3_🧠_Modélisation.py", label="Accéder à la Modélisation", icon="👉")

with st.container(border=True):
    st.markdown("## 🎬 Visualisation de l'agent")
    st.markdown("""
    Rejouez une exécution de Stella **étape par étape** :
    - Visualisez le raisonnement de l'IA
    - Naviguez ou animez les étapes de la décision
    - Identifiez les outils utilisés et dans quel ordre
    """)
    st.page_link("pages/4_🎬_Visualisation_de_l'agent.py", label="Accéder à la Visualisation", icon="👉")

with st.container(border=True):
    st.markdown("## 📄 Rapport de recherche")
    st.markdown("""
    Consultez ou téléchargez le rapport de recherche du projet :
    - Résumé des objectifs et résultats
    - Méthodologie utilisée
    - Recommandations et limites
    """)
    st.page_link("pages/5_📄_Rapport_de_recherche.py", label="Accéder au Rapport", icon="👉")

st.markdown("---")
st.info("💡 Astuce : Vous pouvez toujours revenir ici en cliquant sur **Accueil** dans la barre latérale.")