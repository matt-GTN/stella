# agent/app.py (Le nouveau fichier d'accueil)

import streamlit as st

# Configure la page pour qu'elle ait un titre, mais elle ne sera visible qu'une fraction de seconde.
st.set_page_config(
    page_title="Stella - Assistant Financier",
    layout="centered"
)

# Affiche un message de chargement pendant la redirection
st.title("🚀 Lancement de l'assistant...")
st.write("Veuillez patienter, redirection en cours vers l'accueil de l'application....")

st.switch_page("pages/1_🏠_Accueil.py")