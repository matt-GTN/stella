# agent/app.py (Le nouveau fichier d'accueil)

import streamlit as st
from src.pdf_research import initialize_research_handler



# Configure la page pour qu'elle ait un titre, mais elle ne sera visible qu'une fraction de seconde.
st.set_page_config(
    page_title="Stella - Assistant Financier",
    layout="centered"
)

# Affiche un message de chargement pendant la redirection
st.title("🚀 Lancement de l'assistant...")
st.write("Veuillez patienter, redirection en cours vers l'accueil de l'application....")

# Initialize the research handler when the app starts
if 'research_initialized' not in st.session_state:
    with st.spinner('Chargement du document de recherche...'):
        initialize_research_handler()
        st.session_state.research_initialized = True
        st.switch_page("pages/1_🏠_Accueil.py")
else:
    st.session_state.research_initialized = True
    st.switch_page("pages/1_🏠_Accueil.py")
