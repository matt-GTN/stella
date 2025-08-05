# agent/pages/1_🎬 Visualisation de l'agent.py

import streamlit as st
import time
import os
import sys

# Astuce pour importer des modules depuis le répertoire parent (agent/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import generate_trace_animation_frames

# Configuration de la page
st.set_page_config(layout="wide", page_title="Visualisation de l'agent")
st.title("🎬 Visualisation de l'agent")
st.markdown("Visualisez pas à pas le chemin de décision de la dernière conversation avec l'agent.")

# Vérifier si une conversation a déjà eu lieu
if 'last_run_id' not in st.session_state:
    st.info("👋 Pour commencer, veuillez avoir une conversation avec l'agent sur la page principale '👩🏻 Stella, analyste'.")
    st.stop()

# --- Initialisation de l'état de la session pour la visualisation ---
# On va stocker les frames de l'animation pour ne pas les régénérer à chaque fois
if 'animation_frames' not in st.session_state:
    st.session_state.animation_frames = []
# On stocke l'index de l'étape actuelle
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# --- Interface de contrôle ---
st.subheader("Contrôles")
# On utilise st.columns pour organiser les boutons
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 3])

with col1:
    # Bouton pour charger la trace. Il sera désactivé une fois les frames chargées.
    load_button = st.button(
        "Charger la trace de l'exécution", 
        use_container_width=True, 
        type="primary",
        disabled=bool(st.session_state.animation_frames) # Désactivé si les frames sont déjà là
    )

with col2:
    # Bouton "Précédent"
    prev_button = st.button("⬅️", use_container_width=True, disabled=not st.session_state.animation_frames)

with col3:
    # Bouton "Suivant"
    next_button = st.button("➡️", use_container_width=True, disabled=not st.session_state.animation_frames)

with col4:
    # Bouton "Play" pour lancer l'animation automatique
    play_button = st.button("▶️", use_container_width=True, disabled=not st.session_state.animation_frames)

with col5:
    # Slider pour la vitesse, utile pour le mode "Play"
    speed = st.slider(
        "Vitesse (secondes par étape)", 
        min_value=0.25, 
        max_value=3.0, 
        value=1.0,
        step=0.25
    )

# --- Logique de chargement des données ---
if load_button:
    last_run_id = st.session_state.last_run_id
    with st.spinner("Récupération de la trace et génération des images..."):
        frames = generate_trace_animation_frames(last_run_id)
        if not frames:
            st.error("Impossible de récupérer la trace ou de générer les images. Vérifiez les logs du terminal.")
            st.session_state.animation_frames = []
            st.session_state.current_step = 0
        else:
            st.success(f"Trace trouvée ! {len(frames)} étapes sont prêtes à être visualisées.")
            st.session_state.animation_frames = frames
            st.session_state.current_step = 0
    # On rafraîchit la page pour activer les boutons de contrôle
    st.rerun()

# --- Logique de navigation manuelle ---
if prev_button:
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1

if next_button:
    if st.session_state.current_step < len(st.session_state.animation_frames) - 1:
        st.session_state.current_step += 1

# --- Conteneurs pour l'affichage ---
# On les définit ici pour qu'ils existent toujours
description_placeholder = st.empty()
image_placeholder = st.empty()

# --- Affichage de l'étape actuelle ---
if st.session_state.animation_frames:
    total_steps = len(st.session_state.animation_frames)
    current_step_index = st.session_state.current_step
    
    # Récupérer la description et l'image pour l'étape actuelle
    description, image_bytes = st.session_state.animation_frames[current_step_index]

    # Afficher la description et le compteur d'étapes
    description_placeholder.markdown(f"### {description} `(Étape {current_step_index + 1}/{total_steps})`")
    
    # Afficher l'image
    image_placeholder.image(image_bytes, use_container_width=True)

# --- Logique du mode "Play" (animation automatique) ---
if play_button:
    total_steps = len(st.session_state.animation_frames)
    # On commence à l'étape actuelle pour pouvoir reprendre la lecture
    start_step = st.session_state.current_step

    for i in range(start_step, total_steps):
        st.session_state.current_step = i
        
        description, image_bytes = st.session_state.animation_frames[i]
        
        # Mettre à jour les conteneurs
        description_placeholder.markdown(f"### {description} `(Étape {i + 1}/{total_steps})`")
        image_placeholder.image(image_bytes, use_container_width=True)
        
        # Attendre
        time.sleep(speed)
        
    st.success("🎉 Animation terminée !")
    # On remet l'index à la dernière étape après l'animation
    st.session_state.current_step = total_steps - 1