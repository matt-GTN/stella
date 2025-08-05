# Page Modélisation 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import warnings

warnings.filterwarnings('ignore')


st.set_page_config(page_title="Modélisation du Risque", layout="wide")

# --- Définition des paramètres du meilleur modèle ---
OPTIMAL_PARAMS = {
    'n_estimators': 134,
    'max_depth': 10,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'criterion': 'entropy',
}

# --- Initialisation de st.session_state pour les hyperparamètres ---
# Cela garantit que les valeurs des sliders persistent et peuvent être réinitialisées.
if 'hyperparams' not in st.session_state:
    st.session_state.hyperparams = OPTIMAL_PARAMS.copy()

if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# --- Fonction callback pour le bouton reset ---
def reset_to_optimal():
    """Réinitialise les hyperparamètres dans session_state aux valeurs optimales."""
    st.session_state.hyperparams = OPTIMAL_PARAMS.copy()

# --- Section header ---
st.title("🧠 Modélisation Interactive : De la Prédiction au Filtrage de Risque")
st.markdown("""
Cette page interactive vous emmène au cœur de la partie Modélisation du projet. L'objectif initial : prédire si une action du NASDAQ 100 allait **surperformer le marché (Classe 1)** ou **sous-performer (Classe 0)** en se basant uniquement sur ses données financières fondamentales.

Cependant, nos recherches ont révélé une vérité nuancée mais néanmoins intéressante : s'il est difficile de prédire les "gagnants" avec une certitude absolue, notre modèle s'est avéré **fiable pour identifier les "perdants" potentiels**.

Nous avons donc réorienté notre stratégie. Cet outil n'est pas un preneur de décision, mais un **système de gestion des risques**. Il vous permet de :
- **Explorer** comment les hyperparamètres d'un `RandomForestClassifier` influencent sa capacité à détecter les risques.
- **Comprendre** quelles caractéristiques financières (croissance, rentabilité, endettement) sont les plus déterminantes.
- **Découvrir** comment, en se concentrant sur les prédictions à haute confiance, le modèle devient un filtre de risque très précis.
""")
st.info("Ajustez les paramètres, entraînez le modèle, ou cliquez sur 'Réinitialiser' pour revenir à notre configuration la plus performante.")

# --- Loading de la donnée ---
DATA_PATH = 'notebooks/csv/N100_fundamentals_v3.csv'

@st.cache_data
def load_and_prep_data(path):
    # (Le reste de la fonction est inchangé)
    if not os.path.exists(path) and os.path.exists(os.path.join(os.path.dirname(__file__), '..', path)):
        path = os.path.join(os.path.dirname(__file__), '..', path)
    if not os.path.exists(path):
        st.error(f"Erreur: Le fichier de données est introuvable : `{path}`")
        return None, None, None, None
    df = pd.read_csv(path)
    df = df.sort_values(by='date')
    df['index'] = df.symbol + '_' + df.calendarYear.astype('string')
    df = df.set_index('index')
    df_final = df.dropna()
    if 'netIncomePerShare' in df_final.columns and 'shareValue' in df_final.columns:
        df_final['earningsYield'] = df_final['netIncomePerShare'] / df_final['shareValue']
    columns_to_drop = [
        'return', 'date_NY', 'date', 'benchmark', 'symbol', 'calendarYear', 'shareValue', 'peRatio_YoY_Growth',
        'peRatio', 'shareValue_YoY_Growth', 'marketCap_YoY_Growth', 'roe_YoY_Growth', 'roic_YoY_Growth',
        'netIncomePerShare_YoY_Growth', 'debtToEquity_YoY_Growth', 'netIncomePerShare', 'marginProfit_YoY_Growth'
    ]
    df_final = df_final.drop(columns=[col for col in columns_to_drop if col in df_final.columns], errors='ignore')
    if 'target' not in df_final.columns:
        st.error("Erreur: La colonne 'target' est manquante.")
        return None, None, None, None
    condition = df_final.index.str.contains('2023')
    X_test = df_final[condition]
    X_train = df_final[~condition]
    y_test = X_test.target
    y_train = X_train.target
    X_train = X_train.drop('target', axis=1)
    X_test = X_test.drop('target', axis=1)
    return X_train, y_train, X_test, y_test

# --- Fonctions helper ---
def create_plotly_confusion_matrix(cm, title, colorscale):
    labels = ['Classe 0 (Sous-perf.)', 'Classe 1 (Sur-perf.)']
    fig = px.imshow(cm, labels=dict(x="Prédiction", y="Vraie Valeur", color="Nombre"), x=labels, y=labels,
                    text_auto=True, color_continuous_scale=colorscale, title=title)
    fig.update_layout(xaxis_title="Classe Prédite", yaxis_title="Classe Réelle", yaxis={'autorange': 'reversed'})
    return fig

@st.cache_data
def train_and_evaluate(_X_train, _y_train, _X_test, params):
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(_X_train, _y_train)
    return model, model.predict(_X_test), model.predict_proba(_X_test)

def get_shap_explanation(_model, _data_to_explain):
    explainer = shap.TreeExplainer(_model)
    return explainer(_data_to_explain)

# --- Logique principale de la page ---
X_train, y_train, X_test, y_test = load_and_prep_data(DATA_PATH)
if X_train is None:
    st.stop()

# Section pour les hyperparamètres
st.header("⚙️ Configuration des Hyperparamètres")
st.info("Ajustez les hyperparamètres pour voir leur impact sur la performance. Un modèle plus complexe est-il toujours meilleur ?")

# Le bouton de réinitialisation avec compteur
if st.button("🔄 Réinitialiser aux Paramètres Optimaux"):
    st.session_state.hyperparams = OPTIMAL_PARAMS.copy()
    st.session_state.reset_counter += 1  # Incrémente le compteur pour forcer la recréation des widgets
    st.rerun()

with st.form("hyperparameter_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Structure de la Forêt")
        # Utilisation du reset_counter dans les keys pour forcer la recréation
        n_estimators = st.slider(
            'Nombre d\'arbres (n_estimators)', 
            10, 500, 
            value=st.session_state.hyperparams['n_estimators'],
            key=f'slider_n_estimators_{st.session_state.reset_counter}',
            help="Plus d'arbres réduit le surapprentissage, mais augmente le temps de calcul."
        )
        max_depth = st.slider(
            'Profondeur maximale (max_depth)', 
            3, 30, 
            value=st.session_state.hyperparams['max_depth'],
            key=f'slider_max_depth_{st.session_state.reset_counter}',
            help="Contrôle la complexité de chaque arbre. Une profondeur trop élevée peut mener au surapprentissage."
        )
    with col2:
        st.subheader("Conditions de Division")
        min_samples_leaf = st.slider(
            'Échantillons min. par feuille', 
            1, 20, 
            value=st.session_state.hyperparams['min_samples_leaf'],
            key=f'slider_min_samples_leaf_{st.session_state.reset_counter}',
            help="Exige un nombre minimum d'échantillons dans une feuille, lissant ainsi le modèle."
        )
        max_features = st.select_slider(
            'Caractéristiques max.', 
            ['sqrt', 'log2', None], 
            value=st.session_state.hyperparams['max_features'],
            key=f'slider_max_features_{st.session_state.reset_counter}',
            help="Nombre de caractéristiques à considérer pour chaque division."
        )
        
        criterion_options = ['gini', 'entropy']
        criterion = st.selectbox(
            'Critère de division', 
            criterion_options, 
            index=criterion_options.index(st.session_state.hyperparams['criterion']),
            key=f'slider_criterion_{st.session_state.reset_counter}'
        )

    submitted = st.form_submit_button("🚀 Entraîner le Modèle")

# Mise à jour du session_state quand le formulaire est soumis
if submitted:
    # Mise à jour des hyperparamètres dans le session_state
    st.session_state.hyperparams = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'criterion': criterion,
    }
    
    # Entraînement du modèle
    with st.spinner("Entraînement du modèle en cours..."):
        st.session_state.model, st.session_state.predictions, st.session_state.probabilities = train_and_evaluate(
            X_train, y_train, X_test, st.session_state.hyperparams
        )
    st.session_state.model_trained = True

st.divider()

if 'model_trained' not in st.session_state:
    st.info("Veuillez cliquer sur 'Entraîner le Modèle' pour commencer l'analyse.")
    st.stop()

# --- Performance Globale ---
st.header("📊 Résultats Globaux sur l'Ensemble de Test (Année 2023)")
st.info("Analysez la performance globale. Observez la différence de précision et de rappel entre la **Classe 0 (Sous-performance)** et la **Classe 1 (Surperformance)**. Le modèle est-il plus doué pour l'une que pour l'autre ?")
with st.container(border=True):
  res_col1, res_col2 = st.columns([1, 1])
  with res_col1:
      st.subheader("Rapport de Classification")
      accuracy = accuracy_score(y_test, st.session_state.predictions)
      st.metric("Précision (Accuracy)", f"{accuracy:.2%}")
      st.code(classification_report(y_test, st.session_state.predictions, target_names=['Classe 0 (Sous-perf.)', 'Classe 1 (Sur-perf.)']))
  with res_col2:
      st.subheader("Matrice de Confusion Générale")
      cm = confusion_matrix(y_test, st.session_state.predictions, labels=[0, 1])
      fig_cm = create_plotly_confusion_matrix(cm, "Matrice de Confusion Générale", "Blues")
      st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

st.header("👑 Importance des Caractéristiques : L'ADN d'une Décision")
st.info("Quels sont les indicateurs financiers les plus influents ? Le modèle a appris à raisonner comme un analyste, en se concentrant sur la croissance (`revenuePerShare_YoY_Growth`), la rentabilité (`roic`) et la structure financière (`debtToEquity`).")
with st.container(border=True):
  feature_importances = pd.Series(st.session_state.model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
  fig_imp = px.bar(feature_importances.head(15), orientation='h', title="Top 15 des Caractéristiques les plus Importantes", labels={'value': 'Importance (Gini)', 'index': 'Caractéristique'})
  fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
  st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# --- Analyse Haute-Confiance ---

st.header("🎯 Le Cœur de la Stratégie : Le Filtrage par la Confiance")
st.info("""
C'est ici que la valeur du modèle se révèle. Au lieu de considérer toutes les prédictions, nous ne gardons que celles où le modèle est le plus **sûr de lui**.
En augmentant le seuil de confiance, nous passons d'un modèle de prédiction générale à un **filtre de risque de haute précision**. Observez comment la précision sur les prédictions restantes (notamment pour la **Classe 0**) augmente drastiquement.
""")
with st.container(border=True):
    confidence_threshold = st.slider("Seuil de confiance pour l'analyse", 0.5, 1.0, 0.7, 0.01, help="Filtre pour n'analyser que les prédictions où la probabilité prédite est supérieure à ce seuil.")

    df_results = X_test.copy()
    df_results['true_label'] = y_test
    df_results['prediction'] = st.session_state.predictions
    df_results['confidence'] = np.max(st.session_state.probabilities, axis=1)
    df_results['is_correct'] = (df_results['prediction'] == df_results['true_label']).astype(int)
    high_confidence_df = df_results[df_results['confidence'] > confidence_threshold]
    hc_col1, hc_col2, hc_col3 = st.columns(3)
    total_hc, correct_hc = len(high_confidence_df), high_confidence_df['is_correct'].sum()
    hc_col1.metric("Prédictions à Haute Confiance", f"{total_hc}")
    hc_col2.metric("Correctes", f"{correct_hc} ({correct_hc/total_hc:.1%})" if total_hc > 0 else "0")
    hc_col3.metric("Incorrectes", f"{total_hc - correct_hc} ({(total_hc - correct_hc)/total_hc:.1%})" if total_hc > 0 else "0")

st.divider()

if 'high_confidence_df' in locals() and not high_confidence_df.empty:
    st.subheader(f"Matrice de Confusion (Confiance > {confidence_threshold:.0%})")
    st.info("Notez la forte réduction des erreurs, en particulier des Faux Positifs (prédire une sous-performance qui n'a pas lieu).")
    with st.container(border=True):
        cm_hc = confusion_matrix(high_confidence_df['true_label'], high_confidence_df['prediction'], labels=[0, 1])
        st.plotly_chart(create_plotly_confusion_matrix(cm_hc, f'Matrice de Confusion (Confiance > {confidence_threshold:.0%})', "Greens"), use_container_width=True)
st.divider()

# --- Analyse SHAP ---
st.header("🕵️ Analyse SHAP : Comprendre l'Archétype de l'Entreprise à Risque")
st.markdown("""
Même un bon modèle fait des erreurs. L'analyse SHAP nous permet de les disséquer pour comprendre **pourquoi** le modèle s'est trompé sur les cas les plus difficiles (les erreurs à haute confiance). 
Cela nous aide à définir l'**archétype de l'entreprise à risque** que le modèle a appris à identifier : une combinaison de croissance stagnante, de faible rentabilité et d'une structure financière fragile.
""")

high_confidence_incorrect_df = high_confidence_df[high_confidence_df['is_correct'] == 0] if 'high_confidence_df' in locals() else pd.DataFrame()

if not high_confidence_incorrect_df.empty:
    st.warning(f"**{len(high_confidence_incorrect_df)}** erreur(s) trouvée(s) avec une confiance > {confidence_threshold:.0%}. Analyse en cours...")
    X_to_explain = X_test.loc[high_confidence_incorrect_df.index]
    
    # Création d'une clé de cache unique pour les valeurs SHAP
    error_indices_sorted = sorted(high_confidence_incorrect_df.index.astype(str))
    cache_key = f"shap_{confidence_threshold}_{hash(tuple(error_indices_sorted))}"
    
    # Check si les valeurs SHAP sont déjà en cache
    if (not hasattr(st.session_state, 'current_shap_key') or 
        st.session_state.current_shap_key != cache_key):
        
        with st.spinner("Calcul des valeurs SHAP pour les erreurs..."):
            st.session_state.current_shap_explanation = get_shap_explanation(st.session_state.model, X_to_explain)
            st.session_state.current_shap_key = cache_key
            st.session_state.current_x_indices = list(X_to_explain.index)
    
    shap_explanation = st.session_state.current_shap_explanation
    
    # Vérifie si les indices de X_to_explain correspondent à ceux déjà en cache
    if (hasattr(st.session_state, 'current_x_indices') and 
        st.session_state.current_x_indices != list(X_to_explain.index)):
        # Force le recalcul des valeurs SHAP si les indices ne correspondent pas
        with st.spinner("Recalcul des valeurs SHAP..."):
            st.session_state.current_shap_explanation = get_shap_explanation(st.session_state.model, X_to_explain)
            st.session_state.current_shap_key = cache_key
            st.session_state.current_x_indices = list(X_to_explain.index)
        shap_explanation = st.session_state.current_shap_explanation
    
    st.subheader("Résumé SHAP des Erreurs")
    st.info("Ce graphique montre les caractéristiques qui ont le plus contribué aux **erreurs** du modèle sur le sous-ensemble filtré. Quelles sont les caractéristiques qui 'trompent' le plus notre modèle ?")
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_explanation[:,:,1], show=False, plot_type="dot")
    st.pyplot(fig_summary)
    plt.clf()

    st.subheader("Analyse Détaillée d'une Erreur Spécifique")
    st.info("Disséquons une erreur. Le graphique 'force plot' montre les forces (en rouge) qui ont poussé la prédiction vers la classe incorrecte, et les forces (en bleu) qui poussaient dans la bonne direction.")
    error_choice = st.selectbox(
        "Choisissez une erreur à inspecter en détail :",
        options=X_to_explain.index,
        format_func=lambda idx: f"{idx} (Prédit: {int(high_confidence_incorrect_df.loc[idx, 'prediction'])}, Réel: {int(high_confidence_incorrect_df.loc[idx, 'true_label'])})",
        key=f"error_select_{cache_key}"
    )
    if error_choice:
        instance_info = high_confidence_incorrect_df.loc[error_choice]
        st.write(f"**Vraie Classe :** `{int(instance_info['true_label'])}` | **Classe Prédite :** `{int(instance_info['prediction'])}` | **Confiance :** `{instance_info['confidence']:.2%}`")
        
        try:
            error_position = list(X_to_explain.index).index(error_choice)
            
            if error_position >= shap_explanation.shape[0]:
                st.error(f"Erreur critique: Décalage entre les données. Recalcul forcé...")
                with st.spinner("Recalcul complet des valeurs SHAP..."):
                    st.session_state.current_shap_explanation = get_shap_explanation(st.session_state.model, X_to_explain)
                    st.session_state.current_shap_key = cache_key
                    st.session_state.current_x_indices = list(X_to_explain.index)
                shap_explanation = st.session_state.current_shap_explanation
                
                if error_position < shap_explanation.shape[0]:
                    single_instance_explanation = shap_explanation[error_position, :, 1]
                    plt.figure(figsize=(10, 3))
                    shap.force_plot(single_instance_explanation, matplotlib=True, show=False, text_rotation=15)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.error("Impossible de résoudre le problème de décalage des données.")
            else:
                single_instance_explanation = shap_explanation[error_position, :, 1]
                
                plt.figure(figsize=(10, 3))
                shap.force_plot(single_instance_explanation, matplotlib=True, show=False, text_rotation=15)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.clf()
                
        except (ValueError, IndexError) as e:
            st.error(f"Erreur lors de l'analyse SHAP pour cette instance: {e}")
else:
    st.success(f"✅ Excellent ! Aucune erreur avec une confiance > {confidence_threshold:.0%} n'a été trouvée avec la configuration actuelle.")