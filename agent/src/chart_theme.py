# agent/src/chart_theme.py

# Dictionnaire contenant toutes les préférences graphiques pour les graphiques de Stella.
stella_theme = {

    'colors': [
        '#33FFBD', 
        '#C9B1FF',
        '#FFB81C',
        '#8c564b', 
        '#2ca02c',  
        '#1f77b4',  
        '#e377c2',  
        '#d62728',  
        '#ff7f0e',  
    ],

    # Des couleurs spécifiques pour certaines métriques clés.
    # Utilisé pour le graphique  de synthèse.
    'metric_colors': {
        'roe': '#2ca02c',               # Le ROE est un signe de rentabilité -> Vert
        'debtToEquity': '#d62728',      # La dette est un risque -> Rouge
        'earningsYield': '#1f77b4',     # Le rendement est une info neutre -> Bleu
        'marginProfit': '#9467bd',      # La marge est une info de performance -> Violet
    },
    
    # Le modèle de base pour les graphiques.
    'template': 'plotly_white',
    
    # La police de caractères pour tous les textes du graphique.
    'font': {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': '#333333' # Une couleur de texte sombre mais pas noire pure
    }
}