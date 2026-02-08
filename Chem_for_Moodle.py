"""
Created on Tue Feb  3 01:36:50 2026

@author: Carlos F. Marcos
"""

# -*- coding: utf-8 -*-

import streamlit as st

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Moodle Chemistry Suite",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded" 
)

# --- 2. MODULE IMPORTATION ---
try:
    from Fischer_for_Moodle import render_fischer_app
    from MoleculeToMoodleJSME import render_molecule_app
    from ReactionToMoodleJSME import render_reaction_app
except ImportError as e:
    st.error(f"Error cargando módulos: {e}")

# --- 3. LANGUAGE CENTRAL MANAGEMENT ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"

def toggle_language():
    if st.session_state.lang_toggle:
        st.session_state.lang = "es"
    else:
        st.session_state.lang = "en"
    
# Multilingual texts
ui_texts = {
    "es": {
        "sidebar_title": "🧪 Preguntas de Química $pmatchjme$ para Moodle", 
        "selector_label": "Selecciona una herramienta:", 
        "opt": ["Fórmulas lineoangulares", "Proyecciones de Fischer", "Reacciones Químicas"]
    },
    "en": {
        "sidebar_title": "🧪 $pmatchjme$ Chemistry Questions for Moodle", 
        "selector_label": "Select a tool:", 
        "opt": ["Skeletal Formulas", "Fischer Projections", "Chemical Reactions"]
    }
}

# --- 4. SIDEBAR ---

# Language toggle
with st.sidebar:
    # Styles according to the active language
    # Green (#28a745) for active and grey (#888) for inactive language
    style_es = "color: #28a745; font-weight: bold;" if st.session_state.lang == "es" else "color: #888;"
    style_en = "color: #28a745; font-weight: bold;" if st.session_state.lang == "en" else "color: #888;"
    
    col_txt, col_tgl = st.columns([0.7, 0.3])
    
    with col_txt:
        # Language selection background
        st.markdown(
            f'''
            <div style="
                background-color: rgba(255, 255, 255, 0.4); 
                padding: 5px 10px; 
                border-radius: 15px; 
                text-align: center;
                margin-top: 5px;
                font-size: 0.85rem;
                display: flex;
                justify-content: center;
                gap: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            ">
                <span style="{style_en}">ENGLISH</span>
                <span style="color: #ccc;">|</span>
                <span style="{style_es}">ESPAÑOL</span>
            </div>
            ''', 
            unsafe_allow_html=True
        )

    with col_tgl:
        st.toggle(
            " ", 
            value=(st.session_state.lang == "es"), 
            key="lang_toggle", 
            on_change=toggle_language,
            label_visibility="collapsed" # Hide label to avouid unalignement
        )
    
    st.divider()
    
    # Title
    st.title(ui_texts[st.session_state.lang]["sidebar_title"])
    
    # Tool selector
    opcion = st.radio(
        ui_texts[st.session_state.lang]["selector_label"], 
        ui_texts[st.session_state.lang]["opt"]
    )
    
 # --- Colors dictionary ---
tool_colors = {
    "Fischer": "#e3f2fd",    # Light blue
    "Fórmulas": "#f1f8e9",   # Light green
    "Reacciones": "#fff3e0", # Light orange
    "Skeletal": "#f1f8e9",   # Light green (English)
    "Reactions": "#fff3e0"   # Light orange (English)
}   

# --- 5. ROUTING ---

# Function to change color
def set_sidebar_color(option):
    color = "#ffffff" # White as default
    for key, hex_color in tool_colors.items():
        if key in option:
            color = hex_color
            break
            
    # Inject CSS in sidebar
    st.markdown(f"""
        <style>
            [data-testid="stSidebar"] {{
                background-color: {color};
            }}
        </style>
    """, unsafe_allow_html=True)

# Call function
set_sidebar_color(opcion)

# Initialize app (use indexes [0, 1, 2] to make routing language-independent
opciones_en = ui_texts["en"]["opt"]
opciones_es = ui_texts["es"]["opt"]

if opcion == opciones_en[0] or opcion == opciones_es[0]:
    render_molecule_app(st.session_state.lang)
elif opcion == opciones_en[1] or opcion == opciones_es[1]:
    render_fischer_app(st.session_state.lang)
elif opcion == opciones_en[2] or opcion == opciones_es[2]:
    render_reaction_app(st.session_state.lang)