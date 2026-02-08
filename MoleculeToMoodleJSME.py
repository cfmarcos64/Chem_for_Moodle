
# -*- coding: utf-8 -*-
"""
Moodle Question Generator for Skeletal Structures.
Integrated Version for Chem_To_Moodle Suite
@author: Carlos Fernandez Marcos
"""

import streamlit as st
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import time
import uuid
import json
import io
from my_component import jsme_editor

# --- Availability checks ---
try:
    from rdkit import Chem
except ImportError:
    Chem = None

# ==============================================================================
# 1. BILINGUAL TEXTS
# ==============================================================================
TEXTS = {
    "en": {
        "add_to_list": "Add to list",
        "api_error": "Network error while searching for structure.",
        "bulk_note": "Upload .csv or .xlsx with a **'name'** column.",
        "clear_all": "Clear all",
        "column_error": "🚨 File must contain a 'name' column.",
        "custom_question_name": "Question Label (e.g. Aspirin):",
        "download_xml": "Download Moodle XML",
        "error_not_found": "🚨 SMILES or Name not found for '{0}'.",
        "intro": "Enter the name or SMILES of a compound to generate questions where the molecule's name is shown and the student must draw its complete skeletal structure in the JSME editor.",
        "jsme_processing_error": "Error processing JSME response: {0}",
        "json_decode_error": "Error interpreting JSME editor response (invalid JSON format)",
        "molecule_name": "Molecule Name or SMILES:",
        "no_questions": "No questions added yet.",
        "no_valid_smiles_warning": "No valid SMILES received for question {0}",
        "normalized_success": "Question {0} normalized: {1} → {2}",
        "not_normalized": "⚠️ Pending JSME",
        "preview_footer_mol": "➡️ The student must draw the molecular structure in the editor",
        "preview_intro": "The student will see something like this:",
        "preview_question_mol": "Use the JSME editor to draw the molecular structure of **Paracetamol**:",
        "preview_title": "👁️ View sample Moodle question",
        "q_text_template": "Use the JSME editor to draw the molecular structure of: **{0}**",
        "questions_added_subtitle": "Added Questions",
        "searching": "🔍 Searching NCI CIR for: {0}...",
        "section_bulk": "Bulk Upload",
        "section_individual": "Individual Entry",
        "start_bulk": "Process file",
        "start_standardization": "Standardize {0} pending molecule(s)",
        "title": ":material/hub: Structure Question Generator",
        "xml_error": "Error generating XML: {0}"
    },
    "es": {
        "add_to_list": "Añadir a la lista",
        "api_error": "Error de red al buscar la estructura.",
        "bulk_note": "Sube un .csv o .xlsx con columnas **'name'** (API) y **'nombre'** (Moodle).",
        "clear_all": "Borrar todo",
        "column_error": "🚨 El archivo debe tener columnas 'name' y 'nombre'.",
        "custom_question_name": "Etiqueta en Español (ej. Aspirina):",
        "download_xml": "Descargar XML de Moodle",
        "error_not_found": "🚨 No se encontró SMILES o nombre para '{0}'.",
        "intro": "Introduce el nombre o SMILES de un compuesto para generar preguntas en las que se muestra el nombre de la molécula y el alumno debe dibujar su estructura lineoangular completa en el editor JSME.",
        "jsme_processing_error": "Error procesando respuesta JSME: {0}",
        "json_decode_error": "Error al interpretar la respuesta del editor JSME (formato JSON inválido)",
        "molecule_name": "Nombre de molécula en inglés o SMILES:",
        "no_questions": "Aún no hay preguntas añadidas.",
        "no_valid_smiles_warning": "No se recibió SMILES válido para la pregunta {0}",
        "normalized_success": "Pregunta {0} normalizada: {1} → {2}",
        "not_normalized": "⚠️ Pendiente de JSME",
        "preview_footer_mol": "➡️ El alumno verá el lienzo vacío y deberá dibujar la estructura lineoangular.",
        "preview_intro": "El alumno verá algo como esto:",
        "preview_question_mol": "Utiliza el editor JSME para dibujar la estructura molecular del **Paracetamol**:",
        "preview_title": "👁️ Ver ejemplo de pregunta en Moodle",
        "q_text_template": "Utiliza el editor JSME para dibujar la estructura molecular de: **{0}**",
        "questions_added_subtitle": "Preguntas Añadidas",
        "searching": "🔍 Buscando en NCI CIR: {0}...",
        "section_bulk": "Carga Masiva",
        "section_individual": "Entrada Individual",
        "start_bulk": "Procesar archivo",
        "start_standardization": "Estandarizar {0} molécula(s) pendiente(s)",
        "title": ":material/hub: Generador de Preguntas de Formulación Estructural",
        "xml_error": "Error al generar el XML: {0}"
    }
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def canonicalize_smiles(smiles):
    """Restaurada: Usa RDKit para asegurar que el SMILES es válido y consistente."""
    if not Chem or not smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

def get_smiles_from_name(name):
    """Restaurada: Lógica completa de consulta a la API NCI CIR con timeouts."""
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{requests.utils.quote(name)}/smiles"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text.strip().split('\n')[0]
        return None
    except:
        return None

def process_input_to_smiles(user_input):
    """Restaurada: Discrimina entre SMILES directo y búsqueda por nombre."""
    if not user_input: return None
    user_input = user_input.strip()
    
    # If it looks SMILES (contains typical characteres), canonicalize directly
    if any(c in user_input for c in "=#()[]") or len(user_input) > 25:
        return canonicalize_smiles(user_input)
    
    # If not, search by name
    sm = get_smiles_from_name(user_input)
    return canonicalize_smiles(sm) if sm else None

def escape_smiles_for_xml(s):
    """Restaurada: Escapa caracteres que pmatchjme interpreta como especiales."""
    if not s: return ""
    # Double scape characters required by Moodle pmatchjme
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")

def generate_xml_local(questions, lang):
    """Restaurada: Generación profesional de XML con CDATA y Categorías."""
    texts = TEXTS[lang]
    root = ET.Element("quiz")
    
    # Category node to organize the question bank
    cat_node = ET.SubElement(root, "question", type="category")
    cat_text = ET.SubElement(ET.SubElement(cat_node, "category"), "text")
    cat_text.text = f"$course$/Skeletal_Formulas_{lang.upper()}"

    for name, smiles in questions:
        q_node = ET.SubElement(root, "question", type="pmatchjme")
        
        # Question name
        ET.SubElement(ET.SubElement(q_node, "name"), "text").text = name
        
        # Question text with CDATA for safe HTML
        qtext_val = texts["q_text_template"].format(name)
        qtext_node = ET.SubElement(q_node, "questiontext", format="html")
        ET.SubElement(qtext_node, "text").text = f"<![CDATA[<p>{qtext_val}</p>]]>"
        
        # Answer configuration
        ans_node = ET.SubElement(q_node, "answer", fraction="100")
        escaped_sm = escape_smiles_for_xml(smiles)
        ET.SubElement(ans_node, "text").text = f"match({escaped_sm})"
        
        # Technical fields required by Moodle
        ET.SubElement(q_node, "modelanswer").text = smiles
        ET.SubElement(q_node, "generalfeedback", format="html").text = "<text></text>"

    f = io.BytesIO()
    f.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return f.getvalue()

# ==============================================================================
# 3. MAIN RENDER FUNCTION
# ==============================================================================

def render_molecule_app(lang="en"):
    texts = TEXTS[lang]
    st.title(texts["title"])
    st.markdown(texts["intro"])
    
    # --- Visual example (Fischer/Glucose) ---
    with st.expander(texts["preview_title"]):
        st.write(texts["preview_intro"])
        st.info(f"**{texts['preview_question_mol']}**")
        st.write("---")
        st.write(texts["preview_footer_mol"])

    # Session states
    if "m_questions" not in st.session_state: st.session_state.m_questions = []
    if "m_jsme_in" not in st.session_state: st.session_state.m_jsme_in = None
    if "m_curr_req" not in st.session_state: st.session_state.m_curr_req = None

    def start_jsme_loop():
        idx = next((i for i, q in enumerate(st.session_state.m_questions) if q.get('smiles_norm') is None), None)
        if idx is not None:
            req_id = str(uuid.uuid4())
            q = st.session_state.m_questions[idx]
            st.session_state.m_jsme_in = json.dumps({'smiles': q['smiles_raw'], 'id': req_id})
            st.session_state.m_curr_req = {'id': req_id, 'index': idx}
        else:
            st.session_state.m_jsme_in = None
            st.session_state.m_curr_req = None 

# JSME Editor (invisible)
    jsme_out = jsme_editor(smiles_json=st.session_state.m_jsme_in, key="jsme_mol_processor")
    
    if st.session_state.m_jsme_in and jsme_out:
        try:
            res = json.loads(jsme_out)
            if res.get('id') == st.session_state.m_curr_req.get('id'):
                idx = st.session_state.m_curr_req['index']
                st.session_state.m_questions[idx]['smiles_norm'] = res.get('smiles')
                start_jsme_loop()
                st.rerun()
        except Exception as e:
            st.error(f"Error en estandarización: {e}")

    # --- Column Layout ---
    m_col, l_col = st.columns([1, 1])

    with m_col:
        t_ind, t_bulk = st.tabs([texts["section_individual"], texts["section_bulk"]])
        
        with t_ind:
            status_p = st.empty()
            with st.form("mol_form", clear_on_submit=True):
                n_input = st.text_input(texts["molecule_name"])
                n_label = st.text_input(texts["custom_question_name"])
                if st.form_submit_button(texts["add_to_list"], type="primary", icon=":material/add_task:"):
                    if n_input:
                        status_p.info(texts["searching"].format(n_input))
                        s_raw = process_input_to_smiles(n_input)
                        if s_raw:
                            st.session_state.m_questions.append({
                                'name': n_label if n_label else n_input, 
                                'smiles_raw': s_raw, 
                                'smiles_norm': None
                            })
                            status_p.empty()
                            st.rerun()
                        else:
                            status_p.error(texts["error_not_found"].format(n_input))

        with t_bulk:
            st.info(texts["bulk_note"])
            up = st.file_uploader(texts["section_bulk"], type=['csv', 'xlsx'])
            if st.button(texts["start_bulk"], icon=":material/cloud_upload:"):
                if up:
                    df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
                    # Identify which columns are present
                    has_name_en = 'name' in df.columns
                    has_name_es = 'nombre' in df.columns

                    if has_name_en or has_name_es:
                        # Priority: If both exist, use 'name' for the API 
                        # and 'nombre' for Moodle lable if available
                        col_for_api = 'name' if has_name_en else 'nombre'
                        
                        for _, row in df.iterrows():
                            val_api = str(row[col_for_api]).strip()
                            if val_api and val_api != 'nan':
                                s = process_input_to_smiles(val_api)
                                if s:
                                    # For the name in Moodle: 
                                    # 1. Try 'nombre', 2. If not, 'name', 3. If not, the API value
                                    moodle_label = str(row.get('nombre', row.get('name', val_api)))
                                    
                                    st.session_state.m_questions.append({
                                        'name': moodle_label, 
                                        'smiles_raw': s, 
                                        'smiles_norm': None
                                    })
                        st.rerun()
                    else:
                        st.error(texts["column_error"])

    with l_col:
        st.subheader(texts["questions_added_subtitle"])
        qs = st.session_state.m_questions
        pending = [i for i, q in enumerate(qs) if q.get('smiles_norm') is None]
        
        if qs:
            if pending:
                if st.button(texts["start_standardization"].format(len(pending)), type="primary", icon=":material/rocket_launch:", use_container_width=True):
                    start_jsme_loop()
                    st.rerun()
            else:
                try:
                    xml_data = generate_xml_local([(q['name'], q['smiles_norm']) for q in qs], lang)
                    st.download_button(texts["download_xml"], data=xml_data, file_name="moodle_chemistry.xml", type="primary", icon=":material/download:", use_container_width=True)
                except Exception as e:
                    st.error(texts["xml_error"].format(e))
            
            if st.button(texts["clear_all"], icon=":material/delete_sweep:", use_container_width=True):
                st.session_state.m_questions = []
                st.rerun()

            st.divider()
            for i, q in enumerate(qs):
                c1, c2 = st.columns([4, 1])
                with c1:
                    # Determine which SMILES and state icon show
                    is_normalized = q.get('smiles_norm') is not None
                    status_icon = "✅" if is_normalized else texts["not_normalized"]
                    display_smiles = q['smiles_norm'] if is_normalized else q['smiles_raw']

                    st.write(f"**{i+1}. {q['name']}** {status_icon}")
                    st.caption(f"SMILES: `{display_smiles}`")
                    
                with c2:
                    if st.button("🗑️", key=f"del_m_{i}", help = "Eliminar pregunta" if lang == "es" else "Delete question"):
                        st.session_state.m_questions.pop(i)
                        st.rerun()
        else:
            st.info(texts["no_questions"])

# ==============================================================================
# 4. STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Idioma por defecto
    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    # Language selector
    with st.sidebar:
        lang = st.selectbox(
            "Idioma / Language",
            options=["es", "en"],
            format_func=lambda x: "Español" if x == "es" else "English",
            index=0 if st.session_state.lang == "es" else 1,
            key="lang_selector"
        )
        # Update session_state when it changes
        if lang != st.session_state.lang:
            st.session_state.lang = lang
            st.rerun()

    # Title
    st.set_page_config(
        layout="wide",
        page_title=TEXTS[st.session_state.lang]["title"],
        initial_sidebar_state="expanded"   # or "auto"
    )

    # Run app in current language
    render_molecule_app(st.session_state.lang)