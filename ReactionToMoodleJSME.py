# -*- coding: utf-8 -*-
"""
Moodle Reaction Question Generator
Integrated Version for Chem_To_Moodle Suite
@author: Carlos Fernandez Marcos
"""
import streamlit as st
import xml.etree.ElementTree as ET
import requests
import io
import base64
from PIL import Image
import numpy as np
import pandas as pd
from my_component import jsme_editor
import json
import uuid

# ===================================================================
# 1. MODULE AVAILABILITY CHECKS
# ===================================================================

RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except Exception as e:
    # Esto nos dirá en la consola de Streamlit qué está fallando realmente
    print(f"DEBUG: Error importing RDKit: {e}")
    Chem = None

PANDAS_AVAILABLE = True
NUMPY_AVAILABLE = True if 'np' in globals() else False

# ========================================================================
# 2. MULTILINGUAL TEXTS (ALPHABETICALLY SORTED KEYS)
# ========================================================================

TEXTS = {
    "en": {
        "add_reaction_button": "Add Reaction",
        "add_to_agents": "Add over the Arrow",
        "add_to_products": "Add to Products",
        "add_to_reactants": "Add to Reactants",
        "added_questions_title": "Added Questions",
        "agents_label": "Agents over the Arrow (SMILES or [Text]):",
        "apply_changes": "✅ Apply Changes",
        "bulk_error_empty_missing": "could not be processed: the Missing_Name column is empty.",
        "bulk_error_image_failed": "could not be processed: Error generating the reaction image.",
        "bulk_error_missing_not_in_reaction": "could not be processed: The missing molecule was found, but it is not part of the reactants/products in this row.",
        "bulk_error_no_rp": "could not be processed: No reactants (R*) or products (P*) were found.",
        "bulk_error_read": "Error reading file.",
        "bulk_error_row": "Row {} skipped: {}",
        "bulk_error_smiles_not_found": "could not be processed because the SMILES for '{}' was not found.",
        "bulk_info": "### Bulk Upload\nUpload Excel/CSV with columns:\n"
                     "`Question_Name`, `R1`, `R2`, ..., `P1`, `P2`, ..., `A1`, `A2`, ..., `Missing_Molecule`, `Correct_Feedback`, `Incorrect_Feedback`.\n"
                     "Use R for reactants, P for products and A for reagents/conditions over the arrow. If A is text, write it between square brackets '[ ]'.",
        "bulk_row_prefix": "Row {}",
        "bulk_summary_success": "Completed: {} added, {} failed.",
        "bulk_summary_title": "❌ Skipped Rows ({} Failed)",
        "change_language": "Cambiar idioma a Español",
        "clear_all_button": "Clear All",
        "continue_standardization": "Continue standardization ({0} pending)",
        "correct_feedback_label": "Correct Feedback:",
        "download_xml_button": "Download XML",
        "img_warning": "The reaction image couldn't be generated",
        "incorrect_feedback_label": "Incorrect Feedback (Optional):",
        "intro": "Set up a chemical reaction by hiding one of its components (reactant, product, or agents). The student must identify the missing part and draw it in the JSME editor to complete the sequence.",
        "is_normalized": "✅ Normalized",
        "is_pending": "⚠️ Pending normalization",
        "jsme_error": "None normalized",
        "jsme_partial": "Partial: **{} of {}** normalized",
        "jsme_processing_error": "Error processing JSME response: {0}",
        "jsme_success": "SUCCESS: **{} of {}** normalized correctly",
        "json_decode_error": "Error interpreting JSME editor response (invalid JSON format)",
        "missing_mol_warning": "The selected missing molecule is not in the reaction",
        "name_error": "Could not find SMILES for '{}'.",
        "name_input_label": "Compound name:",
        "name_warning": "Please, introduce a name for the question",
        "new_question": "New Question",
        "no_questions": "No questions added yet.",
        "no_valid_smiles_warning": "No valid SMILES received for question {0}",
        "normalize_button": "Normalize {0} pending reaction(s)",
        "normalize_first": "First normalize with JSME",
        "normalized_success": "Question {0} normalized: {1} → {2}",
        "preview_intro": "The student will see something like this:",
        "preview_footer_react": "➡️ The student must identify the missing component and draw it in the molecular editor.",
        "preview_question_react": "Draw the missing molecule in the reaction:",
        "preview_title": "👁️ View sample Moodle question",
        "process_bulk_button": "Process File",
        "processing_bulk": "Processing {} rows...",
        "products_label": "Products (SMILES, comma-separated):",
        "question_text": "Draw the missing molecule in the reaction:",
        "reactants_label": "Reactants (SMILES, comma-separated):",
        "reaction_added": "Reaction added: {}",
        "reaction_name_label": "Reaction Name:",
        "search_button": "Search SMILES",
        "search_title": "Search by Name (NCI CIR)",
        "select_missing": "Select missing molecule:",
        "select_molecule_warning": "Select a missing molecule from the list.",
        "smiles_empty_error": "Fields cannot be empty.",
        "smiles_found": "SMILES found: {}",
        "smiles_invalid_error": "Invalid SMILES: '{}'.",
        "tab_bulk": "Bulk Upload",
        "tab_manual": "Manual Entry",
        "title": ":material/science: Moodle Reaction Question Generator",
        "upload_file_label": "Select Excel/CSV file:",
        "xml_error": "Error generating XML: {}"
    },
    "es": {
        "add_reaction_button": "Añadir Reacción",
        "add_to_agents": "Añadir sobre la Flecha",
        "add_to_products": "Añadir a Productos",
        "add_to_reactants": "Añadir a Reactivos",
        "added_questions_title": "Preguntas Añadidas",
        "agents_label": "Agentes sobre la Flecha (SMILES o [Texto]):",
        "apply_changes": "✅ Aplicar Cambios",
        "bulk_error_empty_missing": "no se ha podido procesar: la columna Missing_Name está vacía.",
        "bulk_error_image_failed": "no se ha podido procesar: Error al generar la imagen de reacción.",
        "bulk_error_missing_not_in_reaction": "no se ha podido procesar: La molécula faltante se encontró, pero no forma parte de los reactivos/productos de esa fila.",
        "bulk_error_no_rp": "no se ha podido procesar: No se encontraron reactivos (R*) ni productos (P*).",
        "bulk_error_read": "Error al leer el archivo.",
        "bulk_error_row": "Fila {} omitida: {}",
        "bulk_error_smiles_not_found": "no se ha podido procesar porque no se ha encontrado el SMILES de '{}'.",
        "bulk_info": "### Carga Masiva\nSube un archivo Excel/CSV con columnas:\n"
                     "`Nombre_Pregunta`, `R1`, `R2`, ..., `P1`, `P2`, ...,`A1`, `A2`, ...,`Molécula_Faltante`, `Retroalimentación_Correcta`, `Retroalimentación_Incorrecta`.\n"
                     "Usar R para reactivos, P para productos y A para reactivos/condiciones sobre la fecha. Si A es texto, debe ir entre corchetes '[ ]'.",
        "bulk_row_prefix": "La fila {}",
        "bulk_summary_success": "Completado: {} añadidas, {} fallaron.",
        "bulk_summary_title": "❌ Filas Omitidas ({} Fallaron)",
        "change_language": "Change language to English",
        "clear_all_button": "Borrar Todas",
        "continue_standardization": "Continuar estandarización ({0} pendiente(s))",
        "correct_feedback_label": "Retroalimentación Correcta:",
        "download_xml_button": "Descargar XML",
        "img_warning": "No se pudo generar la imagen de la reacción.",
        "incorrect_feedback_label": "Retroalimentación Incorrecta (Opcional):",
        "intro": "Configura una reacción química ocultando uno de sus componentes (reactivo, producto o agentes). El alumno deberá identificar la parte faltante y dibujarla en el editor JSME para completar la secuencia.",
        "is_normalized": "✅ Normalizada",
        "is_pending": "⚠️ Pendiente de normalizar",
        "jsme_error": "Ninguna normalizada",
        "jsme_partial": "Parcial: **{} de {}** normalizadas",
        "jsme_processing_error": "Error procesando respuesta JSME: {0}",
        "jsme_success": "ÉXITO: **{} de {}** normalizadas correctamente",
        "json_decode_error": "Error al interpretar la respuesta del editor JSME (formato JSON inválido)",
        "missing_mol_warning": "La molécula seleccionada como faltante no aparece en reactivos ni productos.",
        "name_error": "No se encontró SMILES para '{}'.",
        "name_input_label": "Nombre del compuesto:",
        "name_warning": "Por favor, introduce un nombre para la pregunta.",
        "new_question": "Nueva Pregunta",
        "no_questions": "Aún no hay preguntas añadidas.",
        "no_valid_smiles_warning": "No se recibió SMILES válido para la pregunta {0}",
        "normalize_button": "Normalizar {0} reaccion(es) pendiente(s)",
        "normalize_first": "Primero normaliza con JSME",
        "normalized_success": "Pregunta {0} normalizada: {1} → {2}",
        "preview_intro": "El alumno verá algo como esto:",
        "preview_footer_react": "➡️ El alumno deberá identificar la parte faltante y dibujarla en el editor.",
        "preview_question_react": "Dibuja la molécula que falta en la siguiente reacción:",
        "preview_title": "👁️ Ver ejemplo de pregunta en Moodle",
        "process_bulk_button": "Procesar Archivo",
        "processing_bulk": "Procesando {} filas...",
        "products_label": "Productos (SMILES, separados por comas):",
        "question_text": "Dibuja la molécula que falta en la siguiente reacción:",
        "reactants_label": "Reactivos (SMILES, separados por comas):",
        "reaction_added": "Reacción añadida: {}",
        "reaction_name_label": "Nombre de la Reacción:",
        "search_button": "Buscar SMILES",
        "search_title": "Búsqueda por Nombre (NCI CIR)",
        "select_missing": "Selecciona la molécula faltante:",
        "select_molecule_warning": "Selecciona una molécula faltante de la lista.",
        "smiles_empty_error": "Los campos no pueden estar vacíos.",
        "smiles_found": "SMILES encontrado: {}",
        "smiles_invalid_error": "SMILES inválido: '{}'.",
        "tab_bulk": "Carga Masiva",
        "tab_manual": "Entrada Manual",
        "title": ":material/science: Generador de Preguntas de Reacción para Moodle",
        "upload_file_label": "Selecciona archivo Excel/CSV:",
        "xml_error": "Error al generar XML: {}"
    }
}

# ========================================================================
# 3. HELPER FUNCTIONS
# ========================================================================

def draw_mol_consistent(mol, fixed_bond_length: float = 25.0, padding: int = 10) -> Image.Image:
    """Draw molecule with consistent bond lengths and cropping."""
    if not mol or not rdMolDraw2D or not NUMPY_AVAILABLE:
        return Image.new('RGB', (50, 50), (255, 255, 255))
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = max(1, int(fixed_bond_length * 0.1))
    opts.fixedBondLength = fixed_bond_length
    opts.fixedFontSize = int(fixed_bond_length * 0.55)
    opts.padding = 0.1
    opts.addStereoAnnotation = False
    opts.clearBackground = True
    large_size = 2048
    drawer = rdMolDraw2D.MolDraw2DCairo(large_size, large_size)
    drawer.SetDrawOptions(opts)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = io.BytesIO(drawer.GetDrawingText())
    img = Image.open(bio).convert('RGB')
    img_array = np.array(img)
    mask = np.any(img_array != [255, 255, 255], axis=-1)
    y_coords, x_coords = np.nonzero(mask)
    if len(y_coords) == 0:
        return Image.new('RGB', (50, 50), (255, 255, 255))
    y0 = max(0, y_coords.min() - padding)
    x0 = max(0, x_coords.min() - padding)
    y1 = min(img.height, y_coords.max() + 1 + padding)
    x1 = min(img.width, x_coords.max() + 1 + padding)
    return img.crop((x0, y0, x1, y1))

def parse_reaction_smiles(reaction_smiles: str) -> tuple:
    """Parse reaction SMILES string in format: Reactants>Agents>Products."""
    parts = reaction_smiles.split('>')
    if len(parts) != 3:
        return [reaction_smiles.split('>>')[0]], [], [reaction_smiles.split('>>')[1]] if '>>' in reaction_smiles else ([],[],[])
    reactants = [s.strip() for s in parts[0].split('.') if s.strip()]
    agents = [s.strip() for s in parts[1].split('.') if s.strip()]
    products = [s.strip() for s in parts[2].split('.') if s.strip()]
    return reactants, agents, products

def normalize_text(text):
    """Normalize special characters for font compatibility."""
    replacements = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'º': 'deg'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def draw_reaction(r_str, p_str, a_str=""):
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    
    rxn_smarts = f"{r_str}>{a_str}>{p_str}"
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        # Cairo es lo único "extra" que dejamos porque es vital en la nube
        d2d = rdMolDraw2D.MolDraw2DCairo(800, 300)
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        return base64.b64encode(d2d.GetDrawingText()).decode('utf-8')
    except:
        return None

def generate_reaction_image(reaction_smiles: str, missing_smiles: str):
    """Replace missing molecule with placeholder and draw reaction."""
    # --- CAMBIO 3: Verificación dinámica de disponibilidad ---
    try:
        from rdkit import Chem
    except ImportError:
        return None

    # El resto de la lógica se mantiene igual para no afectar la funcionalidad
    reactants, agents, products = parse_reaction_smiles(reaction_smiles)
    placeholder = "[*:1]"
    
    def replace_missing(mol_list, missing):
        try:
            m_can = Chem.MolToSmiles(Chem.MolFromSmiles(missing), canonical=True)
        except:
            m_can = missing
        new_list, found = [], False
        for s in mol_list:
            if s.startswith("[") and s.endswith("]") and not ":" in s:
                new_list.append(s)
                continue
            try:
                s_can = Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
                if s_can == m_can and not found:
                    new_list.append(placeholder); found = True 
                else: new_list.append(s)
            except:
                if s == missing and not found:
                    new_list.append(placeholder); found = True
                else: new_list.append(s)
        return new_list, found

    new_reactants, found_in_r = replace_missing(reactants, missing_smiles)
    if found_in_r: 
        new_products = products
    else: 
        new_products, _ = replace_missing(products, missing_smiles)
        new_reactants = reactants
        
    return draw_reaction(".".join(new_reactants), ".".join(new_products), ".".join(agents))

def generate_xml(questions, lang: str) -> bytes:
    """Generate Moodle XML for pmatchjme questions."""
    quiz = ET.Element('quiz')
    instruction_text = TEXTS[lang].get("question_text", "Draw the missing molecule:")
    prompt = f'<p>{instruction_text}</p>'
    for q in questions:
        question = ET.SubElement(quiz, 'question', type='pmatchjme')
        name_el = ET.SubElement(question, 'name')
        ET.SubElement(name_el, 'text').text = q['name']
        qtext = ET.SubElement(question, 'questiontext', format='html')
        ET.SubElement(qtext, 'text').text = f'{prompt}<img src="data:image/png;base64,{q["img_base64"]}" alt="Reaction"/>'
        answer = ET.SubElement(question, 'answer', fraction='100', format='moodle_auto_format')
        escaped_smiles = q['missing_smiles'].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")
        ET.SubElement(answer, 'text').text = f"match({escaped_smiles})"
        ET.SubElement(question, 'modelanswer').text = q['missing_smiles']
        if q['correct_feedback']:
            fb_text = ET.SubElement(ET.SubElement(answer, 'feedback', format='html'), 'text')
            fb_text.text = f'<p>{q["correct_feedback"]}</p>'
        if q['incorrect_feedback']:
            ans_inc = ET.SubElement(question, 'answer', fraction='0', format='moodle_auto_format')
            ET.SubElement(ans_inc, 'text').text = "*"
            fb_inc_text = ET.SubElement(ET.SubElement(ans_inc, 'feedback', format='html'), 'text')
            fb_inc_text.text = f'<p>{q["incorrect_feedback"]}</p>'
            ET.SubElement(ans_inc, 'atomcount').text = "0"
    xml_str = ET.tostring(quiz, encoding='utf-8', method='xml').decode('utf-8')
    import xml.dom.minidom
    return xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ").encode('utf-8')

def get_smiles_from_name(name: str) -> str | None:
    """Fetch SMILES from NCI CIR API."""
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        smiles = res.text.strip().split('\n')[0]
        if "ERROR" in smiles or (RDKIT_AVAILABLE and Chem.MolFromSmiles(smiles) is None):
            return None
        return smiles
    except: return None

def name_to_smiles(name: str) -> str | None:
    """Convert string to SMILES if not already valid."""
    if not name or pd.isna(name): return None
    name = str(name).strip()
    if RDKIT_AVAILABLE and Chem.MolFromSmiles(name) is not None: return name
    return get_smiles_from_name(name)

def build_reaction_smiles(reactants: list, agents: list, products: list) -> str:
    """Combine components into reaction SMILES."""
    return f"{'.'.join(reactants)}>{'.'.join(agents)}>{'.'.join(products)}"

def start_jsme_normalization_loop():
    pending = [
        i for i, q in enumerate(st.session_state.reaction_questions)
        if not q.get('normalized', False)
    ]
    if pending:
        # Tomamos la primera pendiente
        idx = pending[0]
        req_id = str(uuid.uuid4())
        q = st.session_state.reaction_questions[idx]
        payload = {
            'smiles': q['missing_smiles'],
            'id': req_id
        }
        st.session_state.r_jsme_in = json.dumps(payload)
        st.session_state.r_curr_req = {'id': req_id, 'index': idx}
    else:
        st.session_state.r_jsme_in = None
        st.session_state.r_curr_req = None

def process_bulk_file(uploaded_file):
    """Extracts data from Excel/CSV, converts names to SMILES and generates questions."""
    if not uploaded_file: return
    current_lang = st.session_state.get("lang", "en")
    texts = TEXTS[current_lang]
       
    # --- UI: progress and state ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- Read file ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Error reading file: {e}")
        return

    # === Flexible column names mapping ===
    col_map = {}
    possible_names = {
        'missing': ['missing_molecule', 'molécula_faltante', 'missing_name', 'molécula faltante'],
        'q_name': ['question_name', 'nombre_pregunta', 'reaction_name', 'nombre de la pregunta'],
        'fb_pos': ['correct_feedback', 'retroalimentación_correcta', 'retroalimentacion_correcta', 'feedback correcto'],
        'fb_neg': ['incorrect_feedback', 'retroalimentación_incorrecta', 'retroalimentacion_incorrecta', 'feedback incorrecto']
    }

    # Normalize column names
    actual_cols = {str(c).lower().strip(): c for c in df.columns}
    
    for internal_key, aliases in possible_names.items():
        for alias in aliases:
            clean_alias = alias.lower().strip()
            if clean_alias in actual_cols:
                # Save column real name
                col_map[internal_key] = actual_cols[clean_alias]
                break
            
    # --- Validate required column ---
    if 'missing' not in col_map:
        st.error("Column `Missing_Molecule` is required.")
        status_text.empty()
        progress_bar.empty()        
        return

    # --- Counters ---
    added = 0
    failed = 0
    logs = []

    # --- PRINCIPAL LOOP ---
    for idx, row in df.iterrows():
        row_num = idx + 2
        row_prefix = texts["bulk_row_prefix"].format(row_num)
        status_text.text(f"Processing row {row_num}...")
        progress_bar.progress((idx + 1) / len(df))

        # === 1. Missing_Name ===
        missing_name_raw = row.get(col_map['missing'])
        if pd.isna(missing_name_raw) or str(missing_name_raw).strip() == '':
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_empty_missing']}")
            continue
        missing_name = str(missing_name_raw).strip()

        # === 2. Question_Name (optional) ===
        q_name_col = col_map.get('q_name')
        reaction_name = str(row[q_name_col]).strip() if q_name_col and pd.notna(row[q_name_col]) else missing_name

        # === 3. Feedbacks ===
        def clean_fb(key):
            col = col_map.get(key)
            if col and pd.notna(row[col]):
                val = str(row[col]).strip()
                return "" if val.lower() in ["nan", "none", ""] else val
            return ""

        correct_feedback = clean_fb('fb_pos')
        incorrect_feedback = clean_fb('fb_neg')
               
        # === 4. Process Reactants, Agents, and Products ===
        def process_list(prefix, current_row, is_agent=False):
            parts = []
            # Find columns starting with R, A, or P
            cols = [c for c in df.columns if str(c).startswith(prefix) and c not in col_map.values()]
            for c in cols:
                val = current_row.get(c)
                if pd.isna(val) or str(val).strip() == "": continue
                val_str = str(val).strip()
                
                # If it's agent text between brackets, or any text for agents, leave it as is
                if is_agent and val_str.startswith("[") and val_str.endswith("]"):
                    parts.append(val_str)
                else:
                    # Try to convert name/SMILES to a valid SMILES
                    s = name_to_smiles(val_str)
                    if s:
                        parts.append(s)
                    else:
                        # If name_to_smiles fails but it's an agent, we wrap it in brackets to treat as text
                        if is_agent:
                            parts.append(f"[{val_str}]")
                        else:
                            # For R and P, if it's not a valid SMILES/Name, we keep it to try to match Missing_Name
                            parts.append(val_str)
            return parts

        reactants_smiles = process_list('R', row)
        agents_smiles = process_list('A', row, is_agent=True)
        products_smiles = process_list('P', row)

        if not reactants_smiles and not products_smiles:
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_no_rp']}")
            continue

        # === 5. Handle Missing Molecule ===
        # Convert missing_name to SMILES if possible to match the lists
        missing_smiles = name_to_smiles(missing_name)
        if not missing_smiles:
            # Fallback for names that couldn't be converted but are in the reaction as text
            missing_smiles = missing_name

        # Check if it exists in the reaction components
        all_molecules = reactants_smiles + agents_smiles + products_smiles
        if missing_smiles not in all_molecules and f"[{missing_smiles}]" not in all_molecules:
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_missing_not_in_reaction'].format(missing_name)}")
            continue

        # === 6. Build Reaction SMILES ===
        reaction_smiles = build_reaction_smiles(reactants_smiles, agents_smiles, products_smiles)
        
        # === 7. Question Generation ===
        img = generate_reaction_image(reaction_smiles, missing_smiles)
        
        if not img:
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_image_failed']}")
        else:
            st.session_state.reaction_questions.append({
                'name': reaction_name,
                'reactants_smiles': reactants_smiles,
                'agents_smiles': agents_smiles,
                'products_smiles': products_smiles,
                'missing_smiles': missing_smiles,
                'reaction_smiles': reaction_smiles,
                'img_base64': img,
                'correct_feedback': correct_feedback,
                'incorrect_feedback': incorrect_feedback
            })
            added += 1

    # --- Finalize ---
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.bulk_success_message = None
    st.session_state.bulk_error_logs = None

    if added > 0:
        st.session_state.jsme_normalized = False
        st.session_state.bulk_success_message = texts["bulk_summary_success"].format(added, failed)
        
    if failed > 0:
        st.session_state.bulk_error_logs = {
            'count': failed,
            'logs': logs,
            'title': texts["bulk_summary_title"].format(failed)
        }

    st.session_state.jsme_normalized = False
    
    if added > 0 or failed > 0:
        st.rerun()

# ========================================================================
# 4. MAIN APP RENDERER
# ========================================================================

def render_reaction_app(lang=None):
    """Core function to render the Reaction Generator app."""
    
    # --- PARCHE DE COMPATIBILIDAD ---
    # Si la herramienta se ejecuta dentro de la suite y falta esta variable, 
    # la inicializamos para evitar el AttributeError.
    if "lang_toggle" not in st.session_state:
        st.session_state.lang_toggle = False
        
    # --- 1. Initialization ---
    if "reaction_questions" not in st.session_state:
        st.session_state.reaction_questions = []
    if "reactants_str" not in st.session_state:
        st.session_state.reactants_str = ""
    if "agents_str" not in st.session_state:
        st.session_state.agents_str = ""
    if "products_str" not in st.session_state:
        st.session_state.products_str = ""
    if "jsme_normalized" not in st.session_state:
        st.session_state.jsme_normalized = False
    if "show_jsme" not in st.session_state:
        st.session_state.show_jsme = False
    if "search_result" not in st.session_state:
        st.session_state.search_result = None
    if "search_counter" not in st.session_state:
        st.session_state.search_counter = 0

    # --- 2. Language Logic ---
    if lang:
        st.session_state.lang = lang
    elif "lang" not in st.session_state:
        st.session_state.lang = "es"
    
    texts = TEXTS[st.session_state.lang]

    # --- 3. Title and Intro ---
    st.title(texts["title"])
    st.markdown(texts["intro"])
    st.markdown("---")
    
    with st.expander(texts["preview_title"]):
        st.write(texts["preview_intro"])
        st.info(f"**{texts['preview_question_react']}**")
        st.latex(r"CH_3-CH_2-Br \quad + \quad NaOH \quad \xrightarrow{\quad} \quad \text{[ ? ]}")        
        st.write(texts["preview_footer_react"])

    input_col, list_col = st.columns([1.8, 1.2])

    # --- 4. Input Column ---
    with input_col:
        if "lang_toggle" not in st.session_state: st.session_state.lang_toggle = False
        
        tab1, tab2 = st.tabs([texts["tab_manual"], texts["tab_bulk"]])
        
        with tab1:
            # --- BUSCADOR ---
            with st.form("search_form"):
                st.subheader(texts["search_title"])
                c_s1, c_s2 = st.columns([3, 1])
                s_name = c_s1.text_input(texts["name_input_label"], key=f"s_in_{st.session_state.search_counter}", label_visibility="collapsed")
                if c_s2.form_submit_button(texts["search_button"], use_container_width=True):
                    res = get_smiles_from_name(s_name)
                    if res: st.session_state.search_result = res
                    else: st.error(texts["name_error"].format(s_name))
            
            # --- BOTONES PARA AÑADIR ---
            if st.session_state.search_result:
                res = st.session_state.search_result
                st.info(f"SMILES: `{res}`")
                c1, c2, c3 = st.columns(3)
                
                def add_to(key):
                    current = st.session_state.get(key, "")
                    if current: st.session_state[key] = f"{current}, {res}"
                    else: st.session_state[key] = res
                    st.session_state.search_result = None
                    st.session_state.search_counter += 1
                    st.rerun()
    
                if c1.button(texts["add_to_reactants"]): add_to("reactants_str")
                if c2.button(texts["add_to_agents"]): add_to("agents_str")
                if c3.button(texts["add_to_products"]): add_to("products_str")
    
            st.write("---")
            
            # --- CAMPOS DE TEXTO (REACTION BUILDER) ---
            col_r, col_a, col_p = st.columns(3)
            
            # IMPORTANTE: value debe estar vinculado al session_state
            r_val = col_r.text_area(texts["reactants_label"], value=st.session_state.reactants_str, height=100)
            a_val = col_a.text_area(texts["agents_label"], value=st.session_state.agents_str, height=100)
            p_val = col_p.text_area(texts["products_label"], value=st.session_state.products_str, height=100)
            
            # Guardamos lo que el usuario escribe a mano
            st.session_state.reactants_str = r_val
            st.session_state.agents_str = a_val
            st.session_state.products_str = p_val
    
            # Listas para el selector
            r_list = [s.strip() for s in r_val.split(',') if s.strip()]
            a_list = [s.strip() for s in a_val.split(',') if s.strip()]
            p_list = [s.strip() for s in p_val.split(',') if s.strip()]
            all_mols = r_list + p_list
            
            col_m, col_n = st.columns(2)
            missing_idx = col_m.selectbox(
                texts["select_missing"], 
                range(len(all_mols)), 
                format_func=lambda x: f"{all_mols[x]} ({'R' if x < len(r_list) else 'P'})"
            ) if all_mols else None
            
            q_name = col_n.text_input(texts["reaction_name_label"], value=f"Reaction {len(st.session_state.reaction_questions)+1}")
            
            # --- BOTÓN FINAL ---
            cb1, cb2 = st.columns(2)
            if cb1.button(texts["add_reaction_button"], type="primary", use_container_width=True):
                if q_name and missing_idx is not None:
                    miss = all_mols[missing_idx]
                    rxn_smiles = f"{'.'.join(r_list)}>{'.'.join(a_list)}>{'.'.join(p_list)}"
                    img = generate_reaction_image(rxn_smiles, miss)
                    
                    if img:
                        st.session_state.reaction_questions.append({
                            'name': q_name, 'missing_smiles': miss, 'img_base64': img,
                            'correct_feedback': 'Correct!', 'incorrect_feedback': '', 'normalized': False
                        })
                        st.rerun()
                    else:
                        st.error("Error al generar imagen.")
    
            if cb2.button(texts["new_question"], use_container_width=True):
                st.session_state.reactants_str = ""; st.session_state.agents_str = ""; st.session_state.products_str = ""
                st.rerun()
            
            # --- 5. Output Column ---
            with list_col:
                st.subheader(texts["added_questions_title"])
                
                if not st.session_state.reaction_questions:
                    st.info(texts["no_questions"])
                else:
                    # Recalculate pending every render
                    pending_questions = [
                        i for i, q in enumerate(st.session_state.reaction_questions)
                        if not q.get('normalized', False)
                    ]
        
                    # -------------------------------------------------------
                    # Unique JSME editor
                    # It shows whwn there is something in r_jsme_in
                    # -------------------------------------------------------
                    if "r_jsme_in" not in st.session_state:
                        st.session_state.r_jsme_in = None
                    if "r_curr_req" not in st.session_state:
                        st.session_state.r_curr_req = None
        
                    jsme_result = jsme_editor(
                        smiles_json=st.session_state.r_jsme_in,
                        key="reaction_jsme_global_processor"
                    )
        
                    # Process editor editor response on arrival
                    if st.session_state.r_jsme_in and jsme_result:
                        try:
                            res = json.loads(jsme_result)
                            if (
                                st.session_state.r_curr_req
                                and res.get('id') == st.session_state.r_curr_req.get('id')
                            ):
                                idx = st.session_state.r_curr_req['index']
                                new_smiles = res.get('smiles', '').strip()
                                
                                if new_smiles:
                                    old_smiles = st.session_state.reaction_questions[idx]['missing_smiles']
                                    st.session_state.reaction_questions[idx]['missing_smiles'] = new_smiles
                                    st.session_state.reaction_questions[idx]['normalized'] = True
                                    st.success(texts["normalized_success"].format(idx + 1, old_smiles,new_smiles))
                                else:
                                    st.warning(texts["no_valid_smiles_warning"].format(idx + 1))
                                
                                # Clean and go to next (if existent)
                                st.session_state.r_jsme_in = None
                                st.session_state.r_curr_req = None
                                start_jsme_normalization_loop()
                                st.rerun()
                        except json.JSONDecodeError:
                            st.error(texts["json_decode_error"])
                        except Exception as e:
                            st.error(texts["jsme_processing_error"].format(str(e)))
        
                    # -------------------------------------------------------
                    # Button to iniciate / continue normalization
                    # -------------------------------------------------------
                    if pending_questions:
                        num_pending = len(pending_questions)
                        btn_text = texts["normalize_button"].format(num_pending)
                        
                        # Differentiate if there is a proccess running
                        if st.session_state.r_jsme_in:
                            btn_text = texts["continue_standardization"].format(num_pending)
        
                        if st.button(
                            btn_text,
                            type="primary",
                            use_container_width=True,
                            key="btn_start_normalize_reactions",
                            help=texts.get("continue_standardization_help", f"{num_pending} question(s) still pending standardization")
                        ):
                            start_jsme_normalization_loop()
                            st.rerun()
        
                    # -------------------------------------------------------
                    # All normalized → Download XML button
                    # -------------------------------------------------------
                    else:
                        try:
                            xml_data = generate_xml(st.session_state.reaction_questions, st.session_state.lang)
                            st.download_button(
                                label=texts["download_xml_button"],
                                data=xml_data,
                                file_name="moodle_reactions.xml",
                                mime="application/xml",
                                use_container_width=True,
                                type="primary",
                                icon=":material/download:"
                            )
                        except Exception as e:
                            st.error(texts["xml_error"].format(str(e)))
        
                    # -------------------------------------------------------
                    # Clean all button
                    # -------------------------------------------------------
                    if st.button(
                        texts["clear_all_button"],
                        use_container_width=True,
                        icon=":material/delete:",
                        key="btn_clear_all_reactions"
                    ):
                        st.session_state.reaction_questions = []
                        st.session_state.r_jsme_in = None
                        st.session_state.r_curr_req = None
                        st.session_state.jsme_normalized = False
                        st.rerun()
        
                    st.markdown("---")
        
                    # -------------------------------------------------------
                    # List of added questions
                    # -------------------------------------------------------
                    for i, q in enumerate(st.session_state.reaction_questions):
                        is_normalized = q.get('normalized', False)
                        
                        with st.container(border=True):
                            cols = st.columns([5, 1])
                            with cols[0]:
                                if is_normalized:
                                    st.success(texts["is_normalized"])
                                else:
                                    st.warning(texts["is_pending"])
                                
                                st.markdown(f"**{i+1}. {q['name']}**")
                                st.image(
                                    io.BytesIO(base64.b64decode(q['img_base64'])),
                                    use_container_width=True
                                )
                                st.caption(f"SMILES: `{q['missing_smiles']}`")
                            
                            with cols[1]:
                                if st.button("🗑️", key=f"del_reaction_{i}"):
                                    st.session_state.reaction_questions.pop(i)
                                    if (
                                        st.session_state.r_curr_req
                                        and st.session_state.r_curr_req.get('index') == i
                                    ):
                                        st.session_state.r_jsme_in = None
                                        st.session_state.r_curr_req = None
                                    st.rerun()
# ========================================================================
# 5. STANDALONE EXECUTION
# ========================================================================

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
    render_reaction_app(st.session_state.lang)
