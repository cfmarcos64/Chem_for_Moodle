# -*- coding: utf-8 -*-
"""
Moodle Reaction Question Generator (with Bulk Upload)
@author: Carlos Fernandez Marcos
"""
import streamlit as st
import xml.etree.ElementTree as ET
import requests
import io
import base64
from PIL import Image
import numpy as np
import pandas as pd  # Required for bulk upload
from my_component import jsme_editor
import json

# ===================================================================
# 1. MODULE AVAILABILITY CHECKS
# ===================================================================

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    rdDepictor = None
    rdMolDraw2D = None
    RDKIT_AVAILABLE = False

PANDAS_AVAILABLE = True  # Already imported above
NUMPY_AVAILABLE = True if 'np' in globals() else False

# ========================================================================
# 2. MULTILINGUAL TEXTS
# ========================================================================

TEXTS = {
    "es": {
        "add_reaction_button": "Guardar Pregunta",
        "add_to_agents": "Añadir a Flecha (Agentes)",
        "add_to_products": "Añadir a Productos",
        "add_to_reactants": "Añadir a Reactivos",
        "added_questions_title": "Lista de Preguntas",
        "agents_label": "Sobre la Flecha (SMILES o [Texto]):",
        "bulk_error_empty_missing": "no se ha podido procesar: la columna Missing_Name está vacía.",
        "bulk_error_image_failed": "no se ha podido procesar: Error al generar la imagen de reacción.",
        "bulk_error_missing_not_in_reaction": "no se ha podido procesar: La molécula faltante no es parte de la reacción.",
        "bulk_error_no_rp": "no se ha podido procesar: No se encontraron reactivos ni productos.",
        "bulk_error_read": "Error al leer el archivo.",
        "bulk_error_row": "Fila {} omitida: {}",
        "bulk_error_smiles_not_found": "no se ha podido procesar: SMILES no encontrado para '{}'.",
        "bulk_info": "### Carga Masiva\nSube un archivo Excel/CSV con columnas: `Nombre_Pregunta`, `R1`, `R2`, `P1`, `P2`, `A1`, `A2`, `Molécula_Faltante`, `Retroalimentación`.",
        "bulk_row_prefix": "La fila {}",
        "bulk_summary_success": "Completado: {} añadidas, {} fallaron.",
        "bulk_summary_title": "❌ Filas Omitidas ({} Fallaron)",
        "change_language": "Change language to English",
        "clear_all_button": "Borrar Todas",
        "correct_feedback_label": "Retroalimentación Correcta:",
        "download_xml_button": "2. Descargar XML para Moodle",
        "img_warning": "No se pudo generar la imagen de la reacción.",
        "incorrect_feedback_label": "Retroalimentación Incorrecta (Opcional):",
        "intro": "Configura una reacción química ocultando uno de sus componentes (reactivo, producto o agentes). El alumno deberá identificar la parte faltante y dibujarla en el editor JSME para completar la secuencia.",
        "jsme_error": "Ninguna normalizada",
        "jsme_partial": "Parcial: **{} de {}** normalizadas",
        "jsme_success": "ÉXITO: **{} de {}** normalizadas correctamente",
        "missing_mol_warning": "La molécula faltante no aparece en la reacción.",
        "name_error": "No se encontró SMILES para '{}'.",
        "name_input_label": "Nombre del compuesto:",
        "name_warning": "Por favor, introduce un nombre para la pregunta.",
        "new_question": "Nueva Pregunta",
        "no_questions": "Aún no hay preguntas añadidas.",
        "normalize_button": "1. Normalizar con JSME",
        "normalize_first": "⚠️ Primero normaliza con JSME antes de descargar.",
        "preview_intro": "El alumno verá algo como esto:",
        "preview_footer_react": "➡️ El alumno deberá identificar el compuesto que falta y dibujarlo en el editor.",
        "preview_question_react": "Dibuja la molécula que falta en la siguiente reacción:",
        "preview_title": "👁️ Ver ejemplo de pregunta en Moodle",
        "process_bulk_button": "Procesar Archivo",
        "products_label": "Productos (SMILES separados por comas):",
        "question_text": "Dibuja la molécula que falta en la siguiente reacción:",
        "reactants_label": "Reactivos (SMILES separados por comas):",
        "reaction_name_label": "Nombre de la Reacción:",
        "search_button": "Buscar SMILES",
        "search_title": "Búsqueda por Nombre (NCI CIR)",
        "select_missing": "Selecciona la molécula faltante:",
        "select_molecule_warning": "Selecciona una molécula de la lista.",
        "smiles_empty_error": "Los campos no pueden estar vacíos.",
        "smiles_found": "SMILES encontrado: {}",
        "smiles_invalid_error": "SMILES inválido: '{}'.",
        "tab_bulk": "Carga Masiva",
        "tab_manual": "Entrada Manual",
        "title": ":material/science: Generador de Preguntas de Reacción para Moodle",
        "upload_file_label": "Selecciona archivo Excel/CSV:",
        "xml_error": "Error al generar XML: {}"
    },
    "en": {
        "add_reaction_button": "Save Question",
        "add_to_agents": "Add to Arrow (Agents)",
        "add_to_products": "Add to Products",
        "add_to_reactants": "Add to Reactants",
        "added_questions_title": "Questions List",
        "agents_label": "Over the Arrow (SMILES or [Text]):",
        "bulk_error_empty_missing": "could not be processed: Missing_Name column is empty.",
        "bulk_error_image_failed": "could not be processed: Error generating reaction image.",
        "bulk_error_missing_not_in_reaction": "could not be processed: Missing molecule is not in reaction.",
        "bulk_error_no_rp": "could not be processed: No reactants or products found.",
        "bulk_error_read": "Error reading file.",
        "bulk_error_row": "Row {} skipped: {}",
        "bulk_error_smiles_not_found": "could not be processed: SMILES not found for '{}'.",
        "bulk_info": "### Bulk Upload\nUpload Excel/CSV with columns: `Question_Name`, `R1`, `R2`, `P1`, `P2`, `A1`, `A2`, `Missing_Molecule`, `Feedback`.",
        "bulk_row_prefix": "Row {}",
        "bulk_summary_success": "Completed: {} added, {} failed.",
        "bulk_summary_title": "❌ Skipped Rows ({} Failed)",
        "change_language": "Cambiar idioma a Español",
        "clear_all_button": "Clear All",
        "correct_feedback_label": "Correct Feedback:",
        "download_xml_button": "2. Download Moodle XML",
        "img_warning": "The reaction image couldn't be generated.",
        "incorrect_feedback_label": "Incorrect Feedback (Optional):",
        "intro": "Set up a chemical reaction by hiding one of its components (reactant, product, or agents). The student must identify the missing part and draw it in the JSME editor to complete the sequence.",
        "jsme_error": "None normalized",
        "jsme_partial": "Partial: **{} of {}** normalized",
        "jsme_success": "SUCCESS: **{} of {}** normalized correctly",
        "missing_mol_warning": "The missing molecule is not in the reaction.",
        "name_error": "Could not find SMILES for '{}'.",
        "name_input_label": "Compound name:",
        "name_warning": "Please, introduce a name for the question.",
        "new_question": "New Question",
        "no_questions": "No questions added yet.",
        "normalize_button": "1. Normalize with JSME",
        "normalize_first": "⚠️ Normalize with JSME before downloading.",
        "question_text": "Draw the missing molecule in the reaction:",
        "preview_intro": "The student will see something like this:",
        "preview_footer_react": "➡️ The student must identify the missing component and draw it in the molecular editor.",
        "preview_question_react": "Draw the missing molecule in the reaction:",
        "preview_title": "👁️ View sample Moodle question",
        "process_bulk_button": "Process File",
        "products_label": "Products (SMILES, comma-separated):",
        "reactants_label": "Reactants (SMILES, comma-separated):",
        "reaction_name_label": "Reaction Name:",
        "search_button": "Search SMILES",
        "search_title": "Search by Name (NCI CIR)",
        "select_missing": "Select missing molecule:",
        "select_molecule_warning": "Select a molecule from the list.",
        "smiles_empty_error": "Fields cannot be empty.",
        "smiles_found": "SMILES found: {}",
        "smiles_invalid_error": "Invalid SMILES: '{}'.",
        "tab_bulk": "Bulk Upload",
        "tab_manual": "Manual Entry",
        "title": ":material/science: Moodle Reaction Question Generator",
        "upload_file_label": "Select Excel/CSV file:",
        "xml_error": "XML Error: {}"
    }
}

# ========================================================================
# 3. HELPER FUNCTIONS
# ========================================================================

def draw_mol_consistent(mol, fixed_bond_length: float = 25.0, padding: int = 10) -> Image.Image:
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
    """
    Parse reaction SMILES string in format: Reactants>Agents>Products.
    Returns: (reactants_list, agents_list, products_list)
    """
    parts = reaction_smiles.split('>')
    if len(parts) != 3:
        # Basic fallback for simpler strings
        return [reaction_smiles.split('>>')[0]], [], [reaction_smiles.split('>>')[1]] if '>>' in reaction_smiles else ([],[],[])
    
    reactants = [s.strip() for s in parts[0].split('.') if s.strip()]
    agents = [s.strip() for s in parts[1].split('.') if s.strip()]
    products = [s.strip() for s in parts[2].split('.') if s.strip()]
    return reactants, agents, products



##########################################################################
def normalize_text(text):
    """Fallback: normalize special characters if font doesn't support them"""
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N',
        '°': 'deg', 'º': 'deg',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
#########################################################################






def draw_reaction(r_str, p_str, a_str=""):
    if not RDKIT_AVAILABLE or not AllChem:
        return None
    
    # Process agents individually to handle mixed SMILES and [Text]
    agents_list = a_str.split('.') if a_str else []
    final_agents = []
    display_agent_texts = []  # List to hold multiple texts
    text_counter = 2  # Start map numbers from 2
    
    for a in agents_list:
        if a.startswith("[") and a.endswith("]"):
            text = a[1:-1]  # Extract text without brackets
            text = normalize_text(text)
            display_agent_texts.append(text)
            final_agents.append(f"[*:{text_counter}]")  # Unique placeholder
            text_counter += 1
        else:
            final_agents.append(a)
    
    a_part = ".".join(final_agents)
    rxn_smarts = f"{r_str}>{a_part}>{p_str}"
    
    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        d2d = rdMolDraw2D.MolDraw2DCairo(900, 350)
        opts = d2d.drawOptions()
        opts.fixedBondLength = 40.0
                    
        missing_symbol = " { ? } "
        for role_list in [rxn.GetReactants(), rxn.GetAgents(), rxn.GetProducts()]:
            for mol in role_list:
                for atom in mol.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num == 1:
                        atom.SetProp("atomLabel", missing_symbol)
                        atom.SetAtomMapNum(0)
                    elif map_num >= 2:
                        idx = map_num - 2
                        if idx < len(display_agent_texts):
                            atom.SetProp("atomLabel", display_agent_texts[idx])
                        atom.SetAtomMapNum(0)
        
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        return base64.b64encode(d2d.GetDrawingText()).decode('utf-8')
    except:
        return None

def generate_reaction_image(reaction_smiles: str, missing_smiles: str):
    """
    Identifies the missing molecule and replaces ONLY that molecule 
    with a placeholder [*:1], preserving others.
    """
    if not RDKIT_AVAILABLE:
        return None
    
    reactants, agents, products = parse_reaction_smiles(reaction_smiles)
    
    # Placeholder for the missing part
    placeholder = "[*:1]"
    
    # Function to replace only the matching SMILES in a list
    def replace_missing(mol_list, missing):
        # We use canonical SMILES for a safer comparison
        try:
            m_can = Chem.MolToSmiles(Chem.MolFromSmiles(missing), canonical=True)
        except:
            m_can = missing
            
        new_list = []
        found = False
        for s in mol_list:
            # SKIP VALIDATION for text agents like [heat]
            if s.startswith("[") and s.endswith("]") and not ":" in s:
                new_list.append(s)
                continue
                
            try:
                s_can = Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
                if s_can == m_can and not found:
                    new_list.append(placeholder)
                    found = True 
                else:
                    new_list.append(s)
            except:
                if s == missing and not found:
                    new_list.append(placeholder)
                    found = True
                else:
                    new_list.append(s)
        return new_list, found

    # Try to find and replace in reactants, then products
    new_reactants, found_in_r = replace_missing(reactants, missing_smiles)
    if found_in_r:
        new_products = products
    else:
        new_products, _ = replace_missing(products, missing_smiles)
        new_reactants = reactants

    # Join back with dots for the SMARTS
    r_str = ".".join(new_reactants)
    p_str = ".".join(new_products)
    a_str = ".".join(agents)
    
    return draw_reaction(r_str, p_str, a_str)

def generate_xml(questions, lang: str) -> bytes:
    quiz = ET.Element('quiz')
    instruction_text = TEXTS[lang].get("question_text", "Draw the missing molecule:")
    prompt = f'<p>{instruction_text}</p>'
    
    for q in questions:
        smiles_to_use = q['missing_smiles']
        question = ET.SubElement(quiz, 'question', type='pmatchjme')
        
        # Question name
        name_el = ET.SubElement(question, 'name')
        ET.SubElement(name_el, 'text').text = q['name']
        
        # Question with image
        qtext = ET.SubElement(question, 'questiontext', format='html')
        ET.SubElement(qtext, 'text').text = f'{prompt}<img src="data:image/png;base64,{q["img_base64"]}" alt="Reaction"/>'
        
        # Right answer
        answer = ET.SubElement(question, 'answer', fraction='100', format='moodle_auto_format')
        escaped_smiles = smiles_to_use.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")
        ET.SubElement(answer, 'text').text = f"match({escaped_smiles})"
        
        # Model answer
        model = ET.SubElement(question, 'modelanswer')
        model.text = q['missing_smiles']

        # Feedback RIGHT answer
        if q['correct_feedback']:
            fb = ET.SubElement(answer, 'feedback', format='html')
            fb_text = ET.SubElement(fb, 'text')
            fb_text.text = f'<p>{q["correct_feedback"]}</p>'
        
        # Feedback WRONG answer
        if q['incorrect_feedback']:
            ans_inc = ET.SubElement(question, 'answer', fraction='0', format='moodle_auto_format')
            ET.SubElement(ans_inc, 'text').text = "*"
            fb_inc = ET.SubElement(ans_inc, 'feedback', format='html')
            fb_inc_text = ET.SubElement(fb_inc, 'text')
            fb_inc_text.text = f'<p>{q["incorrect_feedback"]}</p>'
            ET.SubElement(ans_inc, 'atomcount').text = "0"
    
    # Generate pretty XML
    xml_str = ET.tostring(quiz, encoding='utf-8', method='xml').decode('utf-8')
    import xml.dom.minidom
    return xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ").encode('utf-8')

def get_smiles_from_name(name: str) -> str | None:
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        smiles = res.text.strip().split('\n')[0]
        return smiles if "ERROR" not in smiles else None
    except:
        return None

def name_to_smiles(name: str) -> str | None:
    if not name: return None
    name = str(name).strip()
    # Si ya es SMILES válido, lo devolvemos
    mol = Chem.MolFromSmiles(name)
    if mol is not None: return name
    # Si no, buscamos en Cactus
    return get_smiles_from_name(name)

def build_reaction_smiles(reactants_smiles: list, agents_smiles: list, products_smiles: list) -> str:
    """
    Build reaction SMILES string in format: Reactants>Agents>Products
    Multiple molecules are joined with '.'
    """
    reactants_part = ".".join(reactants_smiles) if reactants_smiles else ""
    agents_part = ".".join(agents_smiles) if agents_smiles else ""
    products_part = ".".join(products_smiles) if products_smiles else ""
    
    return f"{reactants_part}>{agents_part}>{products_part}"

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

def search_compound_wrapper(counter: int):
    name = st.session_state.get(f"search_input_{counter}", "").strip()
    if not name: return
    # Obtener textos dinámicamente
    current_lang = st.session_state.get("lang", "en")
    local_texts = TEXTS[current_lang]
    
    with st.spinner("Searching..."):
        smiles = get_smiles_from_name(name)
    if smiles: st.session_state.search_result = smiles
    else: st.error(local_texts["name_error"].format(name))

# ========================================================================
# 4. STREAMLIT APP SETUP (Encapsulated for Integration)
# ========================================================================

def reset_reaction_inputs():
    """Limpia los campos de la reacción de forma segura."""
    st.session_state["reactants_str"] = ""
    st.session_state["agents_str"] = ""
    st.session_state["products_str"] = ""
    st.session_state["search_result"] = None
    # Incrementamos el contador para limpiar también el input de búsqueda
    st.session_state["search_counter"] += 1


def render_reaction_app(lang=None):
    """
    Renders the Reaction interface. 
    If lang is provided, it uses it (Integration mode).
    If lang is None, it manages its own state (Standalone mode).
    """
    
    # --- 4.1. Session State Initialization ---
    # We use 'r_' prefix for some keys to avoid collision with other tools
    if "reaction_questions" not in st.session_state:
        st.session_state.reaction_questions = []
    if "search_result" not in st.session_state:
        st.session_state.search_result = None
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
    if "search_counter" not in st.session_state:
        st.session_state.search_counter = 0

    # --- 4.2. Language Logic ---
    if lang:
        # Integrated mode: Use lang from Master App
        st.session_state.lang = lang
    else:
        # Standalone mode: Internal language switch
        if 'lang' not in st.session_state:
            st.session_state.lang = 'es'
        
        col_lang, _ = st.columns([3, 2])
        with col_lang:
            if st.button("EN" if st.session_state.lang == "es" else "ES", key="standalone_lang_btn"):
                st.session_state.lang = "en" if st.session_state.lang == "es" else "es"
                st.rerun()

    texts = TEXTS[st.session_state.lang]

    # --- 4.3. Title and Intro ---
    st.title(texts["title"])
    st.markdown(texts["intro"])
    st.markdown("---")
    
    with st.expander(texts["preview_title"]):
        st.write(texts["preview_intro"])
        st.info(f"**{texts['preview_question_react']}**")
        st.latex(r"CH_3-CH_2-Br \quad + \quad NaOH \quad \xrightarrow{\quad} \quad \text{[ ? ]}")        
        st.write(texts["preview_footer_react"])

   # --- 4.4. Main Layout ---
    input_col, list_col = st.columns([2, 1])

    with input_col:
        tab1, tab2 = st.tabs([texts["tab_manual"], texts["tab_bulk"]])
        
        with tab1:
            st.subheader(texts["search_title"])
            
            # Buscador con clave fija para evitar que se pierda al recargar
            col_search, col_btn = st.columns([4, 1])
            with col_search:
                # Usamos una clave que no cambie durante la sesión para el input
                query = st.text_input(texts["name_input_label"], key="main_search_input", label_visibility="collapsed")
            
            with col_btn:
                if st.button(texts["search_button"], use_container_width=True, type="secondary"):
                    if query:
                        result = name_to_smiles(query)
                        if result:
                            st.session_state["search_result"] = result
                        else:
                            st.error("No se encontró la molécula")
                    else:
                        st.warning("Introduce un nombre")

            # Mostrar resultado de búsqueda y botones de añadir
            if st.session_state.get("search_result"):
                res_smiles = st.session_state["search_result"]
                st.info(f"Resultado: {res_smiles}")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button(texts["add_to_reactants"], use_container_width=True, key="add_r_fix"):
                        old = st.session_state.get("reactants_str", "")
                        st.session_state["reactants_str"] = f"{old}, {res_smiles}".strip(", ")
                        st.session_state["search_result"] = None
                        st.rerun()
                with c2:
                    if st.button(texts["add_to_agents"], use_container_width=True, key="add_a_fix"):
                        old = st.session_state.get("agents_str", "")
                        st.session_state["agents_str"] = f"{old}, {res_smiles}".strip(", ")
                        st.session_state["search_result"] = None
                        st.rerun()
                with c3:
                    if st.button(texts["add_to_products"], use_container_width=True, key="add_p_fix"):
                        old = st.session_state.get("products_str", "")
                        st.session_state["products_str"] = f"{old}, {res_smiles}".strip(", ")
                        st.session_state["search_result"] = None
                        st.rerun()

            st.markdown("---")

            # Áreas de texto: Sincronización directa con session_state
            col_r, col_a, col_p = st.columns(3)
            with col_r:
                st.text_area(texts["reactants_label"], key="reactants_str", height=100)
            with col_a:
                st.text_area(texts["agents_label"], key="agents_str", height=100)
            with col_p:
                st.text_area(texts["products_label"], key="products_str", height=100)

            # Preparar listas para el selector
            r_list = [s.strip() for s in st.session_state.get("reactants_str", "").split(',') if s.strip()]
            p_list = [s.strip() for s in st.session_state.get("products_str", "").split(',') if s.strip()]
            all_mols = r_list + p_list

            col_missing, col_name = st.columns(2)
            with col_missing:
                if all_mols:
                    # El selector usa un índice para no perderse
                    idx_m = st.selectbox(
                        texts["select_missing"],
                        range(len(all_mols)),
                        format_func=lambda x: f"{all_mols[x]} ({'R' if x < len(r_list) else 'P'})",
                        key="missing_selector_widget"
                    )
                    selected_smi = all_mols[idx_m]
                else:
                    st.info(texts["select_molecule_warning"])
                    selected_smi = None

            with col_name:
                n_count = len(st.session_state.get("reaction_questions", [])) + 1
                q_name = st.text_input(texts["reaction_name_label"], value=f"Reacción {n_count}", key="q_name_input")

            # Normalizador silencioso JSME
            final_smi = selected_smi
            if selected_smi:
                js_out = jsme_editor(
                    smiles_json=json.dumps({'smiles': selected_smi, 'id': 'norm'}),
                    key=f"jsme_silent_{selected_smi}" # Se refresca solo si cambia la selección
                )
                if js_out:
                    try: final_smi = json.loads(js_out).get('smiles', selected_smi)
                    except: pass

            # Botones finales
            st.divider()
            if st.button(texts["add_reaction_button"], type="primary", use_container_width=True):
                if not all_mols:
                    st.error("Faltan moléculas")
                else:
                    # Aquí llamarías a build_reaction_smiles y generate_reaction_image
                    # usando r_list, p_list y final_smi
                    st.success(f"Reacción '{q_name}' lista para añadir (lógica de guardado aquí)")

            st.button("Nueva Reacción", on_click=reset_reaction_inputs)

        # ---------------- TAB 2: Bulk ----------------
        with tab2:
            st.markdown(texts["bulk_info"])
            uploaded = st.file_uploader(texts["upload_file_label"], type=['xlsx', 'csv'], key="bulk_uploader_input")
            if uploaded and st.button(texts["process_bulk_button"], type="primary", use_container_width=True, key="bulk_proc_btn"):
                process_bulk_file(uploaded)
            
            if st.session_state.get('bulk_success_message'):
                st.success(st.session_state.bulk_success_message)
            if st.session_state.get('bulk_error_logs'):
                with st.expander(st.session_state.bulk_error_logs['title']):
                    for log in st.session_state.bulk_error_logs['logs']:
                        st.caption(log)

    # === OUTPUT COLUMN ===
    with list_col:
        st.subheader(texts["added_questions_title"])
        if st.session_state.reaction_questions:
            if not st.session_state.jsme_normalized:
                if st.button(texts["normalize_button"], use_container_width=True, type="secondary",
                icon=":material/rocket_launch:", key="norm_btn_list"):
                    st.session_state.show_jsme = True
                    st.rerun()

            if st.session_state.show_jsme:
                st.markdown("### JSME Normalization")
                with st.form(key="jsme_form_norm"):
                    for i, q in enumerate(st.session_state.reaction_questions):
                        st.markdown(f"**{i+1}. {q['name']}**")
                        jsme_editor(q['missing_smiles'], key=f"jsme_norm_{i}")
                    
                    if st.form_submit_button("Apply", type="primary", use_container_width=True):
                        for i in range(len(st.session_state.reaction_questions)):
                            val = st.session_state.get(f"jsme_norm_{i}")
                            if val: st.session_state.reaction_questions[i]['missing_smiles'] = val.strip()
                        st.session_state.jsme_normalized = True
                        st.session_state.show_jsme = False
                        st.rerun()

            if st.session_state.jsme_normalized:
                xml_data = generate_xml(st.session_state.reaction_questions, st.session_state.lang)
                st.download_button(texts["download_xml_button"], data=xml_data, file_name="reactions.xml", mime="application/xml", use_container_width=True, type="primary", icon= ":material/download:", key="dl_xml_btn")
            else:
                st.info(texts["normalize_first"])

            if st.button(texts["clear_all_button"], icon=":material/delete:", use_container_width=True, key="clear_all_btn"):
                st.session_state.reaction_questions = []
                st.session_state.jsme_normalized = False
                st.rerun()

            st.divider()
            for i, q in enumerate(st.session_state.reaction_questions):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{i+1}. {q['name']}**")
                    st.image(io.BytesIO(base64.b64decode(q['img_base64'])), width=450)
                with c2:
                    if st.button("🗑️", key=f"del_q_{i}"):
                        st.session_state.reaction_questions.pop(i)
                        st.rerun()
        else:
            st.info(texts["no_questions"])

# --- Standalone Execution ---
if __name__ == "__main__":
    st.set_page_config(page_title="Reaction Generator", layout="wide")
    render_reaction_app()