# -*- coding: utf-8 -*-
"""
Moodle Reaction Question Generator
Integrated Version for Chem_To_Moodle Suite
@author: Carlos Fernandez Marcos
"""

import re
import json
import uuid
import io
import base64
import xml.dom.minidom

import streamlit as st
import xml.etree.ElementTree as ET
import requests
import numpy as np
import pandas as pd
from PIL import Image
from my_component import jsme_editor

# ==============================================================================
# 1. MODULE AVAILABILITY CHECKS
# ==============================================================================

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

# numpy is imported unconditionally above; flag reflects actual availability
NUMPY_AVAILABLE = True

# ==============================================================================
# 2. BILINGUAL TEXTS (ALPHABETICALLY SORTED KEYS)
# ==============================================================================

TEXTS = {
    "en": {
        "add_reaction_button": "Add Reaction",
        "add_to_agents": "Add over the Arrow",
        "add_to_products": "Add to Products",
        "add_to_reactants": "Add to Reactants",
        "added_questions_title": "Added Questions",
        "agents_label": "Agents over the Arrow (SMILES or [Text]):",
        "bulk_error_empty_missing": "could not be processed: the Missing_Name column is empty.",
        "bulk_error_image_failed": "could not be processed: Error generating the reaction image.",
        "bulk_error_missing_not_in_reaction": "could not be processed: The missing molecule was found, but it is not part of the reactants/products in this row.",
        "bulk_error_no_rp": "could not be processed: No reactants (R*) or products (P*) were found.",
        "bulk_error_row": "Row {} skipped: {}",
        "bulk_info": "### Bulk Upload\nUpload Excel/CSV with columns:\n"
                     "`Question_Name`, `R1`, `R2`, ..., `P1`, `P2`, ..., `A1`, `A2`, ..., `Missing_Molecule`, `Correct_Feedback`, `Incorrect_Feedback`.\n"
                     "Use R for reactants, P for products and A for reagents/conditions over the arrow. "
                     "If A is text, write it between square brackets '[ ]'.",
        "bulk_row_prefix": "Row {}",
        "bulk_summary_success": "Completed: {} added, {} failed.",
        "bulk_summary_title": "❌ Skipped Rows ({} Failed)",
        "clear_all_button": "Clear All",
        "continue_standardization": "Continue standardization ({0} pending)",
        "continue_standardization_help": "{} question(s) still pending standardization",
        "download_xml_button": "Download XML",
        "file_read_error": "Error reading file: {}",
        "image_warning": "The reaction image couldn't be generated",
        "intro": "Set up a chemical reaction by hiding one of its components (reactant, product, or agents). "
                 "The student must identify the missing part and draw it in the JSME editor to complete the sequence.",
        "invalid_smiles_preview": "Invalid SMILES structure",
        "is_normalized": "✅ Normalized",
        "is_pending": "⚠️ Pending normalization",
        "jsme_processing_error": "Error processing JSME response: {0}",
        "json_decode_error": "Error interpreting JSME editor response (invalid JSON format)",
        "missing_column_error": "Column `Missing_Molecule` is required.",
        "mol_preview_error": "Could not preview the structure.",
        "name_error": "Could not find SMILES for '{}'.",
        "name_input_label": "Compound name:",
        "new_question": "New Question",
        "no_questions": "No questions added yet.",
        "no_valid_smiles_warning": "No valid SMILES received for question {0}",
        "normalization_error": "The normalized SMILES of molecules with generic groups may be incorrect. "
                               "Please review the question in Moodle.",
        "normalize_button": "Normalize {0} pending reaction(s)",
        "normalized_success": "Question {0} normalized: {1} → {2}",
        "preview_footer_react": "➡️ The student must identify the missing component and draw it in the molecular editor.",
        "preview_intro": "The student will see something like this:",
        "preview_question_react": "Draw the missing molecule in the reaction:",
        "preview_title": "👁️ View sample Moodle question",
        "process_bulk_button": "Process File",
        "processing_row": "Processing row {}...",
        "products_label": "Products (SMILES, comma-separated):",
        "question_text": "Draw the missing molecule in the reaction:",
        "reactants_label": "Reactants (SMILES, comma-separated):",
        "reaction_name_label": "Reaction Name:",
        "search_button": "Search SMILES",
        "search_title": "Search by Name (NCI CIR)",
        "select_missing": "Select missing molecule:",
        "smiles_found": "SMILES found: {}",
        "tab_bulk": "Bulk Upload",
        "tab_manual": "Manual Entry",
        "title": ":material/science: Moodle Reaction Question Generator",
        "upload_file_label": "Select Excel/CSV file:",
        "xml_error": "Error generating XML: {}",
    },
    "es": {
        "add_reaction_button": "Añadir Reacción",
        "add_to_agents": "Añadir sobre la Flecha",
        "add_to_products": "Añadir a Productos",
        "add_to_reactants": "Añadir a Reactivos",
        "added_questions_title": "Preguntas Añadidas",
        "agents_label": "Agentes sobre la Flecha (SMILES o [Texto]):",
        "bulk_error_empty_missing": "no se ha podido procesar: la columna Missing_Name está vacía.",
        "bulk_error_image_failed": "no se ha podido procesar: Error al generar la imagen de reacción.",
        "bulk_error_missing_not_in_reaction": "no se ha podido procesar: La molécula faltante se encontró, pero no forma parte de los reactivos/productos de esa fila.",
        "bulk_error_no_rp": "no se ha podido procesar: No se encontraron reactivos (R*) ni productos (P*).",
        "bulk_error_row": "Fila {} omitida: {}",
        "bulk_info": "### Carga Masiva\nSube un archivo Excel/CSV con columnas:\n"
                     "`Nombre_Pregunta`, `R1`, `R2`, ..., `P1`, `P2`, ..., `A1`, `A2`, ..., "
                     "`Molécula_Faltante`, `Retroalimentación_Correcta`, `Retroalimentación_Incorrecta`.\n"
                     "Usar R para reactivos, P para productos y A para reactivos/condiciones sobre la flecha. "
                     "Si A es texto, debe ir entre corchetes '[ ]'.",
        "bulk_row_prefix": "La fila {}",
        "bulk_summary_success": "Completado: {} añadidas, {} fallaron.",
        "bulk_summary_title": "❌ Filas Omitidas ({} Fallaron)",
        "clear_all_button": "Borrar Todas",
        "continue_standardization": "Continuar estandarización ({0} pendiente(s))",
        "continue_standardization_help": "{} pregunta(s) pendientes de estandarización",
        "download_xml_button": "Descargar XML",
        "file_read_error": "Error al leer el archivo: {}",
        "image_warning": "No se pudo generar la imagen de la reacción.",
        "intro": "Configura una reacción química ocultando uno de sus componentes (reactivo, producto o agentes). "
                 "El alumno deberá identificar la parte faltante y dibujarla en el editor JSME para completar la secuencia.",
        "invalid_smiles_preview": "Estructura SMILES no válida",
        "is_normalized": "✅ Normalizada",
        "is_pending": "⚠️ Pendiente de normalizar",
        "jsme_processing_error": "Error procesando respuesta JSME: {0}",
        "json_decode_error": "Error al interpretar la respuesta del editor JSME (formato JSON inválido)",
        "missing_column_error": "La columna `Missing_Molecule` es obligatoria.",
        "mol_preview_error": "No se pudo previsualizar la estructura.",
        "name_error": "No se encontró SMILES para '{}'.",
        "name_input_label": "Nombre del compuesto:",
        "new_question": "Nueva Pregunta",
        "no_questions": "Aún no hay preguntas añadidas.",
        "no_valid_smiles_warning": "No se recibió SMILES válido para la pregunta {0}",
        "normalization_error": "El SMILES normalizado de las moléculas con grupos genéricos puede ser incorrecto. "
                               "Revise la pregunta en Moodle.",
        "normalize_button": "Normalizar {0} reaccion(es) pendiente(s)",
        "normalized_success": "Pregunta {0} normalizada: {1} → {2}",
        "preview_footer_react": "➡️ El alumno deberá identificar la parte faltante y dibujarla en el editor.",
        "preview_intro": "El alumno verá algo como esto:",
        "preview_question_react": "Dibuja la molécula que falta en la siguiente reacción:",
        "preview_title": "👁️ Ver ejemplo de pregunta en Moodle",
        "process_bulk_button": "Procesar Archivo",
        "processing_row": "Procesando fila {}...",
        "products_label": "Productos (SMILES, separados por comas):",
        "question_text": "Dibuja la molécula que falta en la siguiente reacción:",
        "reactants_label": "Reactivos (SMILES, separados por comas):",
        "reaction_name_label": "Nombre de la Reacción:",
        "search_button": "Buscar SMILES",
        "search_title": "Búsqueda por Nombre (NCI CIR)",
        "select_missing": "Selecciona la molécula faltante:",
        "smiles_found": "SMILES encontrado: {}",
        "tab_bulk": "Carga Masiva",
        "tab_manual": "Entrada Manual",
        "title": ":material/science: Generador de Preguntas de Reacción para Moodle",
        "upload_file_label": "Selecciona archivo Excel/CSV:",
        "xml_error": "Error al generar XML: {}",
    },
}

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def draw_mol_consistent(mol, fixed_bond_length: float = 25.0, padding: int = 10) -> Image.Image:
    """Draw a molecule with consistent bond lengths and tight cropping.
    Returns a blank white image if RDKit or numpy is unavailable."""
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
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGB')
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
    """Parse a reaction SMILES string in the format Reactants>Agents>Products.
    Returns a tuple of (reactants, agents, products) as lists of SMILES strings."""
    parts = reaction_smiles.split('>')
    if len(parts) != 3:
        if '>>' in reaction_smiles:
            halves = reaction_smiles.split('>>')
            return halves[0].split('.'), [], halves[1].split('.')
        return [], [], []
    reactants = [s.strip() for s in parts[0].split('.') if s.strip()]
    agents = [s.strip() for s in parts[1].split('.') if s.strip()]
    products = [s.strip() for s in parts[2].split('.') if s.strip()]
    return reactants, agents, products


def normalize_text(text: str) -> str:
    """Replace accented and special characters for RDKit Cairo font compatibility.
    Applied to all atom labels extracted from $$label$$ groups and [text] agents
    before they are passed to the drawing engine."""
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n', 'º': 'deg',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def draw_reaction(r_str: str, p_str: str, a_str: str = "") -> str | None:
    """Generate a base64-encoded PNG of a reaction using RDKit.
    Generic group labels ($$label$$) and bracketed text agents ([text]) are
    rendered as custom atom labels on the diagram."""
    if not RDKIT_AVAILABLE or not AllChem:
        return None

    display_labels = []
    text_counter = 2  # Map number 1 is reserved for the { ? } placeholder

    def process_component_string(smiles_str: str) -> str:
        """Replace $$label$$ and [text] placeholders with mapped wildcard atoms
        so that RDKit can parse the SMILES and the labels are rendered correctly."""
        nonlocal text_counter
        parts = [s.strip() for s in smiles_str.split('.') if s.strip()]
        final_parts = []
        for p in parts:
            # A. Handle generic groups: $$label$$
            while "$$" in p:
                match = re.search(r"\$\$\s*([^$]+?)\s*\$\$", p)
                if match:
                    display_labels.append(match.group(1).strip())
                    p = p.replace(match.group(0), f"[*:{text_counter}]", 1)
                    text_counter += 1
                else:
                    break
            # B. Handle bracketed text agents: [catalyst]
            if p.startswith("[") and p.endswith("]") and ":" not in p:
                display_labels.append(normalize_text(p[1:-1].strip()))
                p = f"[*:{text_counter}]"
                text_counter += 1
            final_parts.append(p)
        return ".".join(final_parts)

    clean_r = process_component_string(r_str)
    clean_a = process_component_string(a_str)
    clean_p = process_component_string(p_str)
    rxn_smarts = f"{clean_r}>{clean_a}>{clean_p}"

    try:
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        d2d = rdMolDraw2D.MolDraw2DCairo(900, 350)
        d2d.drawOptions().fixedBondLength = 40.0
        for role_list in [rxn.GetReactants(), rxn.GetAgents(), rxn.GetProducts()]:
            for mol in role_list:
                for atom in mol.GetAtoms():
                    map_num = atom.GetAtomMapNum()
                    if map_num == 1:
                        atom.SetProp("atomLabel", " { ? } ")
                        atom.SetAtomMapNum(0)
                    elif map_num >= 2:
                        idx = map_num - 2
                        if idx < len(display_labels):
                            atom.SetProp("atomLabel", display_labels[idx])
                        atom.SetAtomMapNum(0)
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        return base64.b64encode(d2d.GetDrawingText()).decode('utf-8')
    except Exception:
        return None


def generate_reaction_image(reaction_smiles: str, missing_smiles: str, role: str = None) -> str | None:
    """Generate a reaction image with the specified molecule replaced by a { ? } placeholder.
    Uses strict role-based matching and canonical SMILES to handle duplicates and
    generic groups ($$label$$).  Returns a base64-encoded PNG string, or None on failure."""
    if not RDKIT_AVAILABLE:
        return None

    def rd_safe(s: str) -> str:
        """Replace $$generic_labels$$ with [*] so RDKit can parse the SMILES."""
        if not s:
            return ""
        return re.sub(r"\$\$\s*([^$]+?)\s*\$\$", "[*]", str(s))

    # Parse the reaction SMILES into its three components
    parts = reaction_smiles.split('>')
    if len(parts) != 3:
        return None
    new_r = [s.strip() for s in parts[0].split('.') if s.strip()]
    new_a = [s.strip() for s in parts[1].split('.') if s.strip()]
    new_p = [s.strip() for s in parts[2].split('.') if s.strip()]

    def replace_in_list(mol_list: list, target: str) -> bool:
        """Replace the target molecule in mol_list with a mapped wildcard atom [*:1],
        which the drawing engine renders as { ? }.
        Attempts exact text match first, then canonical SMILES comparison as fallback."""
        target_str = str(target).strip()
        # Step A: exact text match (fast; preserves $$ labels)
        for i, s in enumerate(mol_list):
            if str(s).strip() == target_str:
                mol_list[i] = "[*:1]"
                return True
        # Step B: canonical SMILES match (handles RDKit atom reordering)
        try:
            target_mol = Chem.MolFromSmiles(rd_safe(target_str))
            if target_mol:
                target_can = Chem.MolToSmiles(target_mol, True)
                for i, s in enumerate(mol_list):
                    s_mol = Chem.MolFromSmiles(rd_safe(str(s)))
                    if s_mol and Chem.MolToSmiles(s_mol, True) == target_can:
                        mol_list[i] = "[*:1]"
                        return True
        except Exception:
            pass
        return False

    # Apply role-based replacement to prevent hiding the wrong molecule
    # when the same SMILES appears in both reactants and products
    if role == "R":
        replace_in_list(new_r, missing_smiles)
    elif role == "P":
        replace_in_list(new_p, missing_smiles)
    elif role == "A":
        replace_in_list(new_a, missing_smiles)
    else:
        # Fallback for bulk upload: scan in logical order
        if not replace_in_list(new_r, missing_smiles):
            if not replace_in_list(new_a, missing_smiles):
                replace_in_list(new_p, missing_smiles)

    return draw_reaction(".".join(new_r), ".".join(new_p), ".".join(new_a))


def generate_xml(questions: list, lang: str) -> bytes:
    """Generate a pretty-printed Moodle XML file for pmatchjme reaction questions."""
    quiz = ET.Element('quiz')
    instruction_text = TEXTS[lang]["question_text"]
    prompt = f'<p>{instruction_text}</p>'

    for q in questions:
        question = ET.SubElement(quiz, 'question', type='pmatchjme')
        ET.SubElement(ET.SubElement(question, 'name'), 'text').text = q['name']

        # Embed the reaction image directly as a base64 data URI
        qtext = ET.SubElement(question, 'questiontext', format='html')
        ET.SubElement(qtext, 'text').text = (
            f'{prompt}<img src="data:image/png;base64,{q["img_base64"]}" alt="Reaction"/>'
        )

        # Accepted answer in pmatchjme format (with special characters escaped)
        escaped_smiles = (
            q['missing_smiles']
            .replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )
        answer = ET.SubElement(question, 'answer', fraction='100', format='moodle_auto_format')
        ET.SubElement(answer, 'text').text = f"match({escaped_smiles})"
        ET.SubElement(question, 'modelanswer').text = q['missing_smiles']

        # Optional feedback nodes
        if q['correct_feedback']:
            fb = ET.SubElement(ET.SubElement(answer, 'feedback', format='html'), 'text')
            fb.text = f'<p>{q["correct_feedback"]}</p>'
        if q['incorrect_feedback']:
            ans_inc = ET.SubElement(question, 'answer', fraction='0', format='moodle_auto_format')
            ET.SubElement(ans_inc, 'text').text = "*"
            fb_inc = ET.SubElement(ET.SubElement(ans_inc, 'feedback', format='html'), 'text')
            fb_inc.text = f'<p>{q["incorrect_feedback"]}</p>'
            ET.SubElement(ans_inc, 'atomcount').text = "0"

    xml_str = ET.tostring(quiz, encoding='utf-8', method='xml').decode('utf-8')
    return xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ").encode('utf-8')


def get_smiles_from_name(name: str) -> str | None:
    """Fetch a SMILES string from the NCI CIR API by compound name.
    Returns None on network failure, HTTP error, or invalid SMILES in the response."""
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        smiles = res.text.strip().split('\n')[0]
        if "ERROR" in smiles or (RDKIT_AVAILABLE and Chem.MolFromSmiles(smiles) is None):
            return None
        return smiles
    except Exception:
        return None


def clean_generic_label(s: str) -> str:
    """Strip $$ delimiters from a SMILES string, leaving only the label text.
    For example, '$$CoA$$' becomes 'CoA'."""
    if not s:
        return ""
    return re.sub(r"\$\$\s*([^$]+?)\s*\$\$", r"\1", str(s)).strip()


def name_to_smiles(name: str) -> str | None:
    """Convert a compound name or SMILES string to a validated SMILES.
    Strings containing $$ are treated as generic group labels and returned as-is.
    Returns None if resolution fails."""
    if not name or pd.isna(name):
        return None
    name = str(name).strip()
    # Strings with $$ are generic group labels — pass through unchanged
    if '$$' in name:
        return name
    # If RDKit can already parse it, treat it as a valid SMILES
    if RDKIT_AVAILABLE:
        try:
            if Chem.MolFromSmiles(name) is not None:
                return name
        except Exception:
            pass
    return get_smiles_from_name(name)


def build_reaction_smiles(reactants: list, agents: list, products: list) -> str:
    """Combine reactant, agent, and product SMILES lists into a reaction SMILES string."""
    return f"{'.'.join(reactants)}>{'.'.join(agents)}>{'.'.join(products)}"


def prepare_mol_with_labels(s: str):
    """Parse a SMILES string that may contain $$label$$ groups, replacing them
    with [*] wildcard atoms whose display label is set via atomLabel (not atomNote),
    so that RDKit renders the label text in place of the wildcard symbol.
    Returns None if RDKit is unavailable or the SMILES is invalid."""
    if not RDKIT_AVAILABLE:
        return None
    labels = re.findall(r"\$\$\s*([^$]+?)\s*\$\$", str(s))
    clean_smiles = re.sub(r"\$\$\s*([^$]+?)\s*\$\$", "[*]", str(s))
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol and labels:
        wildcard_idx = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*" and wildcard_idx < len(labels):
                atom.SetProp("atomLabel", labels[wildcard_idx])
                wildcard_idx += 1
    return mol


def start_jsme_normalization_loop():
    """Normalize all pending questions using RDKit and mark them as done.
    For molecules with generic groups ($$), the canonical SMILES of the
    wildcard-substituted structure is stored and a warning flag is set."""
    if 'reaction_questions' not in st.session_state:
        return
    for q in st.session_state.reaction_questions:
        if q.get('normalized', False):
            continue
        # Use the original un-cleaned SMILES as the normalization target
        target_smiles = str(q.get('missing_smiles_raw', q.get('missing_smiles', ""))).strip()
        if not target_smiles:
            continue
        # Detect generic groups ($$ or [R])
        has_generic = "$$" in target_smiles or "[R]" in target_smiles
        normalized_smiles = target_smiles
        try:
            # Replace $$ groups with [*] so RDKit can parse the SMILES
            rd_input = re.sub(r"\$\$\s*([^$]+?)\s*\$\$", "[*]", target_smiles)
            mol = Chem.MolFromSmiles(rd_input)
            if mol:
                normalized_smiles = Chem.MolToSmiles(mol, True)
        except Exception:
            pass  # Keep the original if normalization fails
        q['missing_smiles_normalized'] = normalized_smiles
        q['normalized'] = True
        q['generic_warning'] = has_generic
    st.rerun()


def process_bulk_file(uploaded_file):
    """Read an Excel/CSV file, convert compound names to SMILES, and append
    the resulting reaction questions to the session state."""
    if not uploaded_file:
        return
    current_lang = st.session_state.get("lang", "en")
    texts = TEXTS[current_lang]

    progress_bar = st.progress(0)
    status_text = st.empty()

    # --- Read the uploaded file ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(texts["file_read_error"].format(e))
        return

    # --- Flexible column name mapping ---
    # Accepts both English and Spanish column name variants
    col_map = {}
    possible_names = {
        'missing': ['missing_molecule', 'molécula_faltante', 'missing_name', 'molécula faltante'],
        'q_name':  ['question_name', 'nombre_pregunta', 'reaction_name', 'nombre de la pregunta'],
        'fb_pos':  ['correct_feedback', 'retroalimentación_correcta', 'retroalimentacion_correcta', 'feedback correcto'],
        'fb_neg':  ['incorrect_feedback', 'retroalimentación_incorrecta', 'retroalimentacion_incorrecta', 'feedback incorrecto'],
    }
    actual_cols = {str(c).lower().strip(): c for c in df.columns}
    for internal_key, aliases in possible_names.items():
        for alias in aliases:
            if alias.lower().strip() in actual_cols:
                col_map[internal_key] = actual_cols[alias.lower().strip()]
                break

    # --- Validate required column ---
    if 'missing' not in col_map:
        st.error(texts["missing_column_error"])
        status_text.empty()
        progress_bar.empty()
        return

    added, failed, logs = 0, 0, []

    def clean_feedback(key: str) -> str:
        """Return the feedback string for the given column key, or '' if absent/empty."""
        col = col_map.get(key)
        if col and pd.notna(row[col]):
            val = str(row[col]).strip()
            return "" if val.lower() in ["nan", "none", ""] else val
        return ""

    def process_component_list(prefix: str, current_row, is_agent: bool = False) -> list:
        """Extract and validate SMILES for all columns starting with the given prefix.
        Agent columns accept bracketed text (e.g. [heat]) and $$ generic groups."""
        parts = []
        cols = [c for c in df.columns if str(c).startswith(prefix) and c not in col_map.values()]
        for c in cols:
            val = current_row.get(c)
            if pd.isna(val) or str(val).strip() == "":
                continue
            val_str = str(val).strip()
            # Bracketed text agents and $$ generic groups are used verbatim
            if is_agent and val_str.startswith("[") and val_str.endswith("]"):
                parts.append(val_str)
            elif "$$" in val_str:
                parts.append(val_str)
            else:
                s = name_to_smiles(val_str)
                if s:
                    parts.append(s)
                elif is_agent:
                    # Wrap unresolved agent values in brackets to treat as text
                    parts.append(f"[{val_str}]")
                else:
                    parts.append(val_str)
        return parts

    # --- Main processing loop ---
    for idx, row in df.iterrows():
        row_num = idx + 2
        row_prefix = texts["bulk_row_prefix"].format(row_num)
        status_text.text(texts["processing_row"].format(row_num))
        progress_bar.progress((idx + 1) / len(df))

        # 1. Missing molecule name (required)
        missing_name_raw = row.get(col_map['missing'])
        if pd.isna(missing_name_raw) or str(missing_name_raw).strip() == '':
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_empty_missing']}")
            continue
        missing_name = str(missing_name_raw).strip()

        # 2. Question name (optional — falls back to missing molecule name)
        q_name_col = col_map.get('q_name')
        reaction_name = (
            str(row[q_name_col]).strip()
            if q_name_col and pd.notna(row[q_name_col])
            else missing_name
        )

        # 3. Optional feedback strings
        correct_feedback = clean_feedback('fb_pos')
        incorrect_feedback = clean_feedback('fb_neg')

        # 4. Process reactants, agents, and products
        reactants_smiles = process_component_list('R', row)
        agents_smiles = process_component_list('A', row, is_agent=True)
        products_smiles = process_component_list('P', row)

        if not reactants_smiles and not products_smiles:
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_no_rp']}")
            continue

        # 5. Resolve the missing molecule
        missing_input = str(row.get(col_map['missing'], '')).strip()
        target_raw = name_to_smiles(missing_input) or missing_input

        # Detect the role (R/A/P) by finding which column holds the missing value
        detected_role = None
        for col_name, col_value in row.items():
            if str(col_value).strip() == missing_input:
                first_char = str(col_name).strip().upper()[0]
                if first_char in ['R', 'A', 'P']:
                    detected_role = first_char
                    break

        # Confirm the missing molecule is actually part of the reaction
        all_molecules = reactants_smiles + agents_smiles + products_smiles
        if target_raw not in all_molecules and f"[{target_raw}]" not in all_molecules:
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_missing_not_in_reaction']}")
            continue

        target_clean = clean_generic_label(target_raw)

        # 6. Build reaction SMILES and generate image
        reaction_smiles = build_reaction_smiles(reactants_smiles, agents_smiles, products_smiles)
        img = generate_reaction_image(reaction_smiles, target_raw, role=detected_role)

        if not img:
            failed += 1
            logs.append(f"{row_prefix} {texts['bulk_error_image_failed']}")
        else:
            st.session_state.reaction_questions.append({
                'name': reaction_name,
                'reactants_smiles': reactants_smiles,
                'agents_smiles': agents_smiles,
                'products_smiles': products_smiles,
                'missing_smiles': target_clean,      # Clean version for display and Moodle
                'missing_smiles_raw': target_raw,    # Original version with $$ preserved
                'reaction_smiles': reaction_smiles,
                'role': detected_role,
                'img_base64': img,
                'correct_feedback': correct_feedback,
                'incorrect_feedback': incorrect_feedback,
                'normalized': False,
            })
            added += 1

    # --- Finalise and store messages in session state for persistent display ---
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
            'title': texts["bulk_summary_title"].format(failed),
        }
    if added > 0 or failed > 0:
        st.rerun()

# ==============================================================================
# 4. MAIN APP RENDERER
# ==============================================================================

def render_reaction_app(lang: str = None):
    """Render the full Reaction Question Generator interface."""

    # --- Session state initialisation ---
    defaults = {
        "reaction_questions": [],
        "reactants_str": "",
        "agents_str": "",
        "products_str": "",
        "jsme_normalized": False,
        "show_jsme": False,
        "search_result": None,
        "search_counter": 0,
        "widget_counter": 0,
        "r_jsme_in": None,
        "r_curr_req": None,
        "bulk_success_message": None,
        "bulk_error_logs": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # --- Language resolution ---
    if lang:
        st.session_state.lang = lang
    elif "lang" not in st.session_state:
        st.session_state.lang = "es"
    texts = TEXTS[st.session_state.lang]

    # --- Title and intro ---
    st.title(texts["title"])
    st.markdown(texts["intro"])
    st.markdown("---")

    with st.expander(texts["preview_title"]):
        st.write(texts["preview_intro"])
        st.info(f"**{texts['preview_question_react']}**")
        st.latex(r"CH_3-CH_2-Br \quad + \quad NaOH \quad \xrightarrow{\quad} \quad \text{[ ? ]}")
        st.write(texts["preview_footer_react"])

    input_col, list_col = st.columns([1.8, 1.2])

    # =========================================================================
    # INPUT COLUMN
    # =========================================================================
    with input_col:
        tab1, tab2 = st.tabs([texts["tab_manual"], texts["tab_bulk"]])

        # --- Manual entry tab ---
        with tab1:

            # Search form
            with st.form("search_form"):
                st.subheader(texts["search_title"])
                c_s1, c_s2 = st.columns([3, 1])
                s_name = c_s1.text_input(
                    texts["name_input_label"],
                    key=f"s_in_{st.session_state.search_counter}",
                    label_visibility="collapsed",
                )
                if c_s2.form_submit_button(texts["search_button"], use_container_width=True):
                    if "$$" in s_name:
                        # Strings with $$ are generic group labels — use directly as SMILES
                        st.session_state.search_result = s_name.strip()
                    else:
                        # name_to_smiles validates SMILES directly before calling NCI CIR,
                        # guarding against cases where $$ may have been stripped by the browser
                        res = name_to_smiles(s_name)
                        if res:
                            st.session_state.search_result = res
                        else:
                            st.error(texts["name_error"].format(s_name))

            # Search result display and add-to-reaction buttons
            if st.session_state.search_result:
                res = st.session_state.search_result

                mol = prepare_mol_with_labels(res)
                if mol:
                    col_info, col_img = st.columns([2, 1])
                    with col_info:
                        st.info(texts["smiles_found"].format(f"`{res}`"))
                    with col_img:
                        try:
                            st.image(draw_mol_consistent(mol), use_container_width=False, width=150)
                        except Exception:
                            st.caption(texts["mol_preview_error"])
                else:
                    st.error(texts["invalid_smiles_preview"])

                c1, c2, c3 = st.columns(3)
                if c1.button(texts["add_to_reactants"], use_container_width=True):
                    st.session_state.reactants_str = f"{st.session_state.reactants_str}, {res}".strip(", ")
                    st.session_state.search_result = None
                    st.session_state.search_counter += 1
                    st.rerun()
                if c2.button(texts["add_to_agents"], use_container_width=True):
                    st.session_state.agents_str = f"{st.session_state.agents_str}, {res}".strip(", ")
                    st.session_state.search_result = None
                    st.session_state.search_counter += 1
                    st.rerun()
                if c3.button(texts["add_to_products"], use_container_width=True):
                    st.session_state.products_str = f"{st.session_state.products_str}, {res}".strip(", ")
                    st.session_state.search_result = None
                    st.session_state.search_counter += 1
                    st.rerun()

            st.write("---")

            # Reaction builder: three text areas for reactants, agents, products
            col_r, col_a, col_p = st.columns(3)
            r_val = col_r.text_area(texts["reactants_label"], value=st.session_state.reactants_str, height=100)
            a_val = col_a.text_area(texts["agents_label"], value=st.session_state.agents_str, height=100)
            p_val = col_p.text_area(texts["products_label"], value=st.session_state.products_str, height=100)

            # Persist user edits across reruns
            st.session_state.reactants_str = r_val
            st.session_state.agents_str = a_val
            st.session_state.products_str = p_val

            r_list = [s.strip() for s in r_val.split(',') if s.strip()]
            a_list = [s.strip() for s in a_val.split(',') if s.strip()]
            p_list = [s.strip() for s in p_val.split(',') if s.strip()]
            all_mols = r_list + p_list

            col_m, col_n = st.columns(2)
            missing_idx = col_m.selectbox(
                texts["select_missing"],
                range(len(all_mols)),
                format_func=lambda x: f"{all_mols[x]} ({'R' if x < len(r_list) else 'P'})",
            ) if all_mols else None

            q_name = col_n.text_input(
                texts["reaction_name_label"],
                value=f"Reaction {len(st.session_state.reaction_questions) + 1}",
            )

            cb1, cb2 = st.columns(2)
            if cb1.button(texts["add_reaction_button"], type="primary", use_container_width=True):
                if q_name and missing_idx is not None:
                    missing_smiles_raw = all_mols[missing_idx]

                    # Determine the role from the selectbox label: (R) or (P)
                    role_label = f"{'R' if missing_idx < len(r_list) else 'P'}"
                    role = role_label if role_label in ('R', 'P') else None

                    missing_smiles_clean = clean_generic_label(missing_smiles_raw)
                    rxn_smiles = f"{'.'.join(r_list)}>{'.'.join(a_list)}>{'.'.join(p_list)}"
                    img = generate_reaction_image(rxn_smiles, missing_smiles_raw, role=role)

                    if img:
                        st.session_state.reaction_questions.append({
                            'name': q_name,
                            'missing_smiles': missing_smiles_clean,
                            'missing_smiles_raw': missing_smiles_raw,
                            'img_base64': img,
                            'reactants_smiles': list(r_list),
                            'agents_smiles': list(a_list),
                            'products_smiles': list(p_list),
                            'reaction_smiles': rxn_smiles,
                            'role': role,
                            'correct_feedback': 'Correct!',
                            'incorrect_feedback': '',
                            'normalized': False,
                        })
                        st.rerun()
                    else:
                        st.error(texts["image_warning"])

            if cb2.button(texts["new_question"], use_container_width=True):
                st.session_state.reactants_str = ""
                st.session_state.agents_str = ""
                st.session_state.products_str = ""
                st.rerun()

        # --- Bulk upload tab ---
        with tab2:
            st.markdown(texts["bulk_info"])
            uploaded = st.file_uploader(
                texts["upload_file_label"], type=['xlsx', 'csv'], key="bulk_uploader_input"
            )
            if uploaded and st.button(
                texts["process_bulk_button"], type="primary",
                use_container_width=True, key="bulk_proc_btn"
            ):
                process_bulk_file(uploaded)
                st.session_state.jsme_normalized = False
                st.rerun()

            # Persistent status messages set by process_bulk_file
            if st.session_state.get('bulk_success_message'):
                st.success(st.session_state.bulk_success_message)
            if st.session_state.get('bulk_error_logs'):
                with st.expander(st.session_state.bulk_error_logs['title']):
                    for log in st.session_state.bulk_error_logs['logs']:
                        st.caption(log)

    # =========================================================================
    # QUESTION LIST COLUMN
    # =========================================================================
    with list_col:
        st.subheader(texts["added_questions_title"])

        if not st.session_state.reaction_questions:
            st.info(texts["no_questions"])
        else:
            pending_questions = [
                i for i, q in enumerate(st.session_state.reaction_questions)
                if not q.get('normalized', False)
            ]

            # --- Invisible JSME editor for sequential SMILES normalisation ---
            jsme_result = jsme_editor(
                smiles_json=st.session_state.r_jsme_in,
                key="reaction_jsme_global_processor",
            )

            # Process the editor response when it arrives
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
                            st.success(texts["normalized_success"].format(idx + 1, old_smiles, new_smiles))
                        else:
                            st.warning(texts["no_valid_smiles_warning"].format(idx + 1))
                        # Advance to the next pending question
                        st.session_state.r_jsme_in = None
                        st.session_state.r_curr_req = None
                        start_jsme_normalization_loop()
                        st.rerun()
                except json.JSONDecodeError:
                    st.error(texts["json_decode_error"])
                except Exception as e:
                    st.error(texts["jsme_processing_error"].format(str(e)))

            # --- Normalize / Download button ---
            if pending_questions:
                num_pending = len(pending_questions)
                btn_text = (
                    texts["continue_standardization"].format(num_pending)
                    if st.session_state.r_jsme_in
                    else texts["normalize_button"].format(num_pending)
                )
                if st.button(
                    btn_text,
                    type="primary",
                    use_container_width=True,
                    key="btn_start_normalize_reactions",
                    help=texts["continue_standardization_help"].format(num_pending),
                ):
                    start_jsme_normalization_loop()
                    st.rerun()
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
                        icon=":material/download:",
                    )
                except Exception as e:
                    st.error(texts["xml_error"].format(str(e)))

            # --- Clear all button ---
            if st.button(
                texts["clear_all_button"],
                use_container_width=True,
                icon=":material/delete:",
                key="btn_clear_all_reactions",
            ):
                st.session_state.reaction_questions = []
                st.session_state.r_jsme_in = None
                st.session_state.r_curr_req = None
                st.session_state.jsme_normalized = False
                st.rerun()

            st.markdown("---")

            # --- List of added questions ---
            for i, q in enumerate(st.session_state.reaction_questions):
                is_normalized = q.get('normalized', False)
                has_generic_warning = q.get('generic_warning', False)

                with st.container(border=True):
                    cols = st.columns([5, 1])
                    with cols[0]:
                        # Normalisation status indicator
                        if is_normalized:
                            st.success(texts["is_normalized"])
                            if has_generic_warning:
                                st.warning(texts["normalization_error"])
                        else:
                            st.warning(texts["is_pending"])

                        st.markdown(f"**{i + 1}. {q['name']}**")

                        # Reaction image (decoded from stored base64)
                        st.image(
                            io.BytesIO(base64.b64decode(q['img_base64'])),
                            use_container_width=True,
                        )
                        st.caption(f"SMILES: `{q['missing_smiles']}`")

                    with cols[1]:
                        if st.button("🗑️", key=f"del_reaction_{i}"):
                            st.session_state.reaction_questions.pop(i)
                            # Reset JSME state if the deleted question was being processed
                            if (
                                st.session_state.r_curr_req
                                and st.session_state.r_curr_req.get('index') == i
                            ):
                                st.session_state.r_jsme_in = None
                                st.session_state.r_curr_req = None
                            st.rerun()

# ==============================================================================
# 5. STANDALONE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Default language
    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    # Language selector in the sidebar
    with st.sidebar:
        lang = st.selectbox(
            "Idioma / Language",
            options=["es", "en"],
            format_func=lambda x: "Español" if x == "es" else "English",
            index=0 if st.session_state.lang == "es" else 1,
            key="lang_selector",
        )
        if lang != st.session_state.lang:
            st.session_state.lang = lang
            st.rerun()

    st.set_page_config(
        layout="wide",
        page_title=TEXTS[st.session_state.lang]["title"],
        initial_sidebar_state="expanded",
    )

    render_reaction_app(st.session_state.lang)
