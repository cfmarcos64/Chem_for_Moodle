# -*- coding: utf-8 -*-
"""
Moodle Question Generator for Skeletal Structures from Fischer Projections.
Integrated Version for Chem_To_Moodle Suite
@author: Carlos Fernandez Marcos
"""

import streamlit as st
import xml.etree.ElementTree as ET
import pandas as pd
import pubchempy as pcp
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import re
import io
import base64
import json
import uuid
import numpy as np
from my_component import jsme_editor 

# ==============================================================================
# 1. TEXTOS BILINGÜES
# ==============================================================================

TEXTS = {
    "es": {
        "add_to_list": "Añadir a la lista",
        "bulk_note": "Sube un archivo .csv o .xlsx con una columna **'name'**. Se añadirán a la lista para normalizar.",
        "bulk_success": "✅ Se han añadido {0} moléculas a la lista.",
        "change_language": "Change language to English:",
        "clear_all": "Borrar todas las preguntas",
        "column_error": "🚨 El archivo debe contener la columna 'name'.",
        "custom_question_name": "Etiqueta (ej. D-Glucosa):",
        "download_xml": "Descargar XML de Moodle",
        "error_cyclic": "🚨 No se puede generar una proyección de Fischer de una molécula cíclica.",
        "error_cyclic_sugar": "⚠️ La molécula detectada es cíclica. Se ha intentado convertir a su forma abierta para la proyección.",
        "error_not_found": "🚨 No se pudo encontrar el SMILES para '{0}'. Revisa el nombre.",
        "intro": "Introduce el nombre o SMILES de un compuesto para generar preguntas en las que se muestra una proyección de Fischer y el alumno debe dibujar la estructura lineoangular correspondiente utilizando el editor JSME.",
        "invalid_mol": "🚨 Estructura química inválida.",
        "jsme_processing_error": "Error procesando respuesta JSME: {0}",
        "jsme_status": "Normalizando:",
        "json_decode_error": "Error al interpretar la respuesta del editor JSME (formato JSON inválido)",
        "molecule_name": "Nombre o SMILES:",
        "no_questions": "Aún no hay preguntas añadidas.",
        "no_valid_smiles_warning": "No se recibió SMILES válido para la pregunta {0}",
        "normalized_success": "Pregunta {0} normalizada: {1} → {2}",
        "not_normalized": "⚠️ Pendiente de normalizar",
        "preview_title": "👁️ Ver ejemplo de pregunta en Moodle",
        "preview_intro": "El alumno verá algo como esto:",
        "preview_question_fischer": "Dibuja con la estereoquímica correspondiente la estructura lineoangular del compuesto que se representa como proyección de Fischer:",
        "preview_footer_fischer": "➡️ El alumno deberá responder dibujando la molécula en formato lineoangular en el editor JSME.",
        "question_text_moodle": "Dibuja con la estereoquímica correspondiente la estructura lineoangular del compuesto que se representa como proyección de Fischer:",
        "questions_added_subtitle": "Preguntas Añadidas",
        "section_bulk": "Carga Masiva",
        "section_individual": "Entrada Individual",
        "start_bulk": "Procesar archivo",
        "start_standardization": "Estandarizar {0} molécula(s) pendiente(s)",
        "stereo_warning": "⚠️ Aviso: La estereoquímica de algunos centros quirales no está definida.",
        "stereo_warning_moodle": "⚠️ Nota: Los enlaces ondulados indican estereoquímica no definida en la proyección.",
        "title": ":material/rebase_edit: Generador de Proyecciones de Fischer para Moodle"
    },
    "en": {
        "add_to_list": "Add to list",
        "bulk_note": "Upload a .csv or .xlsx with a **'name'** column. They will be added to the list for normalization.",
        "bulk_success": "✅ {0} molecules added to the list.",
        "change_language": "Cambiar idioma a Español:",
        "clear_all": "Clear all questions",
        "column_error": "🚨 File must contain a 'name' column.",
        "custom_question_name": "Label (e.g. D-Glucose):",
        "download_xml": "Download Moodle XML",
        "error_cyclic": "🚨 Cannot generate a Fischer projection for a cyclic molecule.",
        "error_cyclic_sugar": "⚠️ The detected molecule is cyclic. Attempted to convert to its open-chain form for projection.",
        "error_not_found": "🚨 Could not find SMILES for '{0}'. Please check the name.",
        "intro": "Enter the name or SMILES of a compound to generate questions where a Fischer projection is shown and the student must draw the corresponding skeletal structure using the JSME editor.",
        "invalid_mol": "🚨 Invalid chemical structure.",
        "jsme_processing_error": "Error processing JSME response: {0}",
        "jsme_status": "Normalizing:",
        "json_decode_error": "Error interpreting JSME editor response (invalid JSON format)",
        "molecule_name": "Name or SMILES:",
        "no_questions": "No questions added yet.",
        "no_valid_smiles_warning": "No valid SMILES received for question {0}",
        "normalized_success": "Question {0} normalized: {1} → {2}",
        "not_normalized": "⚠️ Pending normalization",
        "preview_title": "👁️ View sample Moodle question",
        "preview_intro": "The student will see something like this:",
        "preview_question_fischer": "Draw the skeletal structure with the corresponding stereochemistry for the compound represented as a Fischer projection:",
        "preview_footer_fischer": "➡️ The student must answer drawing the correct structural formula in JSME editor.",
        "question_text_moodle": "Draw the skeletal structure with the corresponding stereochemistry for the compound represented as a Fischer projection:",
        "questions_added_subtitle": "Added Questions",
        "section_bulk": "Bulk Upload",
        "section_individual": "Individual Entry",
        "start_bulk": "Process file",
        "start_standardization": "Standardize {0} pending molecule(s)",
        "stereo_warning": "⚠️ Note: Some centers have undefined stereochemistry.",
        "stereo_warning_moodle": "⚠️ Note: Wavy bonds indicate undefined stereochemistry.",
        "title": ":material/rebase_edit: Fischer Projection Generator for Moodle"
    }
}

# ==============================================================================
# 2. HELPER FUNCTIONS (Chemical logic)
# ==============================================================================

def get_mol_and_smiles(user_input, texts):
    if not user_input or str(user_input).strip() == "": return None, None
    name = str(user_input).strip()
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            smiles = response.text.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.GetRingInfo().NumRings() == 0: return mol, smiles
    except: pass
    mol_pc, sm_pc = None, None
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            sm_pc = results[0].smiles
            mol_pc = Chem.MolFromSmiles(sm_pc)
            if mol_pc and mol_pc.GetRingInfo().NumRings() == 0: return mol_pc, sm_pc
    except: pass
    if mol_pc and mol_pc.GetRingInfo().NumRings() > 0:
        try:
            open_results = pcp.get_compounds(f"open chain {name}", 'name')
            if open_results:
                sm_open = open_results[0].smiles
                mol_open = Chem.MolFromSmiles(sm_open)
                if mol_open and mol_open.GetRingInfo().NumRings() == 0:
                    st.session_state.f_error_msg = texts["error_cyclic_sugar"]
                    return mol_open, sm_open
        except: pass
        st.session_state.f_error_msg = texts["error_cyclic"]
        return None, None
    st.session_state.f_error_msg = texts["error_not_found"].format(name)
    return None, None

def format_subscripts(text):
    if not text or text.strip() == "": return ""
    text = text.replace("_", "")
    if len(text) == 1: return f"${text}$"
    res = re.sub(r'(\d+)', r'_{\1}', text)
    return f"${res}$"

def get_condensed_label(mol, start_idx, central_idx):
    atom = mol.GetAtomWithIdx(start_idx)
    symbol = atom.GetSymbol()
    valencia = {'C': 4, 'N': 3, 'O': 2, 'S': 2, 'H': 1}
    bonds = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds() if b.GetOtherAtom(atom).GetSymbol() != 'H')
    h_count = max(0, valencia.get(symbol, 0) - bonds)
    neighbors = [nb for nb in atom.GetNeighbors() if nb.GetIdx() != central_idx]
    has_c_neighbor = any(nb.GetSymbol() == 'C' for nb in neighbors)
    if symbol == 'N':
        if has_c_neighbor: return "NHCH3" if h_count >= 1 else "NCH3"
        return f"NH{h_count}" if h_count > 1 else ("NH" if h_count == 1 else "N")
    if symbol == 'O':
        if has_c_neighbor: return "OCH3"
        return "OH" if h_count > 0 else "O"
    if symbol == 'C':
        o_nb = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() == 'O']
        if len(o_nb) == 2: return "COOH"
        if len(o_nb) == 1:
            is_double = any(mol.GetBondBetweenAtoms(start_idx, x.GetIdx()).GetBondTypeAsDouble() == 2 for x in o_nb)
            return "CHO" if is_double else "CH2OH"
        return "CH3" if h_count == 3 else (f"CH{h_count}" if h_count > 0 else "C")
    return symbol

def generate_fischer_base64(mol):
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    chiral_centers = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    c_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C']
    def get_c_paths(start, path):
        paths = [path]
        for nb in mol.GetAtomWithIdx(start).GetNeighbors():
            if nb.GetSymbol() == 'C' and nb.GetIdx() not in path:
                paths.extend(get_c_paths(nb.GetIdx(), path + [nb.GetIdx()]))
        return paths
    all_paths = []
    for c in c_indices: all_paths.extend(get_c_paths(c, [c]))
    if not all_paths: return "", False
    backbone = list(max(all_paths, key=len))
    def get_c_double_o_pos(chain):
        for i, idx in enumerate(chain):
            at = mol.GetAtomWithIdx(idx)
            for b in at.GetBonds():
                if b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(at).GetSymbol() == 'O': return i
        return 999
    if get_c_double_o_pos(backbone[::-1]) < get_c_double_o_pos(backbone): backbone.reverse()
    internal = backbone[1:-1]
    n = len(internal)
    v_step, h_len, tail_len = 0.5, 0.7, 0.3
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.plot([0, 0], [1 * v_step - tail_len, n * v_step + tail_len], color='black', lw=1.5, zorder=1)
    inv_map = {"OH": "HO", "NH2": "H2N", "NH": "HN", "SH": "HS", "CH3": "H3C", "CH2OH": "HOCH2", "CHO": "OHC", "COOH": "HOOC", "O": "O"}
    has_undefined = False
    for i, idx in enumerate(internal):
        y = (n - i) * v_step
        atom = mol.GetAtomWithIdx(idx)
        config = chiral_centers.get(idx)
        is_chiral = idx in chiral_centers
        is_undef = is_chiral and (config == '?' or config is None)
        laterals = []
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in backbone:
                bond = mol.GetBondBetweenAtoms(idx, nb.GetIdx())
                laterals.append({'idx': nb.GetIdx(), 'symbol': nb.GetSymbol(), 'order': bond.GetBondTypeAsDouble(), 'rank': int(nb.GetProp('_CIPRank')) if nb.HasProp('_CIPRank') else 0})
        laterals.sort(key=lambda x: x['rank'], reverse=True)
        has_double_bond = any(l['order'] == 2 for l in laterals)
        if has_double_bond:
            offset = 0.03
            ax.plot([0, h_len], [y + offset, y + offset], color='black', lw=1.2)
            ax.plot([0, h_len], [y - offset, y - offset], color='black', lw=1.2)
            ax.text(h_len + 0.05, y, "O", ha='left', va='center', fontsize=10)
        elif is_undef:
            has_undefined = True
            xs = np.linspace(-h_len, h_len, 50)
            ys = y + 0.02 * np.sin(xs * 30)
            ax.plot(xs, ys, color='black', lw=1.2)
            for side, l_idx in [(-1, 0), (1, 1)]:
                if l_idx < len(laterals):
                    lbl = get_condensed_label(mol, laterals[l_idx]['idx'], idx)
                    ax.text(side*(h_len+0.05), y, format_subscripts(inv_map.get(lbl, lbl[::-1]) if side < 0 else lbl), ha='right' if side < 0 else 'left', va='center', fontsize=10)
        else:
            ax.plot([-h_len, h_len], [y, y], color='black', lw=1.5)
            if len(laterals) >= 1:
                l_idx, r_idx = (1, 0) if (len(laterals)>1 and config == 'R') else (0, 1)
                if l_idx < len(laterals):
                    lbl_l = get_condensed_label(mol, laterals[l_idx]['idx'], idx)
                    ax.text(-(h_len + 0.05), y, format_subscripts(inv_map.get(lbl_l, lbl_l[::-1])), ha='right', va='center', fontsize=10)
                if r_idx < len(laterals):
                    lbl_r = get_condensed_label(mol, laterals[r_idx]['idx'], idx)
                    ax.text(h_len + 0.05, y, format_subscripts(lbl_r), ha='left', va='center', fontsize=10)
        ax.scatter(0, y, color='black', s=10, zorder=3)
    ax.text(-0.05, n * v_step + tail_len + 0.1, format_subscripts(get_condensed_label(mol, backbone[0], backbone[1])), ha='left', va='bottom', fontsize=10)
    ax.text(-0.05, 1 * v_step - tail_len - 0.1, format_subscripts(get_condensed_label(mol, backbone[-1], backbone[-2])), ha='left', va='top', fontsize=10)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.5, (n + 1) * v_step + 0.5); ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), has_undefined

def generate_xml_local(questions, texts):
    quiz = ET.Element("quiz")
    
    for q in questions:
        qu = ET.SubElement(quiz, "question", type="pmatchjme")
        
        # Nombre de la pregunta
        ET.SubElement(ET.SubElement(qu, "name"), "text").text = q['name']
        
        # Texto de la pregunta con imagen y posible aviso de estereoquímica
        w_txt = f"<p style='color:red;'><i>{texts['stereo_warning_moodle']}</i></p>" if q.get('has_undefined', False) else ""
        txt = (
            f"<p>{texts['question_text_moodle']}</p>"
            f"{w_txt}"
            f"<p style='text-align:center;'><img src='data:image/png;base64,{q['img']}' width='180'></p>"
        )
        ET.SubElement(ET.SubElement(qu, "questiontext", format="html"), "text").text = txt
        
        # Respuesta correcta
        ans = ET.SubElement(qu, "answer", fraction="100")
        esc = q['smiles_normalized'].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("[", "\\[").replace("]", "\\]")
        ET.SubElement(ans, "text").text = f"match({esc})"
        
        # Model answer (para referencia)
        ET.SubElement(qu, "modelanswer").text = q['smiles_normalized']
    
    return ET.tostring(quiz, encoding="utf-8", xml_declaration=True)

# ==============================================================================
# 3. MAIN FUNCTION
# ==============================================================================

def render_fischer_app(lang="en"):
    """Renders Fischer generator interface."""
    
    if "f_questions" not in st.session_state: st.session_state.f_questions = []
    if "f_lang" not in st.session_state: st.session_state.f_lang = "es"
    if "f_is_processing" not in st.session_state: st.session_state.f_is_processing = False
    if "f_error_msg" not in st.session_state: st.session_state.f_error_msg = None

    texts = TEXTS[lang]

    # --- Title and Intro ---
   
    st.title(texts["title"])
    st.markdown(texts["intro"])
    st.markdown("---")
    
    # Define SMILES and projection as example (D-Glucose)
    example_smiles = "C([C@@H]([C@@H]([C@H]([C@@H](C=O)O)O)O)O)O"    
    example_mol = Chem.MolFromSmiles(example_smiles)
    if example_mol:
        img_base64, _ = generate_fischer_base64(example_mol)
        with st.expander(texts["preview_title"]):
            st.write(texts["preview_intro"])
            st.info(f"**{texts['preview_question_fischer']}**")
                
            # Show app-generated image
            st.image(f"data:image/png;base64,{img_base64}", width=180)
                
            st.write(texts["preview_footer_fischer"])
    
    # JSME Logic
    # Session states
    if "f_questions" not in st.session_state: st.session_state.f_questions = []
    if "f_jsme_in" not in st.session_state: st.session_state.f_jsme_in = None
    if "f_curr_req" not in st.session_state: st.session_state.f_curr_req = None
    
    def start_jsme_loop():
        idx = next(
            (i for i, q in enumerate(st.session_state.f_questions)
             if q.get('smiles_normalized') is None),
            None
        )
        if idx is not None:
            req_id = str(uuid.uuid4())
            q = st.session_state.f_questions[idx]
            st.session_state.f_jsme_in = json.dumps({
                'smiles': q['smiles_raw'],
                'id': req_id
            })
            st.session_state.f_curr_req = {
                'id': req_id,
                'index': idx
            }
        else:
            st.session_state.f_jsme_in = None
            st.session_state.f_curr_req = None
    
    
    # Unique JSME editor
    jsme_out = jsme_editor(
        smiles_json=st.session_state.get("f_jsme_in"),
        key="f_jsme_processor"   # ← clave única para esta app
    )
    
    # Processing the editor answer
    if st.session_state.get("f_jsme_in") and jsme_out:
        try:
            res = json.loads(jsme_out)
            if (
                st.session_state.f_curr_req and
                res.get('id') == st.session_state.f_curr_req.get('id')
            ):
                idx = st.session_state.f_curr_req['index']
                new_smiles = res.get('smiles', '').strip()
    
                if new_smiles:
                    old_smiles = st.session_state.f_questions[idx]['smiles_raw']
                    st.session_state.f_questions[idx]['smiles_normalized'] = new_smiles
                    st.success(
                        texts["normalized_success"].format(
                            idx + 1,
                            old_smiles,
                            new_smiles
                        )
                    )
                else:
                    st.warning(
                        texts["no_valid_smiles_warning"].format(idx + 1)
                    )
    
                # Clean and go to the next pending question (if exists)
                st.session_state.f_jsme_in = None
                st.session_state.f_curr_req = None
                start_jsme_loop()
                st.rerun()
    
        except json.JSONDecodeError:
            st.error(texts["json_decode_error"])
    
        except Exception as e:
            st.error(texts["jsme_processing_error"].format(str(e)))

    # --- Column Layout ---
    main_c, list_c = st.columns([1, 1.2])

    with main_c:
        if st.session_state.f_error_msg:
            if "🚨" in st.session_state.f_error_msg:
                st.error(st.session_state.f_error_msg)
            else:
                st.warning(st.session_state.f_error_msg)
                
        t_ind, t_bulk = st.tabs([texts["section_individual"], texts["section_bulk"]])
        
        with t_ind:
            def add_f_ind():
                u_input = st.session_state.f_in_mol
                u_label = st.session_state.f_in_lbl
                
                if not u_input.strip():
                    return
                
                mol, s = get_mol_and_smiles(u_input, texts)
                
                if mol:                    
                    st.session_state.f_error_msg = None # Solo limpiamos si hubo éxito
                    img, und = generate_fischer_base64(mol)
                    st.session_state.f_questions.append({
                        'name': u_label if u_label else u_input, 
                        'smiles_raw': s, 
                        'smiles_normalized': None, 
                        'img': img, 
                        'has_undefined': und
                    })

            # 2. Create FORM (this blocks automatic on_change)
            with st.form(key="fischer_form", clear_on_submit=True):
                st.text_input(texts["molecule_name"], key="f_in_mol")
                st.text_input(texts["custom_question_name"], key="f_in_lbl")

                submit = st.form_submit_button(
                    texts["add_to_list"], 
                    type="primary", 
                    icon=":material/add_task:",
                    use_container_width=True
                )
                
                if submit:
                    add_f_ind()
                    st.rerun()            

        with t_bulk:
            st.info(texts["bulk_note"])
            f = st.file_uploader("", type=['csv', 'xlsx'], key="f_up")
            if st.button(texts["start_bulk"], disabled=not f, icon=":material/upload_file:"):
                try:
                    # Archieve reading
                    df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
                    
                    # Flexible column detection
                    has_name_en = 'name' in df.columns
                    has_name_es = 'nombre' in df.columns

                    if has_name_en or has_name_es:
                        # Search priority: 'name' > 'nombre'
                        col_for_api = 'name' if has_name_en else 'nombre'
                        
                        for _, row in df.iterrows():
                            val = str(row[col_for_api]).strip()
                            
                            if val and val.lower() != 'nan':
                                m, s = get_mol_and_smiles(val, texts)
                                if m:
                                    img, und = generate_fischer_base64(m)
                                    
                                    # Label: Priority 'nombre' (Spanish)
                                    label = str(row.get('nombre', row.get('name', val)))
                                    
                                    st.session_state.f_questions.append({
                                        'name': label, 
                                        'smiles_raw': s, 
                                        'smiles_normalized': None, 
                                        'img': img, 
                                        'has_undefined': und
                                    })
                        st.rerun()
                    else:
                        st.error(texts["column_error"])
                        
                except Exception as e: 
                    st.error(f"Error: {e}")

    with list_c:
        st.subheader(texts["questions_added_subtitle"])
        
        qs = st.session_state.f_questions
        
        # Recalculate pending each render (button dynamic update)
        pending = [i for i, q in enumerate(qs) if q.get('smiles_normalized') is None]
        pending_count = len(pending)
        
        if qs:
            # Normalization button
            if pending_count > 0:
                btn_label = texts["start_standardization"].format(pending_count)
                if st.button(
                    btn_label,
                    type="primary",
                    icon=":material/rocket_launch:",
                    use_container_width=True
                ):
                    start_jsme_loop()
                    st.rerun()
            else:
                # All normalized → show Download button
                xml_data = generate_xml_local(qs, texts)
                st.download_button(
                    label=texts["download_xml"],
                    data=xml_data,
                    file_name="fischer_moodle.xml",
                    mime="application/xml",
                    type="primary",
                    icon=":material/download:",
                    use_container_width=True
                )
            
            # Clear all button
            if st.button(
                texts["clear_all"],
                icon="🗑️",
                use_container_width=True
            ):
                st.session_state.f_questions = []
                st.session_state.f_error_msg = None
                st.rerun()
            
            st.divider()
            
            # Question list
            for i, q in enumerate(qs):
                is_normalized = q.get('smiles_normalized') is not None
                
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1.5, 3, 0.5])
                    with c1:
                        st.image(f"data:image/png;base64,{q['img']}", width=150)
                    with c2:
                        st.markdown(f"**{i+1}. {q['name']}**")
                        if q.get('has_undefined', False):
                            st.warning(texts["stereo_warning"])
                        status = "✅ Normalizada" if is_normalized else texts["not_normalized"]
                        smiles_shown = q['smiles_normalized'] if is_normalized else q['smiles_raw']
                        st.caption(f"{status} – SMILES: `{smiles_shown}`")
                    with c3:
                        if st.button("🗑️", key=f"f_del_{i}", help="Eliminar pregunta"):
                            st.session_state.f_questions.pop(i)
                            st.rerun()
                    st.divider()
        
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
    render_fischer_app(st.session_state.lang)