import os
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "jsme_editor",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("jsme_editor", path=build_dir)


def jsme_editor(smiles_json, key=None):
    """
    Crea una instancia del editor molecular JSME.
    
    Parameters
    ----------
    smiles_json: str
        El JSON string que contiene el SMILES y el ID de la solicitud
        para el procesamiento asíncrono.
    key: str or None
        Clave única para identificar el componente
    
    Returns
    -------
    str
        El código SMILES resultante devuelto por el componente React.
    """
    # CRUCIAL: Se pasa el argumento 'smiles_json' como keyword argument al componente
    component_value = _component_func(smiles_json=smiles_json, key=key, default="")
    
    return component_value
