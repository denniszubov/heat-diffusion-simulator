import streamlit as st
from typing import Any

from core.initial_condition import INITIAL_CONDITIONS_FACTORY


def initial_condition_ui() -> tuple[str, dict[str, Any]]:
    """
    Renders the initial condition selection UI and parameter controls.
    
    Returns:
        tuple: (selected_condition_name, parameters_dict)
    """
    ic_choice = st.sidebar.selectbox(
        "Initial condition", 
        list(INITIAL_CONDITIONS_FACTORY.keys()), 
        index=0
    )

    ic_params = {}
    if ic_choice == "Sine Wave":
        st.sidebar.markdown("#### Initial Condition Parameters")
        ic_params['frequency'] = st.sidebar.slider(
            "Frequency", 0.5, 6.0, 2.0, 0.5,
            help="Number of complete waves across the domain"
        )
        ic_params['amplitude'] = st.sidebar.slider(
            "Amplitude (°C)", 10.0, 100.0, 50.0, 5.0,
            help="Sine wave amplitude"
        )
    elif ic_choice == "Square Wave":
        st.sidebar.markdown("#### Initial Condition Parameters")
        ic_params['steps'] = st.sidebar.slider(
            "Steps", 1, 10, 5, 1,
            help="Number of square wave steps across the domain"
        )
        ic_params['amplitude'] = st.sidebar.slider(
            "Amplitude (°C)", 10.0, 100.0, 50.0, 5.0,
            help="Wave amplitude"
        )

    return ic_choice, ic_params
