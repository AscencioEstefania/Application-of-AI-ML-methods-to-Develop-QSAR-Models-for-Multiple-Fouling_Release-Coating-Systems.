import streamlit as st

# ============================================================
# Page configuration (must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Fouling Release Predictor",
    layout="wide"
)

# ============================================================
# Header image
# ============================================================
st.image("fouling_boat.png", use_container_width=True)

st.title("Fouling Release Predictor")

st.write(" The app is running. Next step: add the system selector and inputs.")
