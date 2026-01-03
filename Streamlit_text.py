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

# ============================================================
# PART 1 — MULTI-SYSTEM INPUT (UI only)
# ============================================================
st.markdown("---")
st.header("1) Select coating system(s)")

SYSTEMS = {
    "SBMA + PDMS": "SBMA_PDMS",
    "PDMS + PEG": "PDMS_PEG",
    "PEG + PMHS": "PEG_PMHS",
}

mode = st.radio(
    "What do you want to evaluate?",
    ["Only one system", "All systems"],
    horizontal=True
)

if mode == "Only one system":
    chosen_label = st.selectbox("Select the system:", list(SYSTEMS.keys()))
    systems_to_show = [chosen_label]
else:
    systems_to_show = list(SYSTEMS.keys())

def system_expander(label: str):
    sys_code = SYSTEMS[label]
    with st.expander(f"System: {label}", expanded=(mode == "Only one system")):
        c1, c2 = st.columns(2)

        with c1:
            mw_a = st.number_input(
                f"[{label}] MW component A",
                min_value=0.0, value=1000.0, step=10.0,
                key=f"{sys_code}_mw_a"
            )
            perc_a = st.number_input(
                f"[{label}] % component A",
                min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                key=f"{sys_code}_perc_a"
            )

        with c2:
            mw_b = st.number_input(
                f"[{label}] MW component B",
                min_value=0.0, value=1000.0, step=10.0,
                key=f"{sys_code}_mw_b"
            )
            perc_b = st.number_input(
                f"[{label}] % component B",
                min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                key=f"{sys_code}_perc_b"
            )

        if (perc_a + perc_b) == 0:
            st.error("Percentages cannot both be zero.")
            return None

        st.caption("Next step: we will connect this input to your descriptor-based model.")

        return {
            "system": sys_code,
            "mw_a": mw_a, "perc_a": perc_a,
            "mw_b": mw_b, "perc_b": perc_b,
        }

user_requests = []
for label in systems_to_show:
    out = system_expander(label)
    if out is not None:
        user_requests.append(out)

st.subheader("Current inputs (debug)")
st.dataframe(user_requests)
