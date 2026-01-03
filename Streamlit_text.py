import streamlit as st
import pandas as pd
import numpy as np

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

# ============================================================
# DATA HELPERS (pure descriptors)
# ============================================================
@st.cache_data
def load_pure_descriptors(path="pure_descriptors.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    if "Component" not in df.columns:
        raise ValueError("pure_descriptors.csv must include a column named 'Component'.")
    df["Component"] = df["Component"].astype(str).str.strip().str.upper()
    return df

def get_pure_row(df_pure, component_name: str):
    comp = component_name.strip().upper()
    row = df_pure[df_pure["Component"] == comp]
    if row.empty:
        raise ValueError(f"Component not found in pure_descriptors.csv: {component_name}")
    return row.iloc[0]

def validate_range(name: str, mw_value: float, mw_min: float, mw_max: float):
    mw = float(mw_value)
    if not (mw_min <= mw <= mw_max):
        raise ValueError(
            f"{name} MW must be between {mw_min} and {mw_max}. You entered {mw_value}."
        )

# ============================================================
# MIX DESCRIPTORS — System 1 rule (SBMA + PDMS)
# fraction = monomerMW / userMW
# mix = SBMA * fraction * % + PDMS * fraction * %
# ============================================================
SBMA_MONOMER_MW = 279.3566
PDMS_MONOMER_MW = 74.1535

SBMA_MW_RANGE = (500.0, 2500.0)
PDMS_MW_RANGE = (1000.0, 5000.0)

def build_mix_system1(row_sbma, row_pdms, feature_cols, mw_sbma, perc_sbma, mw_pdms, perc_pdms):
    p_sbma = float(perc_sbma) / 100.0
    p_pdms = float(perc_pdms) / 100.0

    f_sbma = float(SBMA_MONOMER_MW) / float(mw_sbma)
    f_pdms = float(PDMS_MONOMER_MW) / float(mw_pdms)

    vec_sbma = row_sbma[feature_cols].astype(float).to_numpy()
    vec_pdms = row_pdms[feature_cols].astype(float).to_numpy()

    mix = (vec_sbma * f_sbma * p_sbma) + (vec_pdms * f_pdms * p_pdms)
    return mix.reshape(1, -1), (f_sbma, f_pdms)

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

# ============================================================
# PART 2 — SYSTEM 1 TEST (SBMA + PDMS) — BUILD MIX DESCRIPTORS
# ============================================================
st.markdown("---")
st.header("2) System 1 test — SBMA + PDMS (mix descriptors)")

# load pure descriptors
try:
    df_pure = load_pure_descriptors("pure_descriptors.csv")
except Exception as e:
    st.error(f"Could not load pure_descriptors.csv: {e}")
    st.stop()

# features: all columns except Component
feature_cols = [c for c in df_pure.columns if c != "Component"]

# find System 1 request if present
req_s1 = None
for req in user_requests:
    if req["system"] == "SBMA_PDMS":
        req_s1 = req
        break

if req_s1 is None:
    st.info("Select 'SBMA + PDMS' above to run the System 1 test.")
else:
    # validate ranges
    try:
        validate_range("SBMA", req_s1["mw_a"], SBMA_MW_RANGE[0], SBMA_MW_RANGE[1])
        validate_range("PDMS", req_s1["mw_b"], PDMS_MW_RANGE[0], PDMS_MW_RANGE[1])
    except Exception as e:
        st.error(str(e))
        st.stop()

    # get pure rows
    try:
        row_sbma = get_pure_row(df_pure, "SBMA")
        row_pdms = get_pure_row(df_pure, "PDMS")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # build mix
    try:
        X_mix, (f_sbma, f_pdms) = build_mix_system1(
            row_sbma, row_pdms, feature_cols,
            req_s1["mw_a"], req_s1["perc_a"],
            req_s1["mw_b"], req_s1["perc_b"]
        )

        st.success("✅ System 1 mix descriptors built successfully!")
        st.write(f"SBMA fraction = {f_sbma:.6f} | PDMS fraction = {f_pdms:.6f}")
        st.write(f"X_mix shape: {X_mix.shape}")

        # show a small preview of values
        preview_n = min(12, len(feature_cols))
        st.dataframe(pd.DataFrame(X_mix[:, :preview_n], columns=feature_cols[:preview_n]))

    except Exception as e:
        st.error(f"Failed to build System 1 mix descriptors: {e}")
        st.stop()

