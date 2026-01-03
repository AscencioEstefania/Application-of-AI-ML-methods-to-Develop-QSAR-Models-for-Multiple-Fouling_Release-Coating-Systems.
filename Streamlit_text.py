import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# Page configuration (must be the first Streamlit command)
# ============================================================
st.set_page_config(page_title="Fouling Release Predictor", layout="wide")

# ============================================================
# Header image
# ============================================================
st.image("fouling_boat.png", use_container_width=True)
st.title("Fouling Release Predictor")

# ============================================================
# Pure descriptors CSV
# Your CSV file name in the repo: "Pure_descriptors.csv"
# Your component column name: "Comp"  (PDMS, PEG, SBMA, PMHS)
# ============================================================
PURE_CSV_PATH = "Pure_descriptors.csv"

@st.cache_data
def load_pure_descriptors(path=PURE_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    if "Comp" not in df.columns:
        raise ValueError(f"{path} must include a column named 'Comp'.")

    df["Comp"] = df["Comp"].astype(str).str.strip().str.upper()
    return df

def get_pure_row(df_pure: pd.DataFrame, comp_name: str) -> pd.Series:
    comp = str(comp_name).strip().upper()
    row = df_pure[df_pure["Comp"] == comp]
    if row.empty:
        raise ValueError(f"Component not found in {PURE_CSV_PATH}: {comp_name}")
    return row.iloc[0]

def validate_range(name: str, mw_value: float, mw_min: float, mw_max: float):
    mw = float(mw_value)
    if not (mw_min <= mw <= mw_max):
        raise ValueError(f"{name} MW must be between {mw_min} and {mw_max}. You entered {mw_value}.")

# ============================================================
# MIX DESCRIPTORS — System 1 rule (SBMA + PDMS)
#
# IMPORTANT (scientific meaning):
# We scale pure descriptors by the EFFECTIVE NUMBER OF MONOMERIC UNITS:
#   n_monomers = MW_user / MW_monomer
#
# Then we weight by coating percentage and sum both components:
#   mix = (D_SBMA * n_SBMA * p_SBMA) + (D_PDMS * n_PDMS * p_PDMS)
# where p = (% / 100).
# ============================================================
MONOMER_MW = {
    "SBMA": 279.3566,
    "PDMS": 74.1535,
    "PEG":  62.0668,   # (for later systems)
    "PMHS": None       # (we will fill later)
}

SBMA_MW_RANGE = (500.0, 2500.0)
PDMS_MW_RANGE = (1000.0, 5000.0)

def build_mix_system1(
    row_sbma: pd.Series,
    row_pdms: pd.Series,
    feature_cols: list,
    mw_sbma: float,
    perc_sbma: float,
    mw_pdms: float,
    perc_pdms: float,
):
    # percentages to fractions (0–1)
    p_sbma = float(perc_sbma) / 100.0
    p_pdms = float(perc_pdms) / 100.0

    # effective number of monomeric units (monomers)
    # n_monomers = MW_user / MW_monomer
    n_sbma = float(mw_sbma) / float(MONOMER_MW["SBMA"])
    n_pdms = float(mw_pdms) / float(MONOMER_MW["PDMS"])

    # descriptor vectors
    vec_sbma = row_sbma[feature_cols].astype(float).to_numpy()
    vec_pdms = row_pdms[feature_cols].astype(float).to_numpy()

    # mixed descriptors (this X_mix goes into the ML model)
    mix = (vec_sbma * n_sbma * p_sbma) + (vec_pdms * n_pdms * p_pdms)

    return mix.reshape(1, -1), (n_sbma, n_pdms)

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

mode = st.radio("What do you want to evaluate?", ["Only one system", "All systems"], horizontal=True)

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
        return {"system": sys_code, "mw_a": mw_a, "perc_a": perc_a, "mw_b": mw_b, "perc_b": perc_b}

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

# Load pure descriptors
try:
    df_pure = load_pure_descriptors(PURE_CSV_PATH)
except Exception as e:
    st.error(f"Could not load {PURE_CSV_PATH}: {e}")
    st.stop()

# Feature columns = all columns except Comp
feature_cols = [c for c in df_pure.columns if c != "Comp"]

req_s1 = next((r for r in user_requests if r["system"] == "SBMA_PDMS"), None)

if req_s1 is None:
    st.info("Select 'SBMA + PDMS' above to run the System 1 test.")
else:
    # System 1 mapping:
    # A = SBMA  (mw_a, perc_a)
    # B = PDMS  (mw_b, perc_b)
    try:
        validate_range("SBMA", req_s1["mw_a"], SBMA_MW_RANGE[0], SBMA_MW_RANGE[1])
        validate_range("PDMS", req_s1["mw_b"], PDMS_MW_RANGE[0], PDMS_MW_RANGE[1])
    except Exception as e:
        st.error(str(e))
        st.stop()

    try:
        row_sbma = get_pure_row(df_pure, "SBMA")
        row_pdms = get_pure_row(df_pure, "PDMS")
    except Exception as e:
        st.error(str(e))
        st.stop()

    try:
        X_mix, (n_sbma, n_pdms) = build_mix_system1(
            row_sbma, row_pdms, feature_cols,
            req_s1["mw_a"], req_s1["perc_a"],
            req_s1["mw_b"], req_s1["perc_b"]
        )

        st.success("✅ System 1 mix descriptors built successfully!")
        st.write(f"SBMA monomer units (n) = {n_sbma:.6f} | PDMS monomer units (n) = {n_pdms:.6f}")
        st.write(f"X_mix shape: {X_mix.shape}")

        # show all if few, otherwise show first 12
        if len(feature_cols) <= 12:
            st.dataframe(pd.DataFrame(X_mix, columns=feature_cols))
        else:
            preview_n = 12
            st.dataframe(pd.DataFrame(X_mix[:, :preview_n], columns=feature_cols[:preview_n]))

    except Exception as e:
        st.error(f"Failed to build System 1 mix descriptors: {e}")
        st.stop()
