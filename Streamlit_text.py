import streamlit as st
import pandas as pd
import numpy as np
import os

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
# Pure descriptors CSV (your repo file: "Pure_descriptors.csv")
# Component column: "Comp"  (PDMS, PEG, SBMA, PMHS)
# ============================================================
PURE_CSV_PATH = "Pure_descriptors.csv"
COMP_COL = "Comp"  # <-- your real column name

@st.cache_data
def load_pure_descriptors(path: str = PURE_CSV_PATH) -> pd.DataFrame:
    """
    Loads the pure descriptors file and standardizes component labels.
    Also handles ';' delimiter if needed.
    """
    # Try comma first
    df = pd.read_csv(path, encoding="utf-8")
    # If it looks like a single column, try semicolon
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding="utf-8", delimiter=";")

    df.columns = df.columns.astype(str).str.strip()

    if COMP_COL not in df.columns:
        raise ValueError(f"{path} must include a column named '{COMP_COL}'.")

    df[COMP_COL] = df[COMP_COL].astype(str).str.strip().str.upper()
    return df


def get_pure_row(df_pure: pd.DataFrame, comp_name: str) -> pd.Series:
    comp = str(comp_name).strip().upper()
    row = df_pure[df_pure[COMP_COL] == comp]
    if row.empty:
        raise ValueError(f"Component not found in {PURE_CSV_PATH}: {comp_name}")
    return row.iloc[0]


def validate_range(name: str, mw_value: float, mw_min: float, mw_max: float):
    mw = float(mw_value)
    if not (mw_min <= mw <= mw_max):
        raise ValueError(f"{name} MW must be between {mw_min} and {mw_max}. You entered {mw_value}.")


def validate_p(name: str, p_value: float):
    p = float(p_value)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} p (0–1) must be between 0 and 1. You entered {p_value}.")


# ============================================================
# System 1 constants (SBMA + PDMS)
# MW ranges given by you:
# - SBMA: 500–2500
# - PDMS: 1000–5000
# Monomer MW:
# - SBMA: 279.3566
# - PDMS: 74.1535
# ============================================================
MONOMER_MW = {
    "SBMA": 279.3566,
    "PDMS": 74.1535,
    # (later systems)
    "PEG":  62.0668,
    "PMHS": None,
}

SBMA_MW_RANGE = (500.0, 2500.0)
PDMS_MW_RANGE = (1000.0, 5000.0)


def make_feature_matrix_safe(row: pd.Series, feature_cols: list) -> np.ndarray:
    """
    Converts descriptor columns to float safely.
    If any non-numeric values exist, raises a clean error showing which columns failed.
    """
    s = row[feature_cols].copy()

    # Convert to numeric; non-numeric -> NaN
    numeric = pd.to_numeric(s, errors="coerce")

    bad_cols = list(numeric[numeric.isna()].index)
    if bad_cols:
        raise ValueError(
            "Non-numeric descriptor values detected in columns: "
            + ", ".join(bad_cols[:20])
            + (" ..." if len(bad_cols) > 20 else "")
            + "\nFix your Pure_descriptors.csv so those columns are numeric."
        )

    return numeric.astype(float).to_numpy()


def build_mix_system1(
    row_sbma: pd.Series,
    row_pdms: pd.Series,
    feature_cols: list,
    mw_sbma: float,
    p_sbma: float,          # already 0–1
    mw_pdms: float,
    p_pdms: float,          # already 0–1
    additive_amount: float  # A in [0.01, 0.02]
):
    """
    System 1 mix descriptors:

    n = MW_user / MW_monomer      (monomer units)
    mix = A * (D_SBMA*n_SBMA*p_SBMA + D_PDMS*n_PDMS*p_PDMS)

    Returns:
      X_mix (1, n_features)
      (n_sbma, n_pdms)
    """
    # monomer units (THIS is what you want; not "fraction")
    n_sbma = float(mw_sbma) / float(MONOMER_MW["SBMA"])
    n_pdms = float(mw_pdms) / float(MONOMER_MW["PDMS"])

    vec_sbma = make_feature_matrix_safe(row_sbma, feature_cols)
    vec_pdms = make_feature_matrix_safe(row_pdms, feature_cols)

    mix = additive_amount * ((vec_sbma * n_sbma * p_sbma) + (vec_pdms * n_pdms * p_pdms))
    return mix.reshape(1, -1), (n_sbma, n_pdms)


# ============================================================
# PART 1 — UI: system selector + inputs
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
            p_a = st.number_input(
                f"[{label}] p (0–1) component A",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                key=f"{sys_code}_p_a"
            )

        with c2:
            mw_b = st.number_input(
                f"[{label}] MW component B",
                min_value=0.0, value=1000.0, step=10.0,
                key=f"{sys_code}_mw_b"
            )
            p_b = st.number_input(
                f"[{label}] p (0–1) component B",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                key=f"{sys_code}_p_b"
            )

        additive_amount = st.number_input(
            f"[{label}] Additive Amount A (0.01–0.02)",
            min_value=0.01, max_value=0.02, value=0.01, step=0.001,
            key=f"{sys_code}_A_add"
        )

        if (p_a + p_b) == 0:
            st.error("p values cannot both be zero.")
            return None

        st.caption("Next step: we will connect this input to your descriptor-based model.")

        return {
            "system": sys_code,
            "mw_a": mw_a, "p_a": p_a,
            "mw_b": mw_b, "p_b": p_b,
            "A_add": additive_amount
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

# Load pure descriptors
try:
    df_pure = load_pure_descriptors(PURE_CSV_PATH)
except Exception as e:
    st.error(f"Could not load {PURE_CSV_PATH}: {e}")
    st.stop()

# Feature columns: everything except label columns you might have
drop_cols = {COMP_COL, "No.", "No", "ID"}
feature_cols = [c for c in df_pure.columns if c not in drop_cols]

req_s1 = next((r for r in user_requests if r["system"] == "SBMA_PDMS"), None)

if req_s1 is None:
    st.info("Select 'SBMA + PDMS' above to run the System 1 test.")
else:
    # Mapping for System 1:
    # A = SBMA  (mw_a, p_a)
    # B = PDMS  (mw_b, p_b)
    try:
        validate_range("SBMA", req_s1["mw_a"], SBMA_MW_RANGE[0], SBMA_MW_RANGE[1])
        validate_range("PDMS", req_s1["mw_b"], PDMS_MW_RANGE[0], PDMS_MW_RANGE[1])
        validate_p("SBMA", req_s1["p_a"])
        validate_p("PDMS", req_s1["p_b"])
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
            req_s1["mw_a"], req_s1["p_a"],
            req_s1["mw_b"], req_s1["p_b"],
            additive_amount=req_s1["A_add"]
        )

        st.success("✅ System 1 mix descriptors built successfully!")
        st.write(f"SBMA monomer units (n) = {n_sbma:.6f} | PDMS monomer units (n) = {n_pdms:.6f}")
        st.write(f"Additive Amount A = {req_s1['A_add']}")
        st.write(f"X_mix shape: {X_mix.shape}")

        # Show all if few, otherwise show first 12
        if len(feature_cols) <= 12:
            st.dataframe(pd.DataFrame(X_mix, columns=feature_cols))
        else:
            preview_n = 12
            st.dataframe(pd.DataFrame(X_mix[:, :preview_n], columns=feature_cols[:preview_n]))

    except Exception as e:
        st.error(f"Failed to build System 1 mix descriptors: {e}")
        st.stop()



