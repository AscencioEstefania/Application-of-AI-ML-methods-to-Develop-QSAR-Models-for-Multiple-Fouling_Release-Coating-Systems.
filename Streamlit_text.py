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
# Pure descriptors CSV
# Repo file name: "Pure_descriptors.csv"
# Component column: "Comp"  (PDMS, PEG, SBMA, PMHS)
# ============================================================
PURE_CSV_PATH = "Pure_descriptors.csv"
COMP_COL = "Comp"  # your real column name in the CSV

@st.cache_data
def load_pure_descriptors(path: str = PURE_CSV_PATH) -> pd.DataFrame:
    """
    Loads the pure descriptors file and standardizes component labels.
    Also handles ';' delimiter if needed.
    """
    df = pd.read_csv(path, encoding="utf-8")
    # If it looks like a single column, try semicolon delimiter
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


def validate_fixed_mw(name: str, mw_value: float, allowed_value: float, tol: float = 1e-6):
    """
    Enforces fixed MW (e.g., PDMS must be exactly 750).
    """
    mw = float(mw_value)
    if abs(mw - float(allowed_value)) > tol:
        raise ValueError(f"{name} MW must be exactly {allowed_value}. You entered {mw_value}.")


def validate_total_wt(name: str, wt_value: float, wt_min: float, wt_max: float):
    wt = float(wt_value)
    if not (wt_min <= wt <= wt_max):
        raise ValueError(f"{name} must be between {wt_min} and {wt_max}. You entered {wt_value}.")


def make_feature_matrix_safe(row: pd.Series, feature_cols: list) -> np.ndarray:
    """
    Converts descriptor columns to float safely.
    If any non-numeric values exist, raises a clean error showing which columns failed.
    """
    s = row[feature_cols].copy()
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


# ============================================================
# MONOMER MW (given by you)
# ============================================================
MONOMER_MW = {
    "SBMA": 279.3566,
    "PDMS": 74.1535,
    "PEG":  62.0668,
    "PMHS": 62.1600,   # <- System 3
}

# ============================================================
# System 1 (SBMA + PDMS) constraints
# ============================================================
SBMA_MW_RANGE = (500.0, 2500.0)
PDMS_MW_RANGE = (1000.0, 5000.0)
SYSTEM1_A_RANGE = (0.01, 0.02)

# ============================================================
# System 2 (PDMS + PEG) constraints
# ============================================================
SYSTEM2_FIXED_MW = {"PDMS": 750.0, "PEG": 1000.0}
SYSTEM2_A_RANGE = (0.10, 0.40)

# ============================================================
# System 3 (PEG + PMHS) constraints (your special case)
# - MW ranges: 240–2100 for both
# - NO individual wt% (no pPEG, pPMHS)
# - total wt% (PEG + PMHS) in 0.01–0.10
# ============================================================
SYSTEM3_PEG_MW_RANGE  = (240.0, 2100.0)
SYSTEM3_PMHS_MW_RANGE = (240.0, 2100.0)
SYSTEM3_WTOTAL_RANGE  = (0.01, 0.10)

# ============================================================
# MIX builders
# System 1 & 2 (as you had):
#   n = MW_user / MW_monomer
#   mix = A * (D_A*n_A*p_A + D_B*n_B*p_B)
#
# System 3 (your rule):
#   n = MW_user / MW_monomer
#   mix = wt_total * (D_PEG*n_PEG + D_PMHS*n_PMHS)
#   (NO p split, NO 50/50 assumption)
# ============================================================
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
    n_sbma = float(mw_sbma) / float(MONOMER_MW["SBMA"])
    n_pdms = float(mw_pdms) / float(MONOMER_MW["PDMS"])

    vec_sbma = make_feature_matrix_safe(row_sbma, feature_cols)
    vec_pdms = make_feature_matrix_safe(row_pdms, feature_cols)

    mix = float(additive_amount) * ((vec_sbma * n_sbma * float(p_sbma)) + (vec_pdms * n_pdms * float(p_pdms)))
    return mix.reshape(1, -1), (n_sbma, n_pdms)


def build_mix_system2(
    row_pdms: pd.Series,
    row_peg: pd.Series,
    feature_cols: list,
    mw_pdms: float,
    p_pdms: float,          # already 0–1
    mw_peg: float,
    p_peg: float,           # already 0–1
    additive_amount: float  # A in [0.10, 0.40]
):
    n_pdms = float(mw_pdms) / float(MONOMER_MW["PDMS"])
    n_peg  = float(mw_peg)  / float(MONOMER_MW["PEG"])

    vec_pdms = make_feature_matrix_safe(row_pdms, feature_cols)
    vec_peg  = make_feature_matrix_safe(row_peg,  feature_cols)

    mix = float(additive_amount) * ((vec_pdms * n_pdms * float(p_pdms)) + (vec_peg * n_peg * float(p_peg)))
    return mix.reshape(1, -1), (n_pdms, n_peg)


def build_mix_system3(
    row_peg: pd.Series,
    row_pmhs: pd.Series,
    feature_cols: list,
    mw_peg: float,
    mw_pmhs: float,
    wt_total: float  # total (PEG + PMHS) in [0.01, 0.10]
):
    n_peg  = float(mw_peg)  / float(MONOMER_MW["PEG"])
    n_pmhs = float(mw_pmhs) / float(MONOMER_MW["PMHS"])

    vec_peg  = make_feature_matrix_safe(row_peg,  feature_cols)
    vec_pmhs = make_feature_matrix_safe(row_pmhs, feature_cols)

    mix = float(wt_total) * ((vec_peg * n_peg) + (vec_pmhs * n_pmhs))
    return mix.reshape(1, -1), (n_peg, n_pmhs)


# ============================================================
# PART 1 — MULTI-SYSTEM INPUT (UI)
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
        # ---- System 3 special UI (NO p split) ----
        if sys_code == "PEG_PMHS":
            c1, c2 = st.columns(2)

            wt_total = st.number_input(
                f"[{label}] Total wt% (PEG + PMHS)  [0.01–0.10]",
                min_value=SYSTEM3_WTOTAL_RANGE[0],
                max_value=SYSTEM3_WTOTAL_RANGE[1],
                value=SYSTEM3_WTOTAL_RANGE[0],
                step=0.001,
                format="%.3f",
                key=f"{sys_code}_wt_total"
            )

            with c1:
                mw_a = st.number_input(
                    f"[{label}] MW PEG",
                    min_value=0.0, value=240.0, step=10.0,
                    key=f"{sys_code}_mw_peg"
                )
            with c2:
                mw_b = st.number_input(
                    f"[{label}] MW PMHS",
                    min_value=0.0, value=240.0, step=10.0,
                    key=f"{sys_code}_mw_pmhs"
                )

            st.caption("System 3 uses ONLY total wt% (PEG+PMHS). No individual p values, no 50/50 assumption.")
            return {
                "system": sys_code,
                "mw_a": mw_a, "mw_b": mw_b,
                "wt_total": wt_total,
            }

        # ---- Systems 1 & 2 (original UI) ----
        c1, c2 = st.columns(2)

        # ---- Additive Amount A (depends on system) ----
        if sys_code == "SBMA_PDMS":
            A_add = st.number_input(
                f"[{label}] Additive Amount A",
                min_value=SYSTEM1_A_RANGE[0],
                max_value=SYSTEM1_A_RANGE[1],
                value=SYSTEM1_A_RANGE[0],
                step=0.001,
                format="%.3f",
                key=f"{sys_code}_A_add"
            )
        elif sys_code == "PDMS_PEG":
            A_add = st.number_input(
                f"[{label}] Additive Amount A",
                min_value=SYSTEM2_A_RANGE[0],
                max_value=SYSTEM2_A_RANGE[1],
                value=SYSTEM2_A_RANGE[0],
                step=0.01,
                format="%.2f",
                key=f"{sys_code}_A_add"
            )
        else:
            A_add = st.number_input(
                f"[{label}] Additive Amount A",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.01,
                key=f"{sys_code}_A_add"
            )

        with c1:
            # MW A
            if sys_code == "PDMS_PEG":
                mw_a = st.selectbox(
                    f"[{label}] MW component A (PDMS fixed)",
                    options=[SYSTEM2_FIXED_MW["PDMS"]],
                    key=f"{sys_code}_mw_a_select"
                )
                mw_a = float(mw_a)
            else:
                mw_a = st.number_input(
                    f"[{label}] MW component A",
                    min_value=0.0, value=1000.0, step=10.0,
                    key=f"{sys_code}_mw_a"
                )

            # p A (0-1)
            p_a = st.number_input(
                f"[{label}] p (0–1) component A",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                key=f"{sys_code}_p_a"
            )

        with c2:
            # MW B
            if sys_code == "PDMS_PEG":
                mw_b = st.selectbox(
                    f"[{label}] MW component B (PEG fixed)",
                    options=[SYSTEM2_FIXED_MW["PEG"]],
                    key=f"{sys_code}_mw_b_select"
                )
                mw_b = float(mw_b)
            else:
                mw_b = st.number_input(
                    f"[{label}] MW component B",
                    min_value=0.0, value=1000.0, step=10.0,
                    key=f"{sys_code}_mw_b"
                )

            # p B (0-1)
            p_b = st.number_input(
                f"[{label}] p (0–1) component B",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                key=f"{sys_code}_p_b"
            )

        if (p_a + p_b) == 0:
            st.error("p values cannot both be zero.")
            return None

        st.caption("Next step: connect this to your descriptor-based ML model.")
        return {
            "system": sys_code,
            "mw_a": mw_a, "p_a": p_a,
            "mw_b": mw_b, "p_b": p_b,
            "A_add": A_add,
        }


user_requests = []
for label in systems_to_show:
    out = system_expander(label)
    if out is not None:
        user_requests.append(out)

st.subheader("Current inputs (debug)")
st.dataframe(user_requests)

# ============================================================
# PART 2 — BUILD MIX DESCRIPTORS (System 1, 2, 3)
# ============================================================
st.markdown("---")
st.header("2) Mix descriptors tests")

# Load pure descriptors
try:
    df_pure = load_pure_descriptors(PURE_CSV_PATH)
except Exception as e:
    st.error(f"Could not load {PURE_CSV_PATH}: {e}")
    st.stop()

# Feature columns: everything except label columns
drop_cols = {COMP_COL, "No.", "No", "ID"}
feature_cols = [c for c in df_pure.columns if c not in drop_cols]

# -----------------------------
# System 1 — SBMA + PDMS
# -----------------------------
st.subheader("System 1 — SBMA + PDMS")

req_s1 = next((r for r in user_requests if r["system"] == "SBMA_PDMS"), None)

if req_s1 is None:
    st.info("Select 'SBMA + PDMS' above to run the System 1 test.")
else:
    try:
        validate_range("SBMA", req_s1["mw_a"], SBMA_MW_RANGE[0], SBMA_MW_RANGE[1])
        validate_range("PDMS", req_s1["mw_b"], PDMS_MW_RANGE[0], PDMS_MW_RANGE[1])
        validate_p("SBMA", req_s1["p_a"])
        validate_p("PDMS", req_s1["p_b"])

        if not (SYSTEM1_A_RANGE[0] <= req_s1["A_add"] <= SYSTEM1_A_RANGE[1]):
            raise ValueError(f"System 1 Additive Amount A must be between {SYSTEM1_A_RANGE[0]} and {SYSTEM1_A_RANGE[1]}.")

        row_sbma = get_pure_row(df_pure, "SBMA")
        row_pdms = get_pure_row(df_pure, "PDMS")

        X_mix1, (n_sbma, n_pdms) = build_mix_system1(
            row_sbma=row_sbma,
            row_pdms=row_pdms,
            feature_cols=feature_cols,
            mw_sbma=req_s1["mw_a"],
            p_sbma=req_s1["p_a"],
            mw_pdms=req_s1["mw_b"],
            p_pdms=req_s1["p_b"],
            additive_amount=req_s1["A_add"]
        )

        st.success("✅ System 1 mix descriptors built successfully!")
        st.write(f"n_SBMA = MW_user/MW_monomer = {n_sbma:.6f} | n_PDMS = {n_pdms:.6f}")
        st.write(f"Additive Amount A = {req_s1['A_add']}")
        st.write(f"X_mix1 shape: {X_mix1.shape}")

        preview_n = min(12, len(feature_cols))
        st.dataframe(pd.DataFrame(X_mix1[:, :preview_n], columns=feature_cols[:preview_n]))

    except Exception as e:
        st.error(f"Failed to build System 1 mix descriptors: {e}")
        st.stop()

# -----------------------------
# System 2 — PDMS + PEG
# -----------------------------
st.subheader("System 2 — PDMS + PEG")

req_s2 = next((r for r in user_requests if r["system"] == "PDMS_PEG"), None)

if req_s2 is None:
    st.info("Select 'PDMS + PEG' above to run the System 2 test.")
else:
    try:
        validate_p("PDMS", req_s2["p_a"])
        validate_p("PEG",  req_s2["p_b"])

        # Fixed MW rules (your system 2)
        validate_fixed_mw("PDMS", req_s2["mw_a"], SYSTEM2_FIXED_MW["PDMS"])
        validate_fixed_mw("PEG",  req_s2["mw_b"], SYSTEM2_FIXED_MW["PEG"])

        if not (SYSTEM2_A_RANGE[0] <= req_s2["A_add"] <= SYSTEM2_A_RANGE[1]):
            raise ValueError(f"System 2 Additive Amount A must be between {SYSTEM2_A_RANGE[0]} and {SYSTEM2_A_RANGE[1]}.")

        row_pdms = get_pure_row(df_pure, "PDMS")
        row_peg  = get_pure_row(df_pure, "PEG")

        X_mix2, (n_pdms, n_peg) = build_mix_system2(
            row_pdms=row_pdms,
            row_peg=row_peg,
            feature_cols=feature_cols,
            mw_pdms=req_s2["mw_a"],
            p_pdms=req_s2["p_a"],
            mw_peg=req_s2["mw_b"],
            p_peg=req_s2["p_b"],
            additive_amount=req_s2["A_add"]
        )

        st.success("✅ System 2 mix descriptors built successfully!")
        st.write(f"n_PDMS = {n_pdms:.6f} | n_PEG = {n_peg:.6f}")
        st.write(f"Additive Amount A = {req_s2['A_add']}")
        st.write(f"X_mix2 shape: {X_mix2.shape}")

        preview_n = min(12, len(feature_cols))
        st.dataframe(pd.DataFrame(X_mix2[:, :preview_n], columns=feature_cols[:preview_n]))

    except Exception as e:
        st.error(f"Failed to build System 2 mix descriptors: {e}")
        st.stop()




