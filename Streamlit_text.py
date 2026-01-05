import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(page_title="Fouling Release Predictor", layout="wide")

# ============================================================
# Header
# ============================================================
st.image("fouling_boat.png", use_container_width=True)
st.title("Fouling Release Predictor")

# ============================================================
# Paths and constants
# ============================================================
PURE_CSV_PATH = "Pure_descriptors.csv"
COMP_COL = "Comp"

TRAIN_CSV_PATH = "1_data/5_N_incerta_10_psi_training.csv"
MODEL_PATH = "best_gbr_model.pkl"

MODEL_FEATURES = ["ATSC2se", "ATSC5i", "Xp-4dv", "IC4"]

# ============================================================
# Load pure descriptors
# ============================================================
@st.cache_data
def load_pure_descriptors(path):
    df = pd.read_csv(path, encoding="utf-8")
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding="utf-8", delimiter=";")
    df.columns = df.columns.astype(str).str.strip()
    df[COMP_COL] = df[COMP_COL].astype(str).str.strip().str.upper()
    return df

def get_pure_row(df, comp):
    row = df[df[COMP_COL] == comp]
    if row.empty:
        raise ValueError(f"Component not found: {comp}")
    return row.iloc[0]

def make_feature_matrix_safe(row, feature_cols):
    numeric = pd.to_numeric(row[feature_cols], errors="coerce")
    if numeric.isna().any():
        raise ValueError("Non-numeric values in Pure_descriptors.csv")
    return numeric.values.astype(float)

# ============================================================
# Monomer MW
# ============================================================
MONOMER_MW = {
    "SBMA": 279.3566,
    "PDMS": 74.1535,
    "PEG": 62.0668,
    "PMHS": 62.1600,
}

# ============================================================
# SYSTEM CONSTRAINTS
# ============================================================
SBMA_MW_RANGE = (500.0, 2500.0)
PDMS_MW_RANGE = (1000.0, 5000.0)
SYSTEM1_A_RANGE = (0.01, 0.02)

SYSTEM2_FIXED_MW = {"PDMS": 750.0, "PEG": 1000.0}
SYSTEM2_A_RANGE = (0.10, 0.40)

SYSTEM3_PEG_MW_RANGE = (240.0, 2100.0)
SYSTEM3_PMHS_MW_RANGE = (240.0, 2100.0)
SYSTEM3_WTOTAL_RANGE = (0.01, 0.10)

# ============================================================
# MIX BUILDERS
# ============================================================
def build_mix_system1(row_sbma, row_pdms, feature_cols, mw_sbma, p_sbma, mw_pdms, p_pdms, A):
    n_sbma = mw_sbma / MONOMER_MW["SBMA"]
    n_pdms = mw_pdms / MONOMER_MW["PDMS"]
    v1 = make_feature_matrix_safe(row_sbma, feature_cols)
    v2 = make_feature_matrix_safe(row_pdms, feature_cols)
    mix = A * ((v1 * n_sbma * p_sbma) + (v2 * n_pdms * p_pdms))
    return mix.reshape(1, -1)

def build_mix_system2(row_pdms, row_peg, feature_cols, mw_pdms, p_pdms, mw_peg, p_peg, A):
    n_pdms = mw_pdms / MONOMER_MW["PDMS"]
    n_peg = mw_peg / MONOMER_MW["PEG"]
    v1 = make_feature_matrix_safe(row_pdms, feature_cols)
    v2 = make_feature_matrix_safe(row_peg, feature_cols)
    mix = A * ((v1 * n_pdms * p_pdms) + (v2 * n_peg * p_peg))
    return mix.reshape(1, -1)

def build_mix_system3(row_peg, row_pmhs, feature_cols, mw_peg, mw_pmhs, wt):
    n_peg = mw_peg / MONOMER_MW["PEG"]
    n_pmhs = mw_pmhs / MONOMER_MW["PMHS"]
    v1 = make_feature_matrix_safe(row_peg, feature_cols)
    v2 = make_feature_matrix_safe(row_pmhs, feature_cols)
    mix = wt * ((v1 * n_peg) + (v2 * n_pmhs))
    return mix.reshape(1, -1)

# ============================================================
# UI — SYSTEM SELECTION
# ============================================================
st.markdown("---")
st.header("1) Select coating system")

SYSTEMS = {
    "SBMA + PDMS": "SBMA_PDMS",
    "PDMS + PEG": "PDMS_PEG",
    "PEG + PMHS": "PEG_PMHS",
}

system_label = st.selectbox("Select system", list(SYSTEMS.keys()))
sys_code = SYSTEMS[system_label]

# ============================================================
# SYSTEM INPUTS
# ============================================================
with st.expander(f"System: {system_label}", expanded=True):

    # 🔒 SYSTEM 1 RANGES — VISUAL ONLY
    if sys_code == "SBMA_PDMS":
        st.markdown("#### 🔒 System 1 constraints (SBMA + PDMS)")
        st.info(
            f"""
            **Molecular weight ranges**
            • SBMA: {SBMA_MW_RANGE[0]} – {SBMA_MW_RANGE[1]}  
            • PDMS: {PDMS_MW_RANGE[0]} – {PDMS_MW_RANGE[1]}  

            **Additive amount**
            • A_add: {SYSTEM1_A_RANGE[0]} – {SYSTEM1_A_RANGE[1]}  

            **Composition**
            • p_SBMA: 0–1  
            • p_PDMS: 0–1  
            • p_SBMA + p_PDMS ≠ 0
            """
        )

    if sys_code == "SBMA_PDMS":
        A = st.number_input("Additive Amount A", SYSTEM1_A_RANGE[0], SYSTEM1_A_RANGE[1], SYSTEM1_A_RANGE[0])
        mw_a = st.number_input("MW SBMA", value=1000.0)
        p_a = st.number_input("p SBMA", 0.0, 1.0, 0.5)
        mw_b = st.number_input("MW PDMS", value=1000.0)
        p_b = st.number_input("p PDMS", 0.0, 1.0, 0.5)

    elif sys_code == "PDMS_PEG":
        A = st.number_input("Additive Amount A", SYSTEM2_A_RANGE[0], SYSTEM2_A_RANGE[1], SYSTEM2_A_RANGE[0])
        mw_a = SYSTEM2_FIXED_MW["PDMS"]
        mw_b = SYSTEM2_FIXED_MW["PEG"]
        p_a = st.number_input("p PDMS", 0.0, 1.0, 0.5)
        p_b = st.number_input("p PEG", 0.0, 1.0, 0.5)

    else:
        wt = st.number_input("Total wt% (PEG + PMHS)", SYSTEM3_WTOTAL_RANGE[0], SYSTEM3_WTOTAL_RANGE[1], SYSTEM3_WTOTAL_RANGE[0])
        mw_a = st.number_input("MW PEG", value=240.0)
        mw_b = st.number_input("MW PMHS", value=240.0)

# ============================================================
# BUILD MIX
# ============================================================
st.markdown("---")
st.header("2) Build mix descriptors")

df_pure = load_pure_descriptors(PURE_CSV_PATH)
feature_cols = [c for c in df_pure.columns if c not in {COMP_COL, "No", "No.", "ID"}]

X_mix = None

if sys_code == "SBMA_PDMS":
    row_a = get_pure_row(df_pure, "SBMA")
    row_b = get_pure_row(df_pure, "PDMS")
    X_mix = build_mix_system1(row_a, row_b, feature_cols, mw_a, p_a, mw_b, p_b, A)

elif sys_code == "PDMS_PEG":
    row_a = get_pure_row(df_pure, "PDMS")
    row_b = get_pure_row(df_pure, "PEG")
    X_mix = build_mix_system2(row_a, row_b, feature_cols, mw_a, p_a, mw_b, p_b, A)

else:
    row_a = get_pure_row(df_pure, "PEG")
    row_b = get_pure_row(df_pure, "PMHS")
    X_mix = build_mix_system3(row_a, row_b, feature_cols, mw_a, mw_b, wt)

st.success("Mix descriptors built")

# ============================================================
# MODEL
# ============================================================
st.markdown("---")
st.header("3) Model prediction")

@st.cache_resource
def load_scaler_and_model():
    data = pd.read_csv(TRAIN_CSV_PATH)
    if data.shape[1] == 1:
        data = pd.read_csv(TRAIN_CSV_PATH, delimiter=";")
    data.columns = data.columns.str.strip()
    train = data[data["prediction_training"].str.lower() == "training"]
    X_train = train[MODEL_FEATURES]
    scaler = StandardScaler().fit(X_train.values)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_scaler_and_model()

def prepare_X_for_model(X_mix, feature_cols):
    idx = [feature_cols.index(f) for f in MODEL_FEATURES]
    return X_mix[:, idx]

if st.button("Predict fouling release"):
    X_model = prepare_X_for_model(X_mix, feature_cols)
    X_scaled = scaler.transform(X_model)
    y = model.predict(X_scaled)
    st.subheader("Prediction")
    st.write(float(y[0]))
