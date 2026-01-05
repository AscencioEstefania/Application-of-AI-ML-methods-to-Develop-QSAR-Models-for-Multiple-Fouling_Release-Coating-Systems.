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
# ============================================================
def build_mix_system1(
    row_sbma: pd.Series,
    row_pdms: pd.Series,
    feature_cols: list,
    mw_sbma: float,
    p_sbma: float,
    mw_pdms: float,
    p_pdms: float,
    additive_amount: float
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
    p_pdms: float,
    mw_peg: float,
    p_peg: float,
    additive_amount: float
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
    wt_total: float
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

            with c1:
                mw_a = st.number_input(
                    "PEG molecular weight must be between 240 to 2100",
                    min_value=SYSTEM3_PEG_MW_RANGE[0],
                    max_value=SYSTEM3_PEG_MW_RANGE[1],
                    value=SYSTEM3_PEG_MW_RANGE[0],
                    step=10.0,
                    key=f"{sys_code}_mw_peg"
                )

            with c2:
                mw_b = st.number_input(
                    "PMHS molecular weight must be between 240 to 2100",
                    min_value=SYSTEM3_PMHS_MW_RANGE[0],
                    max_value=SYSTEM3_PMHS_MW_RANGE[1],
                    value=SYSTEM3_PMHS_MW_RANGE[0],
                    step=10.0,
                    key=f"{sys_code}_mw_pmhs"
                )

            wt_total = st.number_input(
                "The PEG + PMHS additive percentage to be added to the coating ranges from 0.01 to 0.1",
                min_value=SYSTEM3_WTOTAL_RANGE[0],
                max_value=SYSTEM3_WTOTAL_RANGE[1],
                value=SYSTEM3_WTOTAL_RANGE[0],
                step=0.001,
                format="%.3f",
                key=f"{sys_code}_wt_total"
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

# ============================================================
# INTERNAL: Build mix descriptors silently (NO UI TABLES / NO DEBUG)
# ============================================================
try:
    df_pure = load_pure_descriptors(PURE_CSV_PATH)
except Exception as e:
    st.error(f"Could not load {PURE_CSV_PATH}: {e}")
    st.stop()

drop_cols = {COMP_COL, "No.", "No", "ID"}
feature_cols = [c for c in df_pure.columns if c not in drop_cols]

X_mix1 = None
X_mix2 = None
X_mix3 = None

# System 1 — SBMA + PDMS
req_s1 = next((r for r in user_requests if r["system"] == "SBMA_PDMS"), None)
if req_s1 is not None:
    try:
        validate_range("SBMA", req_s1["mw_a"], SBMA_MW_RANGE[0], SBMA_MW_RANGE[1])
        validate_range("PDMS", req_s1["mw_b"], PDMS_MW_RANGE[0], PDMS_MW_RANGE[1])
        validate_p("SBMA", req_s1["p_a"])
        validate_p("PDMS", req_s1["p_b"])
        if not (SYSTEM1_A_RANGE[0] <= req_s1["A_add"] <= SYSTEM1_A_RANGE[1]):
            raise ValueError(
                f"System 1 Additive Amount A must be between {SYSTEM1_A_RANGE[0]} and {SYSTEM1_A_RANGE[1]}."
            )

        row_sbma = get_pure_row(df_pure, "SBMA")
        row_pdms = get_pure_row(df_pure, "PDMS")

        X_mix1, _ = build_mix_system1(
            row_sbma=row_sbma,
            row_pdms=row_pdms,
            feature_cols=feature_cols,
            mw_sbma=req_s1["mw_a"],
            p_sbma=req_s1["p_a"],
            mw_pdms=req_s1["mw_b"],
            p_pdms=req_s1["p_b"],
            additive_amount=req_s1["A_add"]
        )
    except Exception as e:
        st.error(f"Failed to build System 1 mix descriptors: {e}")
        st.stop()

# System 2 — PDMS + PEG
req_s2 = next((r for r in user_requests if r["system"] == "PDMS_PEG"), None)
if req_s2 is not None:
    try:
        validate_p("PDMS", req_s2["p_a"])
        validate_p("PEG",  req_s2["p_b"])
        validate_fixed_mw("PDMS", req_s2["mw_a"], SYSTEM2_FIXED_MW["PDMS"])
        validate_fixed_mw("PEG",  req_s2["mw_b"], SYSTEM2_FIXED_MW["PEG"])
        if not (SYSTEM2_A_RANGE[0] <= req_s2["A_add"] <= SYSTEM2_A_RANGE[1]):
            raise ValueError(
                f"System 2 Additive Amount A must be between {SYSTEM2_A_RANGE[0]} and {SYSTEM2_A_RANGE[1]}."
            )

        row_pdms = get_pure_row(df_pure, "PDMS")
        row_peg  = get_pure_row(df_pure, "PEG")

        X_mix2, _ = build_mix_system2(
            row_pdms=row_pdms,
            row_peg=row_peg,
            feature_cols=feature_cols,
            mw_pdms=req_s2["mw_a"],
            p_pdms=req_s2["p_a"],
            mw_peg=req_s2["mw_b"],
            p_peg=req_s2["p_b"],
            additive_amount=req_s2["A_add"]
        )
    except Exception as e:
        st.error(f"Failed to build System 2 mix descriptors: {e}")
        st.stop()

# System 3 — PEG + PMHS
req_s3 = next((r for r in user_requests if r["system"] == "PEG_PMHS"), None)
if req_s3 is not None:
    try:
        validate_total_wt(
            "PEG + PMHS total wt%",
            req_s3["wt_total"],
            SYSTEM3_WTOTAL_RANGE[0],
            SYSTEM3_WTOTAL_RANGE[1]
        )
        validate_range("PEG",  req_s3["mw_a"], SYSTEM3_PEG_MW_RANGE[0],  SYSTEM3_PEG_MW_RANGE[1])
        validate_range("PMHS", req_s3["mw_b"], SYSTEM3_PMHS_MW_RANGE[0], SYSTEM3_PMHS_MW_RANGE[1])

        row_peg  = get_pure_row(df_pure, "PEG")
        row_pmhs = get_pure_row(df_pure, "PMHS")

        X_mix3, _ = build_mix_system3(
            row_peg=row_peg,
            row_pmhs=row_pmhs,
            feature_cols=feature_cols,
            mw_peg=req_s3["mw_a"],
            mw_pmhs=req_s3["mw_b"],
            wt_total=req_s3["wt_total"]
        )
    except Exception as e:
        st.error(f"Failed to build System 3 mix descriptors: {e}")
        st.stop()

# ============================================================
# PART 3 — CONNECT MODEL (WITH INTERNAL SCALING)
# (kept exactly as your code; we only removed the Part 2 UI output)
# ============================================================
st.markdown("---")
st.header("3) Model prediction")

from sklearn.preprocessing import StandardScaler
import pickle
import os

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
TRAIN_CSV_PATH = "1_data/5_N_incerta_10_psi_training.csv"
MODEL_PATH = "best_gbr_model.pkl"  # correct (file is in repo root)

# Model features (fixed)
MODEL_FEATURES = ["ATSC2se", "ATSC5i", "Xp-4dv", "IC4"]


def read_training_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV reader:
    - tries comma first
    - if only 1 column, retries with semicolon
    - strips column names
    """
    df = pd.read_csv(path, encoding="UTF-8")
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding="UTF-8", delimiter=";")
    df.columns = df.columns.astype(str).str.strip()
    return df


@st.cache_resource
def load_scaler_and_model(train_csv_path: str, model_path: str):
    # ---- Load training data (robust) ----
    data = read_training_csv(train_csv_path)

    # ---- Validate required column ----
    if "prediction_training" not in data.columns:
        raise KeyError(
            "Column 'prediction_training' not found in training CSV. "
            f"Found columns: {list(data.columns)[:30]}"
        )

    # ---- Clean split column ----
    data["prediction_training"] = (
        data["prediction_training"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_data = data[data["prediction_training"] == "training"]
    if train_data.empty:
        raise ValueError("No rows found with prediction_training == 'training'.")

    # ---- Validate model feature columns ----
    missing_feats = [f for f in MODEL_FEATURES if f not in data.columns]
    if missing_feats:
        raise KeyError(
            "MODEL_FEATURES missing in training CSV: " + ", ".join(missing_feats)
        )

    X_train = train_data[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    if X_train.isna().any().any():
        bad = X_train.columns[X_train.isna().any()].tolist()
        raise ValueError("NaN detected in training features: " + ", ".join(bad))

    # ---- Fit scaler ONLY on training data ----
    scaler = StandardScaler()
    scaler.fit(X_train.values)

    # ---- Load trained model (PKL) ----
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return scaler, model


def prepare_X_for_model(X_mix: np.ndarray, all_feature_cols: list) -> np.ndarray:
    """
    Extracts MODEL_FEATURES from X_mix in the correct order.
    """
    missing = [f for f in MODEL_FEATURES if f not in all_feature_cols]
    if missing:
        raise ValueError("MODEL_FEATURES missing from feature_cols: " + ", ".join(missing))

    idx = [all_feature_cols.index(f) for f in MODEL_FEATURES]
    X_model = X_mix[:, idx].astype(float)

    if np.isnan(X_model).any():
        raise ValueError("NaN detected in model input features.")

    return X_model


# ---- Load scaler + model once ----
try:
    scaler, model = load_scaler_and_model(TRAIN_CSV_PATH, MODEL_PATH)
    st.success("✅ Scaler and model loaded successfully")
except Exception as e:
    st.error(f"Failed to load scaler/model: {e}")
    try:
        tmp = read_training_csv(TRAIN_CSV_PATH)
        st.write("DEBUG — Training CSV columns:", list(tmp.columns))
        st.write("DEBUG — Training CSV shape:", tmp.shape)
    except Exception as e2:
        st.write(f"DEBUG — Could not read training CSV: {e2}")
    st.stop()


# ------------------------------------------------------------
# Collect available systems
# ------------------------------------------------------------
mix_map = {}

if "X_mix1" in globals() and X_mix1 is not None:
    mix_map["SBMA + PDMS"] = X_mix1

if "X_mix2" in globals() and X_mix2 is not None:
    mix_map["PDMS + PEG"] = X_mix2

if "X_mix3" in globals() and X_mix3 is not None:
    mix_map["PEG + PMHS"] = X_mix3


if not mix_map:
    st.warning("No mix descriptors available. Please run at least one system.")
else:
    st.write("Model input features:", MODEL_FEATURES)

    if st.button("Predict fouling release"):
        results = []

        for system_name, X_mix in mix_map.items():
            try:
                # 1) select features
                X_model = prepare_X_for_model(X_mix, feature_cols)

                # 2) scale using the training scaler
                X_scaled = scaler.transform(X_model)

                # 3) predict
                y_hat = model.predict(X_scaled)

                results.append({
                    "System": system_name,
                    "Prediction": float(y_hat[0])
                })

            except Exception as e:
                results.append({
                    "System": system_name,
                    "Prediction": np.nan,
                    "Error": str(e)
                })

        df_results = pd.DataFrame(results)
        st.subheader("Prediction results")
        st.dataframe(df_results)

        st.download_button(
            "Download predictions",
            data=df_results.to_csv(index=False).encode("utf-8"),
            file_name="fouling_release_predictions.csv",
            mime="text/csv"
        )




