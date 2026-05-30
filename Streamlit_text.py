import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Fouling Release Predictor", layout="wide")

st.image("fouling_boat.png", use_container_width=True)
st.title("Fouling Release Predictor")
st.caption("Models implemented: N. incerta at 10 psi and C. lytica at 20 psi")

COMP_COL = "Comp"

TARGET_OPTIONS = {
    "N. incerta at 10 psi": {
        "pure_csv": "Pure_descriptors.csv",
        "train_csv": "1_data/5_N_incerta_10_psi_training.csv",
        "model_path": "best_gbr_model.pkl",
        "features": ["C-006", "ATSC2c", "IC3"],
        "output_name": "N. incerta at 10 psi",
        "csv_name": "n_incerta_10psi_predictions.csv",
        "pressure_label": "10 psi",
    },
    "C. lytica at 20 psi": {
        "pure_csv": "Pure_descriptors_C_lytica.csv",
        "train_csv": "C_lytica_Unid_Monomericas_20_psi.csv",
        "model_path": "Lytica_best_gbr_model.pkl",
        "features": ["nOHp", "MATS3Z", "ATS4are"],
        "output_name": "C. lytica at 20 psi",
        "csv_name": "c_lytica_20psi_predictions.csv",
        "pressure_label": "20 psi",
    },
}

st.markdown("---")
st.header("0) Select prediction model")

selected_target = st.selectbox(
    "Select microorganism/model:",
    list(TARGET_OPTIONS.keys()),
)

TARGET_CONFIG = TARGET_OPTIONS[selected_target]

PURE_CSV_PATH = TARGET_CONFIG["pure_csv"]
TRAIN_CSV_PATH = TARGET_CONFIG["train_csv"]
MODEL_PATH = TARGET_CONFIG["model_path"]
MODEL_FEATURES = TARGET_CONFIG["features"]
OUTPUT_NAME = TARGET_CONFIG["output_name"]
OUTPUT_CSV_NAME = TARGET_CONFIG["csv_name"]
PRESSURE_LABEL = TARGET_CONFIG["pressure_label"]


@st.cache_data
def load_pure_descriptors(path: str = PURE_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
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
    mw = float(mw_value)
    if abs(mw - float(allowed_value)) > tol:
        raise ValueError(f"{name} MW must be exactly {allowed_value}. You entered {mw_value}.")


def validate_total_wt(name: str, wt_value: float, wt_min: float, wt_max: float):
    wt = float(wt_value)
    if not (wt_min <= wt <= wt_max):
        raise ValueError(f"{name} must be between {wt_min} and {wt_max}. You entered {wt_value}.")


def make_feature_matrix_safe(row: pd.Series, feature_cols: list) -> np.ndarray:
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


MONOMER_MW = {
    "SBMA": 279.3566,
    "PDMS": 74.1535,
    "PEG": 62.0668,
    "PMHS": 62.1600,
}

SBMA_MW_RANGE = (500.0, 2500.0)
PDMS_MW_RANGE = (1000.0, 5000.0)
SYSTEM1_A_RANGE = (0.01, 0.02)

SYSTEM2_FIXED_MW = {"PDMS": 750.0, "PEG": 1000.0}
SYSTEM2_A_RANGE = (0.10, 0.40)

SYSTEM3_PEG_MW_RANGE = (240.0, 2100.0)
SYSTEM3_PMHS_MW_RANGE = (240.0, 2100.0)
SYSTEM3_WTOTAL_RANGE = (0.01, 0.10)


def build_mix_system1(row_sbma, row_pdms, feature_cols, mw_sbma, p_sbma, mw_pdms, p_pdms, additive_amount):
    n_sbma = float(mw_sbma) / float(MONOMER_MW["SBMA"])
    n_pdms = float(mw_pdms) / float(MONOMER_MW["PDMS"])

    vec_sbma = make_feature_matrix_safe(row_sbma, feature_cols)
    vec_pdms = make_feature_matrix_safe(row_pdms, feature_cols)

    mix = float(additive_amount) * (
        (vec_sbma * n_sbma * float(p_sbma)) + (vec_pdms * n_pdms * float(p_pdms))
    )
    return mix.reshape(1, -1), (n_sbma, n_pdms)


def build_mix_system2(row_pdms, row_peg, feature_cols, mw_pdms, p_pdms, mw_peg, p_peg, additive_amount):
    n_pdms = float(mw_pdms) / float(MONOMER_MW["PDMS"])
    n_peg = float(mw_peg) / float(MONOMER_MW["PEG"])

    vec_pdms = make_feature_matrix_safe(row_pdms, feature_cols)
    vec_peg = make_feature_matrix_safe(row_peg, feature_cols)

    mix = float(additive_amount) * (
        (vec_pdms * n_pdms * float(p_pdms)) + (vec_peg * n_peg * float(p_peg))
    )
    return mix.reshape(1, -1), (n_pdms, n_peg)


def build_mix_system3(row_peg, row_pmhs, feature_cols, mw_peg, mw_pmhs, wt_total):
    n_peg = float(mw_peg) / float(MONOMER_MW["PEG"])
    n_pmhs = float(mw_pmhs) / float(MONOMER_MW["PMHS"])

    vec_peg = make_feature_matrix_safe(row_peg, feature_cols)
    vec_pmhs = make_feature_matrix_safe(row_pmhs, feature_cols)

    mix = float(wt_total) * ((vec_peg * n_peg) + (vec_pmhs * n_pmhs))
    return mix.reshape(1, -1), (n_peg, n_pmhs)


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

        if sys_code == "PEG_PMHS":
            c1, c2 = st.columns(2)

            with c1:
                mw_a = st.number_input(
                    "PEG molecular weight must be between 240 to 2100",
                    min_value=SYSTEM3_PEG_MW_RANGE[0],
                    max_value=SYSTEM3_PEG_MW_RANGE[1],
                    value=SYSTEM3_PEG_MW_RANGE[0],
                    step=10.0,
                    key=f"{sys_code}_mw_peg",
                )

            with c2:
                mw_b = st.number_input(
                    "PMHS molecular weight must be between 240 to 2100",
                    min_value=SYSTEM3_PMHS_MW_RANGE[0],
                    max_value=SYSTEM3_PMHS_MW_RANGE[1],
                    value=SYSTEM3_PMHS_MW_RANGE[0],
                    step=10.0,
                    key=f"{sys_code}_mw_pmhs",
                )

            wt_total = st.number_input(
                "The PEG + PMHS additive percentage to be added to the coating ranges from 0.01 to 0.1",
                min_value=SYSTEM3_WTOTAL_RANGE[0],
                max_value=SYSTEM3_WTOTAL_RANGE[1],
                value=SYSTEM3_WTOTAL_RANGE[0],
                step=0.001,
                format="%.3f",
                key=f"{sys_code}_wt_total",
            )

            return {"system": sys_code, "mw_a": mw_a, "mw_b": mw_b, "wt_total": wt_total}

        c1, c2 = st.columns(2)

        if sys_code == "SBMA_PDMS":
            A_add = st.number_input(
                "SBMA + PDMS additive percentage to add to the coating",
                min_value=SYSTEM1_A_RANGE[0],
                max_value=SYSTEM1_A_RANGE[1],
                value=SYSTEM1_A_RANGE[0],
                step=0.001,
                format="%.3f",
                key=f"{sys_code}_A_add",
            )
        elif sys_code == "PDMS_PEG":
            A_add = st.number_input(
                "PDMS + PEG additive percentage to be added to the coating: 0.1–0.4",
                min_value=SYSTEM2_A_RANGE[0],
                max_value=SYSTEM2_A_RANGE[1],
                value=SYSTEM2_A_RANGE[0],
                step=0.01,
                format="%.2f",
                key=f"{sys_code}_A_add",
            )

        with c1:
            if sys_code == "PDMS_PEG":
                mw_a = float(st.selectbox("PDMS molecular weight must be 750", options=[750.0], key=f"{sys_code}_mw_a_select"))
            else:
                mw_a = st.number_input(
                    "SBMA molecular weight must be between 500 to 2500",
                    min_value=SBMA_MW_RANGE[0],
                    max_value=SBMA_MW_RANGE[1],
                    value=SBMA_MW_RANGE[0],
                    step=10.0,
                    key=f"{sys_code}_mw_a",
                )

            p_a = st.number_input(
                "The content % used additive must be between 0 to 1.",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key=f"{sys_code}_p_a",
            )

        with c2:
            if sys_code == "PDMS_PEG":
                mw_b = float(st.selectbox("PEG molecular weight must be 1000", options=[1000.0], key=f"{sys_code}_mw_b_select"))
            else:
                mw_b = st.number_input(
                    "PDMS molecular weight must be between 1000 to 5000",
                    min_value=PDMS_MW_RANGE[0],
                    max_value=PDMS_MW_RANGE[1],
                    value=PDMS_MW_RANGE[0],
                    step=10.0,
                    key=f"{sys_code}_mw_b",
                )

            p_b = st.number_input(
                "The content % of the second additive must be between 0 and 1.",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key=f"{sys_code}_p_b",
            )

        if (p_a + p_b) == 0:
            st.error("p values cannot both be zero.")
            return None

        return {"system": sys_code, "mw_a": mw_a, "p_a": p_a, "mw_b": mw_b, "p_b": p_b, "A_add": A_add}


user_requests = []
for label in systems_to_show:
    out = system_expander(label)
    if out is not None:
        user_requests.append(out)

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

req_s1 = next((r for r in user_requests if r["system"] == "SBMA_PDMS"), None)
if req_s1 is not None:
    try:
        row_sbma = get_pure_row(df_pure, "SBMA")
        row_pdms = get_pure_row(df_pure, "PDMS")

        X_mix1, _ = build_mix_system1(
            row_sbma, row_pdms, feature_cols,
            req_s1["mw_a"], req_s1["p_a"],
            req_s1["mw_b"], req_s1["p_b"],
            req_s1["A_add"],
        )
    except Exception as e:
        st.error(f"Failed to build System 1 mix descriptors: {e}")
        st.stop()

req_s2 = next((r for r in user_requests if r["system"] == "PDMS_PEG"), None)
if req_s2 is not None:
    try:
        row_pdms = get_pure_row(df_pure, "PDMS")
        row_peg = get_pure_row(df_pure, "PEG")

        X_mix2, _ = build_mix_system2(
            row_pdms, row_peg, feature_cols,
            req_s2["mw_a"], req_s2["p_a"],
            req_s2["mw_b"], req_s2["p_b"],
            req_s2["A_add"],
        )
    except Exception as e:
        st.error(f"Failed to build System 2 mix descriptors: {e}")
        st.stop()

req_s3 = next((r for r in user_requests if r["system"] == "PEG_PMHS"), None)
if req_s3 is not None:
    try:
        row_peg = get_pure_row(df_pure, "PEG")
        row_pmhs = get_pure_row(df_pure, "PMHS")

        X_mix3, _ = build_mix_system3(
            row_peg, row_pmhs, feature_cols,
            req_s3["mw_a"], req_s3["mw_b"],
            req_s3["wt_total"],
        )
    except Exception as e:
        st.error(f"Failed to build System 3 mix descriptors: {e}")
        st.stop()

st.markdown("---")
st.header("2) Model prediction")

from sklearn.preprocessing import StandardScaler
import pickle

# Model files, training CSV, and descriptors are selected above from TARGET_OPTIONS.


def read_training_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="UTF-8")
    if df.shape[1] == 1:
        df = pd.read_csv(path, encoding="UTF-8", delimiter=";")
    df.columns = df.columns.astype(str).str.strip()
    return df


@st.cache_resource
def load_scaler_model_and_ad(train_csv_path: str, model_path: str):
    data = read_training_csv(train_csv_path)

    data["prediction_training"] = data["prediction_training"].astype(str).str.strip().str.lower()
    train_data = data[data["prediction_training"] == "training"]

    X_train = train_data[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)

    n = X_train_scaled.shape[0]
    p = X_train_scaled.shape[1]

    X_design = np.column_stack([np.ones(n), X_train_scaled])
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)

    h_star = 3.0 * (p + 1) / n

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return scaler, model, xtx_inv, float(h_star)


def prepare_X_for_model(X_mix: np.ndarray, all_feature_cols: list) -> np.ndarray:
    missing_features = [f for f in MODEL_FEATURES if f not in all_feature_cols]

    if missing_features:
        raise ValueError(
            "The following model descriptors are missing from the pure descriptor CSV: "
            + ", ".join(missing_features)
        )

    idx = [all_feature_cols.index(f) for f in MODEL_FEATURES]
    X_model = X_mix[:, idx].astype(float)
    return X_model


def compute_leverage(x_scaled_row: np.ndarray, xtx_inv: np.ndarray) -> float:
    x = np.concatenate([[1.0], x_scaled_row.ravel()]).reshape(-1, 1)
    h = float((x.T @ xtx_inv @ x).ravel()[0])
    return h


def style_confidence(df: pd.DataFrame):

    def _color_conf(v):
        v = str(v).strip().upper()

        if v == "HIGH":
            return "background-color: #0b7a0b; color: white; font-weight: bold;"

        if v == "LOW":
            return "background-color: #b00020; color: white; font-weight: bold;"

        return ""

    return df.style.map(_color_conf, subset=["Confidence"])


try:
    scaler, model, xtx_inv, h_star = load_scaler_model_and_ad(TRAIN_CSV_PATH, MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load scaler/model/AD: {e}")
    st.stop()

mix_map = {}
if X_mix1 is not None:
    mix_map[f"System 1 (% removal {PRESSURE_LABEL})"] = X_mix1
if X_mix2 is not None:
    mix_map[f"System 2 (% removal {PRESSURE_LABEL})"] = X_mix2
if X_mix3 is not None:
    mix_map[f"System 3 (% removal {PRESSURE_LABEL})"] = X_mix3

if not mix_map:
    st.warning("No mix descriptors available. Please run at least one system.")
else:
    if st.button("Predict fouling release"):
        rows = []

        for col_name, X_mix in mix_map.items():
            try:
                X_model = prepare_X_for_model(X_mix, feature_cols)

                X_scaled_ad = scaler.transform(X_model)
                X_scaled_model = X_scaled_ad.copy()

                expected = getattr(model, "n_features_in_", X_scaled_model.shape[1])

                if X_scaled_model.shape[1] < expected:
                    missing = expected - X_scaled_model.shape[1]
                    X_scaled_model = np.hstack(
                        [X_scaled_model, np.zeros((X_scaled_model.shape[0], missing))]
                    )

                elif X_scaled_model.shape[1] > expected:
                    X_scaled_model = X_scaled_model[:, :expected]

                y_hat = model.predict(X_scaled_model)

                h = compute_leverage(X_scaled_ad[0, :], xtx_inv)
                inside_ad = h <= h_star

                rows.append({
                    "Std_residual": "Inside AD" if inside_ad else "Outside AD",
                    "Confidence": "HIGH" if inside_ad else "LOW",
                    col_name: float(y_hat[0]),
                })

            except Exception as e:
                rows.append({
                    "Std_residual": "Outside AD",
                    "Confidence": "LOW",
                    col_name: np.nan,
                    "Error": str(e),
                })

        df_out = pd.DataFrame(rows)
        st.subheader(OUTPUT_NAME)
        st.dataframe(style_confidence(df_out), use_container_width=True)

        st.download_button(
            "Download predictions",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=OUTPUT_CSV_NAME,
            mime="text/csv",
        )
    
