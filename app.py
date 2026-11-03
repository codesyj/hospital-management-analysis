"""
Hospital Management Analytics Dashboard  (version 2)
====================================================
Run from the project folder:

    pip install -r requirements.txt
    streamlit run app.py

Needs:  data/cleaned/healthcare_cleaned.csv   (created by the cleaning notebook)
        reports/cleaning_audit.csv            (optional)
You can also upload the cleaned CSV from the sidebar.

Sections: Overview | Distributions | Billing | Trends | Advanced analytics |
          Data quality | Prediction model | Data & export
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.cluster.hierarchy import dendrogram, linkage

    SCIPY_AVAILABLE = True
except Exception:  # SciPy missing -> dendrogram is skipped
    SCIPY_AVAILABLE = False

st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    APP_DIR / "data" / "cleaned" / "healthcare_cleaned.csv",
    APP_DIR / "healthcare_cleaned.csv",
    Path("data/cleaned/healthcare_cleaned.csv"),
    Path("healthcare_cleaned.csv"),
]
AUDIT_PATHS = [APP_DIR / "reports" / "cleaning_audit.csv", Path("reports/cleaning_audit.csv")]

REQUIRED_COLUMNS = ["age", "gender", "medical_condition", "admission_type", "billing_amount"]

# Spelling / abbreviation variants found during EDA (see report, section 6)
CONDITION_MAP = {
    "Asthama": "Asthma",
    "Obesty": "Obesity",
    "Arthritus": "Arthritis",
    "Cancr": "Cancer",
    "Diabtes": "Diabetes",
    "Hypertention": "Hypertension",
}
ADMISSION_MAP = {
    "Eme.": "Emergency",
    "Ele.": "Elective",
    "Urg.": "Urgent",
    "Emerg": "Emergency",
    "Elect": "Elective",
}
GENDER_MAP = {"m": "Male", "male": "Male", "mal": "Male",
              "f": "Female", "female": "Female", "femal": "Female", "femlae": "Female", "other": "Other"}
VALID_BLOOD_TYPES = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}

# Found by running the dashboard on the real cleaned CSV: these were still messy there.
INSURANCE_MAP = {
    "bluecross": "Blue Cross", "blue cross": "Blue Cross", "bcbs": "Blue Cross",
    "aetna": "Aetna", "aetna inc": "Aetna",
    "medicare": "Medicare", "medi-care": "Medicare",
    "cigna": "Cigna",
    "unitedhealthcare": "UnitedHealthcare", "united healthcare": "UnitedHealthcare", "uhc": "UnitedHealthcare",
}
MEDICATION_MAP = {
    "ibuprofin": "Ibuprofen", "paracetemol": "Paracetamol", "asprin": "Aspirin", "penicilin": "Penicillin",
}


def normalize_labels(series: pd.Series, mapping: dict, strip_punct: bool = False) -> pd.Series:
    """Case-insensitive lookup in `mapping`; unmapped values are trimmed and title-cased.
    Repeated spaces are collapsed; with strip_punct=True characters such as '.' and '?' are removed."""
    s = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    if strip_punct:
        s = s.str.replace(r"[^A-Za-z ]", "", regex=True).str.strip()
    mapped = s.str.lower().map(mapping)
    return mapped.where(mapped.notna(), s.str.title())


def clean_blood_type(series: pd.Series) -> pd.Series:
    """'a positive', 'A +', 'OPOSITIVE' -> 'A+', 'A+', 'O+'; anything that is not a real group -> missing."""
    s = (series.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
         .str.replace("POSITIVE", "+", regex=False).str.replace("NEGATIVE", "-", regex=False))
    return s.where(s.isin(VALID_BLOOD_TYPES), np.nan)


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
COLUMN_ALIASES = {
    "medical_condition": ["condition", "diagnosis"],
    "date_of_admission": ["admission_date", "date_admitted"],
    "discharge_date": ["date_of_discharge"],
    "billing_amount": ["bill_amount", "amount"],
    "insurance_provider": ["insurance", "insurance_company"],
    "admission_type": ["type_of_admission"],
    "room_number": ["room_no"],
    "blood_type": ["bloodgroup", "blood_group"],
    "test_results": ["test_result"],
    "length_of_stay": ["stay_days", "los"],
    "gender": ["sex"],
    "age": ["patient_age"],
    "hospital": ["hospital_name"],
    "medication": ["medicine"],
    "name": ["patient_name"],
}

BILLING_VIEWS = [
    "Non-negative, outliers removed (IQR rule)",
    "Non-negative, all values",
    "All values (including negatives)",
]


def standardize_name(name) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Make the cleaned CSV analysis-ready (same rules as the notebook's dashboard cell)."""
    df = raw.copy()
    df.columns = [standardize_name(c) for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.match(r"^unnamed(:_\d+)?$")]

    for target, options in COLUMN_ALIASES.items():
        if target not in df.columns:
            for option in options:
                if option in df.columns:
                    df = df.rename(columns={option: target})
                    break

    # Billing may still contain "$" and "," if a raw file is uploaded
    if "billing_amount" in df.columns and not pd.api.types.is_numeric_dtype(df["billing_amount"]):
        df["billing_amount"] = df["billing_amount"].astype("string").str.replace(r"[$,]", "", regex=True)

    for col in ["age", "billing_amount", "room_number", "length_of_stay"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["date_of_admission", "discharge_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    if "age" in df.columns:  # impossible ages -> missing
        df.loc[~df["age"].between(0, 120), "age"] = np.nan

    if "medical_condition" in df.columns:
        df["medical_condition"] = (
            df["medical_condition"].astype("string").str.strip().str.title().replace(CONDITION_MAP)
        )
    if "admission_type" in df.columns:
        df["admission_type"] = (
            df["admission_type"].astype("string").str.strip().str.title().replace(ADMISSION_MAP)
        )
    if "insurance_provider" in df.columns:
        df["insurance_provider"] = normalize_labels(df["insurance_provider"], INSURANCE_MAP)
    if "medication" in df.columns:
        df["medication"] = normalize_labels(df["medication"], MEDICATION_MAP)
    if "gender" in df.columns:
        df["gender"] = normalize_labels(df["gender"], GENDER_MAP, strip_punct=True)
    if "test_results" in df.columns:
        df["test_results"] = normalize_labels(df["test_results"], {}, strip_punct=True)
    if "blood_type" in df.columns:
        df["blood_type"] = clean_blood_type(df["blood_type"])

    if "length_of_stay" not in df.columns and {"date_of_admission", "discharge_date"} <= set(df.columns):
        df["length_of_stay"] = (df["discharge_date"] - df["date_of_admission"]).dt.days
    if "length_of_stay" in df.columns:  # a negative stay means the dates are inconsistent
        df.loc[df["length_of_stay"] < 0, "length_of_stay"] = np.nan

    if "billing_amount" in df.columns:
        df["billing_review"] = np.where(
            df["billing_amount"] < 0,
            "Negative / review",
            np.where(df["billing_amount"].isna(), "Missing", "Non-negative"),
        )

    # Plotly cannot group by pandas' nullable "string" type when it contains pd.NA,
    # so store text columns as plain objects with NaN for missing values.
    for col in df.select_dtypes(include="string").columns:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


@st.cache_data(show_spinner="Loading and preparing data...")
def load_from_path(path: str, mtime: float) -> pd.DataFrame:
    """`mtime` is only part of the cache key, so edits to the CSV refresh the app."""
    return prepare_data(pd.read_csv(path, low_memory=False))


@st.cache_data(show_spinner="Loading and preparing data...")
def load_from_bytes(content: bytes) -> pd.DataFrame:
    return prepare_data(pd.read_csv(io.BytesIO(content), low_memory=False))


def iqr_summary(values: pd.Series, column: str) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = (v < lower) | (v > upper)
    return {
        "Column": column,
        "Q1": round(q1, 2),
        "Q3": round(q3, 2),
        "Lower bound": round(lower, 2),
        "Upper bound": round(upper, 2),
        "Flagged values": int(flagged.sum()),
        "Percent flagged": round(flagged.mean() * 100, 2),
    }


def apply_filters(df, selections: dict, age_range, keep_missing_age) -> pd.DataFrame:
    """selections maps column name -> list of chosen values (empty list = no filter)."""
    mask = pd.Series(True, index=df.index)
    for col, chosen in selections.items():
        if chosen and col in df.columns:
            mask &= df[col].astype(str).isin(chosen)
    age_ok = df["age"].between(age_range[0], age_range[1])
    if keep_missing_age:
        age_ok |= df["age"].isna()
    return df[mask & age_ok]


def billing_subset(data: pd.DataFrame, view: str, upper_bound: float) -> pd.DataFrame:
    out = data.dropna(subset=["billing_amount"])
    if view == BILLING_VIEWS[2]:
        return out
    out = out[out["billing_amount"] >= 0]
    if view == BILLING_VIEWS[0]:
        out = out[out["billing_amount"] <= upper_bound]
    return out


@st.cache_data(show_spinner=False)
def to_csv_bytes(_data: pd.DataFrame, key: tuple) -> bytes:
    """`_data` is not hashed (leading underscore); `key` identifies the filter state."""
    return _data.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------
# Model layer (same pipeline as the notebook)
# --------------------------------------------------------------------------
MODEL_FEATURES = ["age", "gender", "blood_type", "admission_type"]
MODEL_TARGET = "medical_condition"


@st.cache_data(show_spinner="Training Random Forest...")
def train_model(model_df: pd.DataFrame) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    m = model_df.astype(object)
    m = m.where(pd.notna(m), np.nan)
    m["age"] = pd.to_numeric(m["age"], errors="coerce")
    m = m.dropna(subset=[MODEL_TARGET])

    X = m[MODEL_FEATURES]
    y = m[MODEL_TARGET].astype(str)

    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), ["age"]),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["gender", "blood_type", "admission_type"],
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=30, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=1
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = (
        pd.DataFrame({"Feature": names, "Importance": model.named_steps["classifier"].feature_importances_})
        .assign(Feature=lambda d: d["Feature"].str.replace(r"^(numeric|categorical)__", "", regex=True))
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    classes = sorted(y.unique())
    report = pd.DataFrame(classification_report(y_test, preds, output_dict=True, zero_division=0)).T
    comparison = (
        pd.DataFrame({"Actual": y_test.value_counts(), "Predicted": pd.Series(preds).value_counts()})
        .fillna(0)
        .sort_index()
        .reset_index()
    )
    comparison.columns = ["Medical condition", "Actual", "Predicted"]

    return {
        "rows": len(m),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": classes,
        "accuracy": float(accuracy_score(y_test, preds)),
        "baseline": float(y_test.value_counts(normalize=True).max()),
        "report": report,
        "importance": importance,
        "comparison": comparison,
        "confusion": pd.DataFrame(
            confusion_matrix(y_test, preds, labels=classes), index=classes, columns=classes
        ),
    }


# --------------------------------------------------------------------------
# Styling helpers
# --------------------------------------------------------------------------
PALETTE = ["#2F5597", "#ED7D31", "#2E9E6B", "#C0392B", "#8E6BBF", "#8C564B", "#17A2B8", "#7F7F7F"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    [data-testid="stMetric"] {background:#F5F7FA; border:1px solid #E1E6EE; border-left:5px solid #2F5597;
                              padding:12px 16px; border-radius:8px;}
    [data-testid="stMetricLabel"] {color:#5A6270;}
    [data-testid="stMetricLabel"] p {white-space: normal;}
    [data-testid="stMetricValue"] {font-size: 1.7rem; color:#1F3864;}
    .banner {background:linear-gradient(90deg,#1F3864,#2F5597); color:white; padding:18px 24px;
             border-radius:10px; margin-bottom:14px;}
    .banner h1 {color:white; margin:0; font-size:1.9rem;}
    .banner p {margin:4px 0 0 0; opacity:.85;}
    </style>
    """,
    unsafe_allow_html=True,
)


def show(fig, container=st):
    """Draw a Plotly figure with consistent margins and title size."""
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), title_font_size=16, template="plotly_white",
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#1F2937"))
    container.plotly_chart(fig, theme=None)  # theme=None keeps our light styling in every browser theme


def labelled(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Copy of `frame` where missing category values are shown as 'Missing' (safe for px colour/x)."""
    out = frame.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(object).where(out[c].notna(), "Missing")
    return out


def counts_of(frame: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    return (
        frame[col].fillna("Missing").astype(str).value_counts()
        .rename_axis(label).reset_index(name="Records")
    )


def bar_counts(frame, col, title, label, horizontal=False):
    counts = counts_of(frame, col, label)
    if horizontal:
        fig = px.bar(counts, x="Records", y=label, color=label, text_auto=True, title=title, orientation="h")
    else:
        fig = px.bar(counts, x=label, y="Records", color=label, text_auto=True, title=title)
    fig.update_layout(showlegend=False)
    return fig


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
st.sidebar.title("🏥 Hospital Dashboard")
st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader("Upload cleaned CSV (optional)", type="csv")

if uploaded is not None:
    df = load_from_bytes(uploaded.getvalue())
    source_label = uploaded.name
else:
    found = next((p for p in CANDIDATE_PATHS if p.exists()), None)
    if found is None:
        st.title("Hospital Analytics Dashboard")
        st.error(
            "Cleaned dataset not found. Place `healthcare_cleaned.csv` in `data/cleaned/` "
            "next to this file, or upload it from the sidebar."
        )
        st.stop()
    df = load_from_path(str(found), found.stat().st_mtime)
    source_label = str(found)

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    st.title("Hospital Analytics Dashboard")
    st.error(f"The dataset is missing required columns: {', '.join(missing_cols)}")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
def options_for(col: str) -> list[str]:
    return sorted(df[col].dropna().astype(str).unique()) if col in df.columns else []


st.sidebar.subheader("Filters")
FILTER_COLUMNS = [
    ("medical_condition", "Medical condition"),
    ("admission_type", "Admission type"),
    ("gender", "Gender"),
    ("insurance_provider", "Insurance provider"),
    ("blood_type", "Blood type"),
    ("test_results", "Test results"),
]
selections = {}
for col, label in FILTER_COLUMNS:
    if col in df.columns:
        selections[col] = st.sidebar.multiselect(f"{label} (empty = all)", options_for(col))

age_lo = int(df["age"].min()) if df["age"].notna().any() else 0
age_hi = int(df["age"].max()) if df["age"].notna().any() else 120
sel_age = st.sidebar.slider("Age range", age_lo, age_hi, (age_lo, age_hi))
keep_missing_age = st.sidebar.checkbox("Include records with unknown age", value=True)

st.sidebar.subheader("Chart options")
billing_view = st.sidebar.radio("Billing charts show", BILLING_VIEWS)
sample_size = st.sidebar.slider("Plot sample (scatter / advanced charts)", 250, 10000, 2500, 250)

data = apply_filters(df, selections, sel_age, keep_missing_age)
filter_key = (
    tuple((k, tuple(v)) for k, v in selections.items()), sel_age, keep_missing_age, len(df)
)

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.markdown(
    '<div class="banner"><h1>🏥 Hospital Analytics Dashboard</h1>'
    f"<p>{len(df):,} patient records · source: {Path(source_label).name}</p></div>",
    unsafe_allow_html=True,
)

if data.empty:
    st.warning("No records match the current filters. Widen the filters in the sidebar.")
    st.stop()

nonneg_billing = df.loc[df["billing_amount"] >= 0, "billing_amount"]
billing_upper = (
    iqr_summary(nonneg_billing, "billing_amount")["Upper bound"] if nonneg_billing.notna().any() else float("inf")
)

k1, k2, k3 = st.columns(3)
k4, k5, k6 = st.columns(3)
k1.metric("Filtered records", f"{len(data):,}", delta=f"{len(data) / len(df):.0%} of all", delta_color="off")
k2.metric("Unique patients", f"{data['name'].nunique():,}" if "name" in data.columns else "N/A")
k3.metric("Average age", f"{data['age'].mean():.1f}")
k4.metric("Median billing", f"{data['billing_amount'].median():,.0f}")
k5.metric("Avg stay (days)", f"{data['length_of_stay'].mean():.1f}" if "length_of_stay" in data.columns else "N/A")
k6.metric("Negative bills", f"{int((data['billing_amount'] < 0).sum()):,}")

section = st.radio(
    "Section",
    ["Overview", "Distributions", "Billing", "Trends", "Advanced analytics",
     "Data quality", "Prediction model", "Data & export"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()

bill_df = billing_subset(data, billing_view, billing_upper)
HAS_STAY = "length_of_stay" in data.columns

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
if section == "Overview":
    c1, c2, c3 = st.columns(3)
    show(bar_counts(data, "medical_condition", "Records by medical condition", "Condition"), c1)
    adm = counts_of(data, "admission_type", "Admission type")
    show(px.pie(adm, names="Admission type", values="Records", hole=0.45, title="Admission type"), c2)
    gen = counts_of(data, "gender", "Gender")
    show(px.pie(gen, names="Gender", values="Records", hole=0.45, title="Gender"), c3)

    c4, c5 = st.columns(2)
    ages = data.dropna(subset=["age"])
    if not ages.empty:
        show(px.histogram(ages, x="age", nbins=30, marginal="box", title="Patient age distribution"), c4)
    if "blood_type" in data.columns:
        show(bar_counts(data, "blood_type", "Records by blood type", "Blood type", True), c5)

    c6, c7, c8 = st.columns(3)
    if "insurance_provider" in data.columns:
        show(bar_counts(data, "insurance_provider", "Insurance providers", "Provider", True), c6)
    if "medication" in data.columns:
        show(bar_counts(data, "medication", "Medication", "Medication", True), c7)
    if "test_results" in data.columns:
        res = counts_of(data, "test_results", "Result")
        show(px.pie(res, names="Result", values="Records", title="Test results"), c8)

# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------
elif section == "Distributions":
    d1, d2 = st.columns(2)
    show(px.box(labelled(data.dropna(subset=["age"]), ["medical_condition"]), x="medical_condition", y="age", color="medical_condition",
                points=False, title="Age distribution by condition"), d1)

    ct = pd.crosstab(data["medical_condition"], data["admission_type"], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars="medical_condition", var_name="Admission type", value_name="Percent")
    show(px.bar(ct, x="medical_condition", y="Percent", color="Admission type",
                title="Admission type mix by condition (%)"), d2)

    if HAS_STAY:
        stay = data.dropna(subset=["medical_condition", "length_of_stay"])
        d3, d4 = st.columns(2)
        if not stay.empty:
            show(px.box(stay, x="medical_condition", y="length_of_stay", color="medical_condition",
                        points=False, title="Length of stay by condition"), d3)
        valid_stays = data[data["length_of_stay"].between(0, 365)]
        if not valid_stays.empty:
            show(px.histogram(valid_stays, x="length_of_stay", nbins=40, marginal="box",
                              title="Hospital stay length (0-365 days)"), d4)

    d5, d6 = st.columns(2)
    if not bill_df.empty:
        show(px.box(labelled(bill_df, ["medical_condition"]), x="medical_condition", y="billing_amount", color="medical_condition",
                    points=False, title="Billing by condition"), d5)
        show(px.box(labelled(bill_df, ["admission_type"]), x="admission_type", y="billing_amount", color="admission_type",
                    points=False, title="Billing by admission type"), d6)
    st.caption(f"Billing charts: {billing_view.lower()} ({len(bill_df):,} records). Change in the sidebar.")

# --------------------------------------------------------------------------
# Billing
# --------------------------------------------------------------------------
elif section == "Billing":
    billed = data["billing_amount"].dropna()
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Billed records", f"{len(billed):,}")
    b2.metric("Median bill", f"{billed.median():,.0f}")
    b3.metric("Mean bill", f"{billed.mean():,.0f}")
    b4.metric("Largest bill", f"{billed.max():,.0f}")
    st.caption("The mean is far above the median because of a few multi-million-dollar records - use the median.")

    e1, e2 = st.columns(2)
    if not bill_df.empty:
        show(px.histogram(bill_df, x="billing_amount", nbins=45, marginal="box",
                          title="Billing amount distribution"), e1)
        sample = labelled(bill_df.sample(min(len(bill_df), sample_size), random_state=42), ["medical_condition"])
        show(px.scatter(sample, x="age", y="billing_amount", color="medical_condition", opacity=0.5,
                        title="Age vs billing amount (sample)"), e2)

    f1, f2 = st.columns(2)
    if "insurance_provider" in bill_df.columns and not bill_df.empty:
        by_ins = (bill_df.groupby("insurance_provider")["billing_amount"].median()
                  .sort_values().reset_index())
        show(px.bar(by_ins, x="billing_amount", y="insurance_provider", orientation="h",
                    title="Median bill by insurance provider"), f1)
    by_cond = (bill_df.groupby("medical_condition")["billing_amount"].median()
               .sort_values().reset_index()) if not bill_df.empty else pd.DataFrame()
    if not by_cond.empty:
        show(px.bar(by_cond, x="billing_amount", y="medical_condition", orientation="h",
                    title="Median bill by condition"), f2)

    st.subheader("Billing statistics by condition (all billed records)")
    stats = (data.groupby("medical_condition")["billing_amount"]
             .agg(["count", "mean", "median", "min", "max"]).round(2)
             .sort_values("mean", ascending=False))
    st.dataframe(stats)

    st.subheader("Ten largest bills")
    cols = [c for c in ["medical_condition", "admission_type", "insurance_provider", "billing_amount"] if c in data.columns]
    st.dataframe(data.nlargest(10, "billing_amount")[cols])

# --------------------------------------------------------------------------
# Trends
# --------------------------------------------------------------------------
elif section == "Trends":
    dated = data.dropna(subset=["date_of_admission"])
    today = pd.Timestamp.today().normalize()
    valid = dated[dated["date_of_admission"] <= today]
    n_future = len(dated) - len(valid)

    split_by = st.radio("Split monthly chart by", ["None", "Medical condition", "Admission type"], horizontal=True)
    month = valid["date_of_admission"].dt.to_period("M").dt.to_timestamp()
    if split_by == "None":
        monthly = valid.groupby(month).size().reset_index()
        monthly.columns = ["Month", "Admissions"]
        fig = px.line(monthly, x="Month", y="Admissions", title="Monthly hospital admissions")
    else:
        split_col = "medical_condition" if split_by == "Medical condition" else "admission_type"
        monthly = valid.groupby([month, valid[split_col].fillna("Missing")]).size().reset_index()
        monthly.columns = ["Month", split_by, "Admissions"]
        fig = px.line(monthly, x="Month", y="Admissions", color=split_by,
                      title=f"Monthly admissions by {split_by.lower()}")
    show(fig)

    t1, t2 = st.columns(2)
    yearly = valid.groupby(valid["date_of_admission"].dt.year).size().reset_index()
    yearly.columns = ["Year", "Admissions"]
    show(px.bar(yearly, x="Year", y="Admissions", text_auto=True, title="Admissions per year"), t1)

    moy = valid.groupby(valid["date_of_admission"].dt.month).size().reset_index()
    moy.columns = ["Month of year", "Admissions"]
    show(px.bar(moy, x="Month of year", y="Admissions", title="Admissions by calendar month"), t2)

    billed_dates = valid.dropna(subset=["billing_amount"])
    billed_dates = billed_dates[billed_dates["billing_amount"] >= 0]
    if not billed_dates.empty:
        med = (billed_dates.groupby(billed_dates["date_of_admission"].dt.to_period("M").dt.to_timestamp())
               ["billing_amount"].median().reset_index())
        med.columns = ["Month", "Median billing"]
        show(px.line(med, x="Month", y="Median billing", title="Median billing by admission month"))

    st.caption(
        f"{n_future:,} records dated after today ({today:%d %b %Y}) and "
        f"{len(data) - len(dated):,} records without an admission date are excluded from these charts."
    )

# --------------------------------------------------------------------------
# Advanced analytics (all notebook Plotly charts)
# --------------------------------------------------------------------------
elif section == "Advanced analytics":
    plot_data = data.sample(sample_size, random_state=42) if len(data) > sample_size else data.copy()
    st.caption(f"Charts below use a random sample of {len(plot_data):,} filtered records for responsiveness.")

    numeric_cols = plot_data.select_dtypes(include=np.number).columns.tolist()
    a1, a2 = st.columns(2)
    if len(numeric_cols) >= 2:
        corr = plot_data[numeric_cols].corr(numeric_only=True)
        show(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                       zmin=-1, zmax=1, title="Correlation heatmap"), a1)

    xy = plot_data[["age", "billing_amount"]].dropna()
    xy = xy[xy["billing_amount"] >= 0]
    if billing_view == BILLING_VIEWS[0]:
        xy = xy[xy["billing_amount"] <= billing_upper]
    if len(xy) >= 10:
        show(px.density_contour(xy, x="age", y="billing_amount", nbinsx=25, nbinsy=25,
                                title="Contour density: age vs billing",
                                labels={"age": "Age", "billing_amount": "Billing amount"}), a2)

    agg_name = st.radio("Heatmap statistic", ["Median", "Mean"], horizontal=True)
    pivot = bill_df.pivot_table(index="medical_condition", columns="admission_type",
                                values="billing_amount", aggfunc=agg_name.lower())
    if not pivot.empty:
        show(px.imshow(pivot, text_auto=".0f", aspect="auto", color_continuous_scale="Viridis",
                       title=f"{agg_name} billing by condition and admission type",
                       labels=dict(x="Admission type", y="Condition", color=f"{agg_name} billing")))
        st.caption("Means are pulled up by a few extreme bills; medians are more robust.")

    parallel_cols = [c for c in ["age", "billing_amount", "length_of_stay", "room_number"] if c in plot_data.columns]
    base = bill_df.sample(min(len(bill_df), sample_size), random_state=42) if not bill_df.empty else plot_data
    p = base[parallel_cols + ["medical_condition"]].dropna().head(1500).copy()
    if "room_number" in p.columns and len(p) > 20:       # a few room numbers go up to 9,999
        p = p[p["room_number"] <= p["room_number"].quantile(0.99)]
    if len(parallel_cols) >= 2 and not p.empty:
        p["_condition_code"] = p["medical_condition"].astype("category").cat.codes
        show(px.parallel_coordinates(p, dimensions=parallel_cols, color="_condition_code",
                                     title="Parallel coordinates: patient profiles"))

    radar_fields = [c for c in ["age", "billing_amount", "length_of_stay"] if c in data.columns]
    group_means = bill_df.groupby("medical_condition")[radar_fields].mean().dropna()
    b1, b2 = st.columns(2)
    if len(group_means) >= 1 and len(radar_fields) >= 2:
        normalized = group_means.copy()
        for c in radar_fields:
            lo, hi = normalized[c].min(), normalized[c].max()
            normalized[c] = (normalized[c] - lo) / (hi - lo) if hi != lo else 0
        fig = go.Figure()
        for i, (condition, row) in enumerate(normalized.iterrows()):
            vals = row.tolist()
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=radar_fields + [radar_fields[0]],
                                          fill="toself", name=str(condition)))
        fig.update_layout(title="Radar: normalised averages by condition",
                          polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        show(fig, b1)
        b1.caption("Min-max scaling stretches small differences to the full 0-1 range.")

    positive = data.loc[data["billing_amount"] > 0, "billing_amount"].dropna()
    if len(positive) > 10:
        edges = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 41)
        counts_, _ = np.histogram(positive, bins=edges)
        log_hist = pd.DataFrame({"Billing amount": np.sqrt(edges[:-1] * edges[1:]), "Records": counts_})
        show(px.bar(log_hist, x="Billing amount", y="Records", log_x=True,
                    title="Positive billing on a logarithmic axis"), b2)

    d1, d2 = st.columns(2)
    tri = data[["age", "billing_amount", "length_of_stay"]].dropna()
    tri = tri[(tri >= 0).all(axis=1)]
    if len(tri) > 10:
        ranks = tri.rank(pct=True)                      # 0-1 percentile rank of each measure
        props = ranks.div(ranks.sum(axis=1), axis=0).mul(100).sample(min(len(ranks), 1500), random_state=42)
        show(px.scatter_ternary(props, a="age", b="billing_amount", c="length_of_stay",
                                title="Ternary: relative rank of age / billing / stay",
                                labels={"age": "Age share", "billing_amount": "Billing share",
                                        "length_of_stay": "Stay share"}, hover_data=[]), d1)
        d1.caption("Each measure is converted to a percentile rank first, so no single measure dominates. "
                   "Exploratory view only - not a clinical score.")

    if SCIPY_AVAILABLE:
        if len(group_means) >= 3:
            grouped = group_means
            std = grouped.std(ddof=0).replace(0, 1)
            standardized = (grouped - grouped.mean()) / std
            Z = linkage(standardized.fillna(0).values, method="ward")
            dendro = dendrogram(Z, labels=grouped.index.astype(str).tolist(), no_plot=True)
            fig = go.Figure()
            for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip"))
            order = dendro["leaves"]
            fig.update_layout(
                title="Dendrogram: similarity of conditions",
                xaxis=dict(tickmode="array", tickvals=[5 + 10 * i for i in range(len(order))],
                           ticktext=[str(grouped.index[i]) for i in order]),
                yaxis_title="Cluster distance",
            )
            show(fig, d2)
    else:
        d2.info("SciPy is not installed, so the dendrogram is skipped.")

# --------------------------------------------------------------------------
# Data quality (full dataset)
# --------------------------------------------------------------------------
elif section == "Data quality":
    st.caption("This section describes the full loaded dataset and ignores the sidebar filters.")
    neg_bills = int((df["billing_amount"] < 0).sum())
    if {"date_of_admission", "discharge_date"} <= set(df.columns):
        bad_stays = int((df["discharge_date"] < df["date_of_admission"]).sum())
        future_dates = int((df["date_of_admission"] > pd.Timestamp.today()).sum())
    else:
        bad_stays, future_dates = 0, 0

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    q3.metric("Negative bills", f"{neg_bills:,}")
    q4.metric("Discharge before admission", f"{bad_stays:,}")
    q5.metric("Future admission dates", f"{future_dates:,}")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values().rename_axis("Column").reset_index(name="Missing values")
    m1, m2 = st.columns(2)
    show(px.bar(missing, x="Missing values", y="Column", orientation="h", title="Missing values by column"), m1)
    pct = missing.assign(Percent=(missing["Missing values"] / len(df) * 100).round(2))
    show(px.bar(pct, x="Percent", y="Column", orientation="h", title="Missing values (% of rows)"), m2)

    outliers = [iqr_summary(df["age"], "age"),
                iqr_summary(df.loc[df["billing_amount"] >= 0, "billing_amount"], "billing_amount")]
    if "length_of_stay" in df.columns:
        outliers.append(iqr_summary(df.loc[df["length_of_stay"] >= 0, "length_of_stay"], "length_of_stay"))
    st.subheader("IQR outlier summary")
    st.dataframe(pd.DataFrame(outliers))

    audit_path = next((p for p in AUDIT_PATHS if p.exists()), None)
    if audit_path is not None:
        st.subheader("Cleaning audit (from the cleaning notebook)")
        st.dataframe(pd.read_csv(audit_path))

# --------------------------------------------------------------------------
# Prediction model
# --------------------------------------------------------------------------
elif section == "Prediction model":
    st.write(
        "Random Forest that tries to predict **medical condition** from age, gender, blood type and "
        "admission type (same settings as the notebook). Trained on the full dataset - sidebar filters do not apply."
    )
    if any(c not in df.columns for c in MODEL_FEATURES + [MODEL_TARGET]):
        st.info("The dataset does not contain all columns needed for the model.")
    else:
        result = train_model(df[MODEL_FEATURES + [MODEL_TARGET]])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']:.2%}")
        m2.metric("Majority-class baseline", f"{result['baseline']:.2%}")
        m3.metric("Random guess (1/k)", f"{1 / len(result['classes']):.2%}")
        m4.metric("Train / test rows", f"{result['n_train']:,} / {result['n_test']:,}")

        if result["accuracy"] <= result["baseline"] + 0.02:
            st.warning(
                "Accuracy is within about two points of guessing the most common class, so these four "
                "features carry almost no information about the diagnosis in this dataset."
            )
        r1, r2 = st.columns(2)
        show(px.bar(result["comparison"], x="Medical condition", y=["Actual", "Predicted"],
                    barmode="group", title="Actual vs predicted (test set)"), r1)
        show(px.imshow(result["confusion"], text_auto=True, aspect="auto", color_continuous_scale="Blues",
                       title="Confusion matrix (test set)",
                       labels=dict(x="Predicted", y="Actual", color="Records")), r2)
        show(px.bar(result["importance"].sort_values("Importance"), x="Importance", y="Feature",
                    orientation="h", title="Top feature importances"))
        st.subheader("Classification report")
        st.dataframe(result["report"].round(3))

# --------------------------------------------------------------------------
# Data & export
# --------------------------------------------------------------------------
else:
    st.write(f"Showing the first 1,000 of {len(data):,} filtered records.")
    st.dataframe(data.head(1000))
    st.download_button(
        "Download filtered records (CSV)",
        data=to_csv_bytes(data, filter_key),
        file_name="hospital_filtered_records.csv",
        mime="text/csv",
    )
