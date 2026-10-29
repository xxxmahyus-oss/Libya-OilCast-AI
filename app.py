import io
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

APP_NAME = "Libya OilCast AI"
st.set_page_config(page_title=APP_NAME, page_icon="🛢️", layout="wide")


# ---------- Styling ----------
st.markdown("""
<style>
    :root {
        --bg-main: #07111f;
        --bg-panel: #0e2034;
        --bg-sidebar: #091728;
        --border-color: #24445f;
        --accent: #4de2d0;
        --accent-blue: #38bdf8;
        --text-main: #e5edf7;
        --text-muted: #9bb0c5;
    }

    /* Main application */
    .stApp {
        background:
            radial-gradient(
                ellipse at 85% 0%,
                rgba(8, 126, 139, 0.12),
                transparent 38%
            ),
            var(--bg-main);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f0f7ff !important;
        font-weight: 750 !important;
        letter-spacing: -0.3px;
    }

    h2 {
        border-bottom: 1px solid rgba(77, 226, 208, 0.16);
        padding-bottom: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #091728 0%,
            #0b1d31 100%
        );
        border-right: 1px solid #203c55;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c6d8e9;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #10253b,
            #0c1b2c
        );
        border: 1px solid #24445f;
        border-top: 2px solid #168b96;
        border-radius: 15px;
        padding: 18px 18px 15px 18px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        transition: border-color 0.2s ease,
                    transform 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #2c8e9d;
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] {
        color: #a9c4da !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-weight: 800;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid #168b96;
        border-radius: 10px;
        background: linear-gradient(
            120deg,
            #103b55,
            #10505b
        );
        color: #ffffff;
        font-weight: 700;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--accent);
        background: #15536a;
        color: #ffffff;
        box-shadow: 0 4px 16px rgba(32, 184, 197, 0.12);
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus {
        box-shadow: 0 0 0 2px rgba(77, 226, 208, 0.25);
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #102238;
        border-color: #294761;
        border-radius: 9px;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #e5edf7;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #24445f;
    }

    .stTabs [data-baseweb="tab"] {
        background: #0e2034;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid var(--accent);
        color: var(--accent) !important;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #24445f;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expanders */
    details[data-testid="stExpander"] {
        background: rgba(14, 32, 52, 0.65);
        border: 1px solid #24445f;
        border-radius: 12px;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Links */
    a {
        color: var(--accent-blue) !important;
    }

    /* Captions */
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    /* Separators */
    hr {
        border-color: #20364c;
    }

    /* Small reusable note */
    .small-note {
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    /* Responsive spacing */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------- Synthetic data ----------
@st.cache_data(show_spinner=False)
def make_synthetic_data(n_wells=8, n_days=900, seed=42):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    fields = ["Sirte Demo Field", "Murzuq Demo Field", "Ghadames Demo Field"]
    rows = []
    for wi in range(n_wells):
        well = f"WELL-{wi+1:03d}"
        field = fields[wi % len(fields)]
        initial = rng.uniform(450, 1500)
        decline = rng.uniform(0.00025, 0.0017)
        phase = rng.uniform(0, 2*np.pi)
        water_cut_start = rng.uniform(0.05, 0.32)
        pressure_start = rng.uniform(2500, 4200)
        for t, date in enumerate(dates):
            decline_component = initial * np.exp(-decline*t)
            seasonal = 1 + 0.025*np.sin(2*np.pi*t/180 + phase)
            operational_noise = rng.normal(1, 0.035)
            production = decline_component * seasonal * operational_noise
            # Occasional shut-ins and partial operating interruptions
            if rng.random() < 0.018:
                production *= rng.uniform(0.0, 0.25)
            production = max(0.0, production)
            water_cut = np.clip(water_cut_start + 0.00022*t + rng.normal(0, 0.012), 0.01, 0.92)
            pressure = max(400, pressure_start - 0.65*t + rng.normal(0, 18))
            choke = np.clip(rng.normal(68 - 0.012*t, 5), 15, 100)
            uptime = np.clip(rng.normal(96, 5), 0, 100)
            rows.append({
                "date": date, "field": field, "well": well,
                "oil_production_bopd": round(production, 2),
                "water_cut_pct": round(water_cut*100, 2),
                "pressure_psi": round(pressure, 1),
                "choke_pct": round(choke, 1),
                "uptime_pct": round(uptime, 1)
            })
    return pd.DataFrame(rows)

FEATURES = ["day_index", "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_mean_7", "rolling_mean_30", "water_cut_pct",
            "pressure_psi", "choke_pct", "uptime_pct", "month"] 

def detect_oil_columns(df): 
    """Detect likely date, oil production, well, and field columns."""
    import re

    def normalize(name):
        name = str(name).strip().lower()
        name = re.sub(r"[\s\-/]+", "_", name)
        return name

    aliases = {
        "date": [
            "date", "datetime", "time", "timestamp",
            "التاريخ", "تاريخ", "اليوم"
        ],
        "oil_production_bopd": [
            "oil_production_bopd", "oil_rate", "oil_production",
            "oil_rate_bopd", "bopd", "oil", "production",
            "daily_oil_production", "oil_volume",
            "إنتاج_النفط", "الانتاج", "إنتاج", "النفط"
        ],
        "well": [
            "well", "well_name", "well_id", "well_number",
            "البئر", "اسم_البئر", "رقم_البئر"
        ],
        "field": [
            "field", "field_name", "oilfield",
            "الحقل", "اسم_الحقل"
        ],
    }

    normalized_columns = {
        col: normalize(col) for col in df.columns
    }

    detected = {}

    for target, names in aliases.items():
        matches = [
            col for col, normalized in normalized_columns.items()
            if normalized in names
        ]
        if matches:
            detected[target] = matches[0]

    return detected 
    
def prepare_well_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["oil_production_bopd"] = pd.to_numeric(df["oil_production_bopd"], errors="coerce")
    df = df.dropna(subset=["date", "oil_production_bopd"])
    df = df[df["oil_production_bopd"] >= 0]
    if "well" not in df:
        df["well"] = "WELL-001"
    if "field" not in df:
        df["field"] = "Imported Field"
    for c, default in [("water_cut_pct", 25.0), ("pressure_psi", 2500.0),
                       ("choke_pct", 65.0), ("uptime_pct", 95.0)]:
        if c not in df:
            df[c] = default
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(default)
    df = df.sort_values(["well", "date"])
    # If duplicate daily records exist for a well, average them rather than silently summing.
    df = df.groupby(["field", "well", "date"], as_index=False).agg({
        "oil_production_bopd":"mean", "water_cut_pct":"mean",
        "pressure_psi":"mean", "choke_pct":"mean", "uptime_pct":"mean"
    })
    return df

def add_features(one):
    one = one.sort_values("date").copy()
    one["day_index"] = (one["date"] - one["date"].min()).dt.days
    y = one["oil_production_bopd"]
    for lag in [1, 7, 14, 30]:
        one[f"lag_{lag}"] = y.shift(lag)
    one["rolling_mean_7"] = y.shift(1).rolling(7).mean()
    one["rolling_mean_30"] = y.shift(1).rolling(30).mean()
    one["month"] = one["date"].dt.month
    return one.dropna(subset=FEATURES + ["oil_production_bopd"])

def model_bank(seed):
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=180, learning_rate=0.07, random_state=seed)
    }
    

@st.cache_resource(show_spinner=False)
def evaluate_models(df, seed=42):
    feature_frames = []

    for (field, well), group in df.groupby(["field", "well"]):
        f = add_features(group)
        if not f.empty:
            feature_frames.append(f)

    if not feature_frames:
        return pd.DataFrame(), pd.DataFrame(), {}

    data = pd.concat(feature_frames, ignore_index=True)
    data = data.sort_values(["date", "well"]).reset_index(drop=True)

    if len(data) < 60:
        return pd.DataFrame(), data, {}

    dates = np.sort(data["date"].unique())

    if len(dates) < 10:
        return pd.DataFrame(), data, {}

    train_cut = int(len(dates) * 0.60)
    val_cut = int(len(dates) * 0.80)

    train_dates = dates[:train_cut]
    val_dates = dates[train_cut:val_cut]
    test_dates = dates[val_cut:]

    train = data[data["date"].isin(train_dates)].copy()
    val = data[data["date"].isin(val_dates)].copy()
    test = data[data["date"].isin(test_dates)].copy()

    if len(train) < 20 or len(val) < 5 or len(test) < 5:
        return pd.DataFrame(), data, {}

    X_train = train[FEATURES]
    y_train = train["oil_production_bopd"]
    X_val = val[FEATURES]
    y_val = val["oil_production_bopd"]

    def calculate_metrics(actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        result = {
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R²": (
                r2_score(actual, predicted)
                if len(actual) > 1 and np.unique(actual).size > 1
                else np.nan
            )
        }
        return result

    rows = []
    validation_scores = {}
    validation_models = {}

    baseline_val = np.maximum(0, val["lag_1"].to_numpy(dtype=float))

    baseline_metrics = calculate_metrics(y_val, baseline_val)
    rows.append({
        "Model": "Persistence Baseline",
        **baseline_metrics
    })
    validation_scores["Persistence Baseline"] = baseline_metrics["RMSE"]

    for name, model in model_bank(seed).items():
        model.fit(X_train, y_train)

        pred_val = np.maximum(
            0, model.predict(X_val)
        )

        model_metrics = calculate_metrics(y_val, pred_val)

        rows.append({
            "Model": name,
            **model_metrics
        })

        validation_scores[name] = model_metrics["RMSE"]
        validation_models[name] = model

    metrics = (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    best_name = min(validation_scores, key=validation_scores.get)

    training_data = pd.concat([train, val], ignore_index=True)

    if best_name == "Persistence Baseline":
        final_model = None
    else:
        final_model = model_bank(seed)[best_name]
        final_model.fit(
            training_data[FEATURES],
            training_data["oil_production_bopd"]
        )

    X_test = test[FEATURES]
    y_test = test["oil_production_bopd"]

    if best_name == "Persistence Baseline":
        test_pred = np.maximum(
            0, test["lag_1"].to_numpy(dtype=float)
        )
    else:
        test_pred = np.maximum(
            0, final_model.predict(X_test)
        )

    test_metrics = calculate_metrics(y_test, test_pred)

    test_predictions = test[
        ["date", "field", "well", "oil_production_bopd"]
    ].copy()

    test_predictions["predicted_oil_production_bopd"] = test_pred
    test_predictions["model"] = best_name

    fitted = dict(validation_models)

    if final_model is not None:
        fitted[best_name] = final_model

    fitted["Persistence Baseline"] = None

    return metrics, test_predictions, {
        "models": fitted,
        "best_name": best_name,
        "features": FEATURES,
        "feature_data": data,
        "test_metrics": test_metrics,
        "validation_scores": validation_scores,
        "split_dates": {
            "train_end": pd.Timestamp(train_dates[-1]),
            "validation_end": pd.Timestamp(val_dates[-1]),
            "test_start": pd.Timestamp(test_dates[0]),
            "test_end": pd.Timestamp(test_dates[-1])
        }
    }

@st.cache_data(show_spinner=False)
def walk_forward_validation(df, seed=42, n_splits=3):
    feature_frames = []

    for (field, well), group in df.groupby(["field", "well"]):
        f = add_features(group)
        if not f.empty:
            feature_frames.append(f)

    if not feature_frames:
        return pd.DataFrame()

    data = pd.concat(feature_frames, ignore_index=True)
    data = data.sort_values(["date", "well"]).reset_index(drop=True)

    dates = np.sort(data["date"].unique())
    initial_size = int(len(dates) * 0.50)

    if len(dates) < 60 or len(dates) - initial_size < n_splits:
        return pd.DataFrame()

    later_dates = dates[initial_size:]
    date_folds = [
        fold for fold in np.array_split(later_dates, n_splits)
        if len(fold) > 0
    ]

    predictions = []

    for fold_number, fold_dates in enumerate(date_folds, start=1):
        train = data[data["date"] < fold_dates[0]]
        valid = data[data["date"].isin(fold_dates)]

        if len(train) < 20 or valid.empty:
            continue

        X_train = train[FEATURES]
        y_train = train["oil_production_bopd"]
        X_valid = valid[FEATURES]
        y_valid = valid["oil_production_bopd"]

        baseline_pred = valid["lag_1"].to_numpy()

        for name, model in model_bank(seed).items():
            model.fit(X_train, y_train)
            pred = np.maximum(0, model.predict(X_valid))

            predictions.append({
                "Fold": fold_number,
                "Model": name,
                "MAE": mean_absolute_error(y_valid, pred),
                "RMSE": np.sqrt(mean_squared_error(y_valid, pred)),
                "R²": (
                    r2_score(y_valid, pred)
                    if y_valid.nunique() > 1 else np.nan
                ),
                "Samples": len(valid)
            })

        predictions.append({
            "Fold": fold_number,
            "Model": "Persistence Baseline",
            "MAE": mean_absolute_error(y_valid, baseline_pred),
            "RMSE": np.sqrt(
                mean_squared_error(y_valid, baseline_pred)
            ),
            "R²": (
                r2_score(y_valid, baseline_pred)
                if y_valid.nunique() > 1 else np.nan
            ),
            "Samples": len(valid)
        })

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions)


def forecast_one_well(group, model, horizon):
    hist = (
        group.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .copy()
        .reset_index(drop=True)
    )

    required = [
        "date",
        "well",
        "field",
        "oil_production_bopd",
        "water_cut_pct",
        "pressure_psi",
        "choke_pct",
        "uptime_pct",
    ]

    if any(col not in hist.columns for col in required):
        return pd.DataFrame()

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["oil_production_bopd"] = pd.to_numeric(
        hist["oil_production_bopd"], errors="coerce"
    )

    
    hist = hist.dropna(
        subset=["date", "oil_production_bopd"]
    ).reset_index(drop=True)

    # استبعاد قراءات الإنتاج السالبة
    hist = hist[
        hist["oil_production_bopd"] >= 0
    ].copy().reset_index(drop=True)

    # التأكد من توفر تاريخ فريد وقراءات كافية
    if (
        len(hist) < 30
        or hist["date"].nunique() < 30
        or horizon < 1
    ):
        return pd.DataFrame()

    numeric_columns = [
        "water_cut_pct",
        "pressure_psi",
        "choke_pct",
        "uptime_pct",
    ]

    for col in numeric_columns:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
        hist[col] = hist[col].ffill().bfill()

        if hist[col].isna().all():
            hist[col] = 0.0

    result = []

    for step in range(int(horizon)):
        last_date = hist["date"].max()
        new_date = last_date + pd.Timedelta(days=1)

        y = hist["oil_production_bopd"].astype(float)

        if len(y) < 30:
            break

        new = {
            "date": new_date,
            "field": hist["field"].iloc[-1],
            "well": hist["well"].iloc[-1],
            "oil_production_bopd": np.nan,
            "water_cut_pct": float(hist["water_cut_pct"].iloc[-1]),
            "pressure_psi": float(hist["pressure_psi"].iloc[-1]),
            "choke_pct": float(hist["choke_pct"].iloc[-1]),
            "uptime_pct": float(hist["uptime_pct"].iloc[-1]),
        }

        feat = {
            "day_index": int(
                (new_date - hist["date"].min()).days
            ),
            "lag_1": float(y.iloc[-1]),
            "lag_7": float(y.iloc[-7]),
            "lag_14": float(y.iloc[-14]),
            "lag_30": float(y.iloc[-30]),
            "rolling_mean_7": float(y.iloc[-7:].mean()),
            "rolling_mean_30": float(y.iloc[-30:].mean()),
            "water_cut_pct": new["water_cut_pct"],
            "pressure_psi": new["pressure_psi"],
            "choke_pct": new["choke_pct"],
            "uptime_pct": new["uptime_pct"],
            "month": new_date.month,
        }

        if model is None:
            pred = max(0.0, float(y.iloc[-1]))
        else:
            feature_row = pd.DataFrame([feat])[FEATURES]
            pred = max(
                0.0,
                float(model.predict(feature_row)[0])
            )

        new["oil_production_bopd"] = pred

        result.append({
            "date": new_date,
            "field": new["field"],
            "well": new["well"],
            "forecast_oil_production_bopd": round(pred, 2),
            "scenario": "ثبات ظروف التشغيل الحالية",
        })

        hist = pd.concat(
            [hist, pd.DataFrame([new])],
            ignore_index=True
        )

    return pd.DataFrame(result)


def make_pdf_report(title, metrics, forecast):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        buff = io.BytesIO()
        doc = SimpleDocTemplate(buff, pagesize=landscape(A4), rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
        styles = getSampleStyleSheet()
        story = [Paragraph(title, styles["Title"]),
                 Paragraph("Generated by Libya OilCast AI. Forecasts are estimates, not operational instructions.", styles["BodyText"]),
                 Spacer(1, 14)]
        if not metrics.empty:
            story.append(Paragraph("Model evaluation (chronological holdout)", styles["Heading2"]))
            m = metrics.round(3)
            story.append(Table([list(m.columns)] + m.astype(str).values.tolist(), repeatRows=1))
            story.append(Spacer(1, 12))
        if not forecast.empty:
            story.append(Paragraph("Forecast preview", styles["Heading2"]))
            f = forecast.head(25).copy()
            f["date"] = f["date"].astype(str)
            story.append(Table([list(f.columns)] + f.astype(str).values.tolist(), repeatRows=1))
        for item in story:
            if isinstance(item, Table):
                item.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#155e75")),
                    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("GRID",(0,0),(-1,-1),0.35,colors.grey),
                    ("FONTSIZE",(0,0),(-1,-1),8),
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eef6f7")])
                ]))
        doc.build(story)
        buff.seek(0)
        return buff.getvalue()
    except Exception as exc:
        st.error(f"تعذر إنشاء PDF: {exc}. جرّب تثبيت reportlab من requirements.txt.")
        return None

# ---------- Sidebar ----------
st.sidebar.title("🛢️ Libya OilCast AI")
st.sidebar.caption("Oil production forecasting research prototype")

page = st.sidebar.radio(
    "التنقل",
    [
        "نظرة عامة",
        "تحليل الإنتاج",
        "تدريب ومقارنة النماذج",
        "التنبؤ المستقبلي",
        "توقع إنتاج الحقول",
        "البيانات والتصدير",
        "عن المشروع",
    ]
)

st.sidebar.divider()
st.sidebar.subheader("إعدادات البيانات الافتراضية")
n_wells = st.sidebar.slider("عدد الآبار الافتراضية", 3, 20, 8)
n_days = st.sidebar.slider("عدد أيام التاريخ", 180, 1200, 900, step=30)
seed = st.sidebar.number_input("Seed للبيانات", min_value=1, max_value=999999, value=42)
st.sidebar.caption("البيانات الافتراضية محاكاة تعليمية وليست بيانات حقيقية من ليبيا.")

synthetic = make_synthetic_data(n_wells, n_days, int(seed))

if "uploaded_data" not in st.session_state:
     st.session_state.uploaded_data = None

if "use_uploaded_data" not in st.session_state:
    st.session_state.use_uploaded_data = False

use_uploaded = st.sidebar.toggle(
    "استخدام البيانات المرفوعة",
    value=st.session_state.use_uploaded_data,
    key="uploaded_data_toggle"
)

st.session_state.use_uploaded_data = use_uploaded

if use_uploaded and st.session_state.uploaded_data is not None:
    raw = st.session_state.uploaded_data.copy()
    data_source = "Imported CSV/Excel"
else:
    raw = synthetic.copy()
    data_source = "Synthetic demo data"

if data_source == "Imported CSV/Excel":
    st.info("مصدر البيانات الحالي: ملف مرفوع من المستخدم.")
else:
    st.info("مصدر البيانات الحالي: بيانات تجريبية مولّدة للعرض والاختبار.")
    
df = prepare_well_data(raw)
fields = sorted(df["field"].dropna().unique().tolist())
selected_field = st.sidebar.selectbox("الحقل", ["كل الحقول"] + fields)
filtered = df if selected_field == "كل الحقول" else df[df["field"] == selected_field]
wells = sorted(filtered["well"].dropna().unique().tolist())
selected_well = st.sidebar.selectbox("البئر", ["كل الآبار"] + wells)
view_df = filtered if selected_well == "كل الآبار" else filtered[filtered["well"] == selected_well]
st.sidebar.caption(f"مصدر البيانات الحالي: {data_source}")

st.title("🛢️ Libya OilCast AI")
st.subheader("ليبيا أويل كاست للذكاء الاصطناعي")
st.caption(
    "منصة لتحليل إنتاج النفط ومقارنة نماذج التعلم الآلي "
    "والتنبؤ بإنتاج الحقول النفطية الليبية."
)

st.warning("تنبيه بحثي: البيانات الافتراضية مصطنعة لأغراض تطوير واختبار البرمجية فقط. لا تستخدم التوقعات لاتخاذ قرارات تشغيلية ميدانية دون التحقق من بيانات حقيقية ومراجعة مهندس مختص.")


# تجهيز متغيرات تقييم النماذج
metrics = pd.DataFrame()
test_preds = pd.DataFrame()
evaluation = {}

if page in [
    "تدريب ومقارنة النماذج",
    "التنبؤ المستقبلي",
    "توقع إنتاج الحقول",
]:
    with st.spinner("جاري تجهيز خصائص السلاسل الزمنية وتقييم النماذج..."):
        metrics, test_preds, evaluation = evaluate_models(df, int(seed))

    # إحصائيات البيانات المستخدمة
    n_wells = df[["field", "well"]].drop_duplicates().shape[0]
    n_fields = df["field"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("عدد الآبار", n_wells)
    c2.metric("عدد الحقول النفطية", n_fields)

# ---------- Pages ----------

if page == "نظرة عامة":

    st.markdown("""
    <div style="
        position: relative;
        overflow: hidden;
        padding: 32px 30px;
        border-radius: 20px;
        border: 1px solid #24516b;
        background: linear-gradient(115deg, #081525 0%, #10334c 58%, #075e66 100%);
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        margin-bottom: 18px;
    ">
        <div style="color: #65e6dc; font-size: 12px;
            font-weight: 700; letter-spacing: 2.5px; margin-bottom: 12px;">
            PETROLEUM ENGINEERING • ARTIFICIAL INTELLIGENCE
        </div>

        <div style="color: #ffffff; font-size: 36px;
            font-weight: 800; line-height: 1.25; margin-bottom: 8px;">
            Libya OilCast AI
        </div>

        <div style="color: #c7e8f5; font-size: 21px;
            font-weight: 650; margin-bottom: 16px;">
            ليبيا أويل كاست للذكاء الاصطناعي
        </div>

        <div style="color: #e2edf7; font-size: 15px;
            line-height: 1.9; max-width: 850px;">
            منصة بحثية لتحليل إنتاج النفط ومقارنة نماذج التعلم الآلي
            والتنبؤ بإنتاج الحقول النفطية الليبية.
        </div>

        <div style="margin-top: 20px; padding-top: 14px;
            border-top: 1px solid rgba(200,235,245,0.22);
            color: #a9d7e3; font-size: 12px; letter-spacing: 0.4px;">
            OIL PRODUCTION ANALYTICS &nbsp; • &nbsp;
            MACHINE LEARNING FORECASTING
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "تنبيه بحثي: البيانات الافتراضية مصطنعة لأغراض تطوير واختبار "
        "البرمجية فقط. لا تستخدم التوقعات لاتخاذ قرارات تشغيلية ميدانية "
        "دون التحقق من بيانات حقيقية ومراجعة مهندس مختص."
    )
    
    if view_df.empty:
        st.warning("لا توجد بيانات متاحة للعرض. غيّر مرشحات البيانات أو ارفع ملفًا يحتوي على قراءات إنتاج.")
    else:
        dashboard_data = view_df.copy()
        dashboard_data["date"] = pd.to_datetime(
            dashboard_data["date"], errors="coerce"
        )
        dashboard_data["oil_production_bopd"] = pd.to_numeric(
            dashboard_data["oil_production_bopd"], errors="coerce"
        )
        dashboard_data = dashboard_data.dropna(
            subset=["date", "oil_production_bopd", "well"]
        )

        if dashboard_data.empty:
            st.warning("لا توجد قراءات إنتاج صالحة بعد تنظيف البيانات.")
        else:
            total_prod = (
                dashboard_data.groupby("date")["oil_production_bopd"]
                .sum()
                .sort_index()
            )

            latest_date = dashboard_data["date"].max()
            latest_total = total_prod.get(latest_date, 0)

            well_count = dashboard_data["well"].nunique()
            field_count = (
                dashboard_data["field"].nunique()
                if "field" in dashboard_data.columns
                else 0
            )
            avg_prod = dashboard_data["oil_production_bopd"].mean()
            date_count = dashboard_data["date"].nunique()

            st.markdown("""
            <div style="font-size:20px; font-weight:700; margin:8px 0 14px 0;">
                مؤشرات الإنتاج الرئيسية
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("عدد الآبار", f"{well_count:,}")
            c2.metric("عدد الحقول", f"{field_count:,}")
            c3.metric("متوسط الإنتاج", f"{avg_prod:,.1f} BOPD")
            c4.metric("إنتاج أحدث تاريخ", f"{latest_total:,.1f} BOPD")

            st.caption(
                f"آخر تاريخ متاح: {latest_date:%Y-%m-%d} "
                f"| عدد الأيام المسجلة: {date_count:,}"
            )

            st.divider()

            left, right = st.columns([1.6, 1])

            with left:
                st.markdown("### اتجاه الإنتاج")

                trend = (
                    dashboard_data.groupby("date", as_index=False)
                    ["oil_production_bopd"].sum()
                    .sort_values("date")
                )

                fig = px.area(
                    trend,
                    x="date",
                    y="oil_production_bopd",
                    title="إجمالي الإنتاج اليومي للآبار المختارة",
                    labels={
                        "date": "التاريخ",
                        "oil_production_bopd": "الإنتاج (BOPD)"
                    }
                )

                fig.update_traces(
                    line=dict(color="#0891B2", width=2.5),
                    fillcolor="rgba(8,145,178,0.16)"
                )

                fig.update_layout(
                    height=390,
                    hovermode="x unified",
                    margin=dict(l=15, r=15, t=65, b=15),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h")
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="overview_production_trend"
                )

            with right:
                st.markdown("### مقارنة الآبار")

                by_well = (
                    dashboard_data.groupby("well", as_index=False)
                    ["oil_production_bopd"].mean()
                    .sort_values("oil_production_bopd", ascending=True)
                    .tail(12)
                )

                fig2 = px.bar(
                    by_well,
                    x="oil_production_bopd",
                    y="well",
                    orientation="h",
                    labels={
                        "well": "البئر",
                        "oil_production_bopd": "متوسط الإنتاج (BOPD)"
                    }
                )

                fig2.update_traces(
                    marker_color="#0E9F9A",
                    marker_line_width=0
                )

                fig2.update_layout(
                    height=390,
                    margin=dict(l=10, r=15, t=25, b=15),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    key="overview_well_comparison"
                )

            st.divider()

            st.markdown("### أحدث قراءات الإنتاج")

            recent = (
                dashboard_data.sort_values("date", ascending=False)
                .head(15)
                .copy()
            )

            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "تنزيل أحدث القراءات CSV",
                data=recent.to_csv(index=False).encode("utf-8-sig"),
                file_name="libya_oilcast_recent_production.csv",
                mime="text/csv",
                key="overview_recent_download"
            )

            st.caption(
                "Libya OilCast AI | التحليلات مبنية على البيانات المتاحة "
                "والآبار المحددة في المرشحات."
            )


elif page == "تحليل الإنتاج":
    st.subheader("تحليل الإنتاج والمؤشرات التشغيلية")
    a,b = st.columns(2)
    with a:
        fig = px.line(view_df, x="date", y="oil_production_bopd", color="well", title="إنتاج كل بئر")
        fig.update_layout(xaxis_title="التاريخ", yaxis_title="BOPD")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.scatter(view_df.sample(min(2500,len(view_df)), random_state=int(seed)),
                         x="water_cut_pct", y="oil_production_bopd", color="well",
                         title="علاقة Water Cut بإنتاج النفط", trendline="ols")
        fig.update_layout(xaxis_title="Water Cut (%)", yaxis_title="BOPD")
        st.plotly_chart(fig, use_container_width=True)
    c,d = st.columns(2)
    with c:
        fig = px.line(view_df, x="date", y="pressure_psi", color="well", title="ضغط البئر (بيانات افتراضية/مرفوعة)")
        fig.update_layout(xaxis_title="التاريخ", yaxis_title="Pressure (psi)")
        st.plotly_chart(fig, use_container_width=True)
    with d:
        fig = px.line(view_df, x="date", y="water_cut_pct", color="well", title="تطور نسبة الماء")
        fig.update_layout(xaxis_title="التاريخ", yaxis_title="Water Cut (%)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("الارتباطات المرئية لا تثبت السببية. تأكد من وحدات القياس وجودة البيانات قبل تفسيرها هندسياً.")


elif page == "تدريب ومقارنة النماذج":
    st.subheader("تقييم النماذج على فترة اختبار زمنية")

    if metrics.empty or not evaluation:
        st.error(
            "البيانات غير كافية للتقييم. "
            "تأكد من وجود قراءات يومية كافية لكل بئر."
        )
    else:
        st.markdown("### نتائج التحقق واختيار النموذج")
        st.dataframe(
            metrics.round(3),
            use_container_width=True,
            hide_index=True
        )

        best_name = evaluation["best_name"]
        test_metrics = evaluation.get("test_metrics", {})

        if best_name == "Persistence Baseline":
            st.info(
                "خط الأساس Persistence Baseline حقق أفضل نتيجة "
                "على بيانات التحقق، ولم يتفوق عليه أي نموذج تعلم آلي."
            )
        else:
            st.success(
                f"النموذج المختار بناءً على بيانات التحقق: {best_name}"
            )

        st.markdown("### أداء النموذج على الاختبار النهائي")

        if not test_preds.empty:
            st.metric(
                "عدد قراءات الاختبار النهائي",
                f"{len(test_preds):,}"
            )
       
        if test_metrics:
            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "النموذج المختار",
                best_name
            )

            c2.metric(
                "RMSE",
                f"{test_metrics['RMSE']:.2f} BOPD"
            )

            c3.metric(
                "MAE",
                f"{test_metrics['MAE']:.2f} BOPD"
            )

            r2_value = test_metrics.get("R²", np.nan)

            c4.metric(
                "R²",
                f"{r2_value:.3f}"
                if pd.notna(r2_value)
                else "غير متاح"
            )

            st.caption(
                "RMSE وMAE يقيسان حجم أخطاء التنبؤ، والأقل أفضل. "
                "R² يقيس مدى تفسير النموذج لتباين الإنتاج؛ "
                "وقد تكون قيمته سالبة إذا كان الأداء ضعيفاً."
            )
        else:
            st.warning("نتائج الاختبار النهائي غير متاحة.")


        fig = px.bar(
            metrics,
            x="Model",
            y="RMSE",
            title="مقارنة RMSE على بيانات التحقق — الأقل أفضل"
        )
        fig.update_layout(
            xaxis_title="النموذج",
            yaxis_title="RMSE (BOPD)"
        )
        st.plotly_chart(fig, use_container_width=True)

        
        if not test_preds.empty:
            available_pairs = (
                test_preds[["field", "well"]]
                .drop_duplicates()
                .sort_values(["field", "well"])
            )

            available_pairs["label"] = (
                available_pairs["field"].astype(str)
                + " — "
                + available_pairs["well"].astype(str)
            )

            selected_pair = st.selectbox(
                "اختر الحقل والبئر لعرض الاختبار النهائي",
                available_pairs["label"].tolist()
            )

            chosen = available_pairs[
                available_pairs["label"] == selected_pair
            ].iloc[0]

            p = test_preds[
                (test_preds["field"] == chosen["field"])
                & (test_preds["well"] == chosen["well"])
            ].sort_values("date")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=p["date"],
                    y=p["oil_production_bopd"],
                    mode="lines",
                    name="الإنتاج الفعلي"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=p["date"],
                    y=p["predicted_oil_production_bopd"],
                    mode="lines",
                    name="الإنتاج المتوقع"
                )
            )

            fig.update_layout(
                title=f"الإنتاج الفعلي مقابل المتوقع — {chosen['field']} — {chosen['well']}",
                xaxis_title="التاريخ",
                yaxis_title="الإنتاج (BOPD)",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "يُختار النموذج باستخدام فترة التحقق، ثم يُقيّم مرة واحدة "
            "على فترة اختبار نهائية لم تُستخدم في الاختيار. "
            "النتائج الحالية تعتمد على بيانات اصطناعية، ولا تثبت "
            "دقة النموذج على حقول نفطية حقيقية."
        )

        st.download_button(
            "تنزيل جدول المقاييس CSV",
            metrics.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_metrics.csv",
            mime="text/csv"
        )
        
        st.markdown("### الاختبار الزمني المتكرر (Walk-Forward Validation)")

        with st.spinner("جاري اختبار النماذج عبر 3 فترات زمنية..."):
            wf_results = walk_forward_validation(
                df,
                seed=int(seed),
                n_splits=3
            )

        if wf_results.empty:
            st.warning(
                "تعذر إجراء الاختبار الزمني المتكرر. "
                "تأكد من توفر بيانات كافية."
            )
        else:
            st.markdown("#### نتائج كل فترة زمنية")

            st.dataframe(
                wf_results.round(3),
                use_container_width=True,
                hide_index=True
            )

            wf_summary = (
                wf_results.groupby("Model", as_index=False)
                .agg(
                    Mean_MAE=("MAE", "mean"),
                    Mean_RMSE=("RMSE", "mean"),
                    Mean_R2=("R²", "mean")
                )
                .sort_values("Mean_RMSE")
                .reset_index(drop=True)
            )

            st.markdown("#### متوسط أداء النماذج عبر الفترات الثلاث")

            st.dataframe(
                wf_summary.round(3),
                use_container_width=True,
                hide_index=True
            )

            fig_wf = px.bar(
                wf_summary,
                x="Model",
                y="Mean_RMSE",
                title="متوسط RMSE عبر الفترات الزمنية — الأقل أفضل"
            )

            fig_wf.update_layout(
                xaxis_title="النموذج",
                yaxis_title="متوسط RMSE (BOPD)"
            )

            st.plotly_chart(fig_wf, use_container_width=True)



elif page == "التنبؤ المستقبلي":
    st.subheader("تنبؤ إنتاج البئر للأيام القادمة")

    st.warning(
        "تنبيه هندسي: نتائج التنبؤ تقديرات مبنية على بيانات الإنتاج التاريخية "
        "وافتراض استمرار نمط البيانات وظروف التشغيل. "
        "لا تعتبر النتائج ضمانًا للإنتاج الفعلي أو بديلًا عن التقييم الهندسي."
    )

    if not evaluation or not evaluation.get("models"):
        st.error("تعذر تدريب النماذج. تأكد أن لكل بئر بيانات كافية.")

    else:
        candidate_models = list(evaluation["models"].keys())

        default_model = (
            evaluation["best_name"]
            if evaluation["best_name"] in candidate_models
            else candidate_models[0]
        )

        available_fields = sorted(
            df["field"].dropna().astype(str).unique().tolist()
        )

        if not available_fields:
            st.warning("لا توجد حقول متاحة للتنبؤ.")

        else:
            chosen_field = st.selectbox(
                "اختر الحقل",
                available_fields,
                key="forecast_field"
            )

            field_data = df[
                df["field"].astype(str) == chosen_field
            ].copy()

            available_wells = sorted(
                field_data["well"].dropna().astype(str).unique().tolist()
            )

            if not available_wells:
                st.warning("لا توجد آبار متاحة في هذا الحقل.")

            else:
                chosen_well = st.selectbox(
                    "اختر البئر",
                    available_wells,
                    key="forecast_well"
                )

                history = field_data[
                    field_data["well"].astype(str) == chosen_well
                ].copy()

                history = history.sort_values("date").drop_duplicates(
                    subset=["date"], keep="last"
                )

                horizon = st.slider(
                    "أفق التنبؤ (يوم)",
                    7, 90, 30,
                    key="forecast_horizon"
                )

                chosen_model = st.selectbox(
                    "نموذج التنبؤ",
                    candidate_models,
                    index=candidate_models.index(default_model),
                    key="forecast_model"
                )

                run_forecast = st.button(
                    "حساب توقع إنتاج البئر",
                    key="run_single_well_forecast"
                )

                if run_forecast:
                    if len(history) < 30:
                        st.error(
                            "هذا البئر يحتاج إلى 30 قراءة يومية على الأقل "
                            "لتكوين خصائص التأخير."
                        )

                    else:
                        forecast = forecast_one_well(
                            history,
                            evaluation["models"][chosen_model],
                            horizon
                        )

                        if forecast.empty:
                            st.error(
                                "تعذر إنشاء التنبؤ. راجع التواريخ والقراءات "
                                "والأعمدة التشغيلية للبئر."
                            )

                        else:
                            fig = go.Figure()

                            recent = history.tail(120)

                            fig.add_trace(go.Scatter(
                                x=recent["date"],
                                y=recent["oil_production_bopd"],
                                mode="lines",
                                name="التاريخ الفعلي"
                            ))

                            fig.add_trace(go.Scatter(
                                x=forecast["date"],
                                y=forecast["forecast_oil_production_bopd"],
                                mode="lines+markers",
                                name="الإنتاج المتوقع"
                            ))

                            fig.update_layout(
                                title=(
                                    f"توقع الإنتاج — {chosen_field} / "
                                    f"{chosen_well} باستخدام {chosen_model}"
                                ),
                                xaxis_title="التاريخ",
                                yaxis_title="Oil Production (BOPD)",
                                hovermode="x unified"
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key="single_well_forecast_chart"
                            )

                            total = forecast[
                                "forecast_oil_production_bopd"
                            ].sum()

                            avg = forecast[
                                "forecast_oil_production_bopd"
                            ].mean()

                            m1, m2, m3 = st.columns(3)

                            m1.metric(
                                "متوسط الإنتاج المتوقع",
                                f"{avg:,.1f} BOPD"
                            )

                            m2.metric(
                                "مجموع الإنتاج المتوقع",
                                f"{total:,.1f} برميل"
                            )

                            m3.metric(
                                "فترة التنبؤ",
                                f"{len(forecast)} يوم"
                            )

                            st.dataframe(
                                forecast,
                                use_container_width=True,
                                hide_index=True
                            )

                            safe_field = "".join(
                                c if c.isalnum() else "_"
                                for c in chosen_field.lower()
                            )

                            safe_well = "".join(
                                c if c.isalnum() else "_"
                                for c in chosen_well.lower()
                            )

                            st.download_button(
                                "تنزيل التنبؤ CSV",
                                forecast.to_csv(
                                    index=False
                                ).encode("utf-8-sig"),
                                file_name=(
                                    f"forecast_{safe_field}_{safe_well}.csv"
                                ),
                                mime="text/csv",
                                key="single_well_forecast_download"
                            )

                            st.caption(
                                "التنبؤ تكراري؛ وقد يتراكم عدم اليقين مع زيادة "
                                "المدة. ظروف التشغيل المستقبلية مفترضة ثابتة "
                                "عند آخر قراءة متاحة."
                            )


elif page == "توقع إنتاج الحقول":
    st.subheader("التنبؤ بإنتاج الحقل بالكامل")

    st.warning(
        "التوقعات تقديرية وتعتمد على البيانات التاريخية "
        "وافتراض ثبات ظروف التشغيل. لا تمثل ضمانًا للإنتاج الفعلي."
    )

    if not evaluation or not evaluation.get("models"):
        st.error("تعذر تدريب النماذج. راجع البيانات أولًا.")
    else:
        available_fields = sorted(
            df["field"].dropna().astype(str).unique().tolist()
        )

        if not available_fields:
            st.warning("لا توجد حقول متاحة.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                field_choice = st.selectbox(
                    "اختر الحقل",
                    available_fields,
                    key="field_forecast_field"
                )

            with col2:
                horizon_field = st.slider(
                    "مدة التنبؤ بالأيام",
                    min_value=7,
                    max_value=90,
                    value=30,
                    key="field_forecast_horizon"
                )

            candidate_models = list(evaluation["models"].keys())
            default_model = evaluation["best_name"]

            if default_model not in candidate_models:
                default_model = candidate_models[0]

            chosen_field_model = st.selectbox(
                "نموذج التنبؤ",
                candidate_models,
                index=candidate_models.index(default_model),
                key="field_forecast_model"
            )

            if st.button(
                "احسب إنتاج الحقل المتوقع",
                key="run_field_forecast"
            ):
                field_data = df[
                    df["field"].astype(str) == field_choice
                ].copy()

                all_forecasts = []
                skipped_wells = []

                model = evaluation["models"][chosen_field_model]

                with st.spinner("جاري حساب توقعات آبار الحقل..."):
                    for well_name, group in field_data.groupby("well"):
                        group = group.sort_values("date").copy()

                        if (
                            group["date"].nunique() < 30
                            or group["oil_production_bopd"]
                            .notna().sum() < 30
                        ):
                            skipped_wells.append(str(well_name))
                            continue

                        forecast_part = forecast_one_well(
                            group,
                            model,
                            int(horizon_field)
                        )

                        if forecast_part.empty:
                            skipped_wells.append(str(well_name))
                            continue

                        all_forecasts.append(forecast_part)

                if not all_forecasts:
                    st.error(
                        "لم نتمكن من توقع إنتاج أي بئر. "
                        "تأكد من توفر 30 قراءة صالحة على الأقل لكل بئر."
                    )
                else:
                    well_forecasts = pd.concat(
                        all_forecasts,
                        ignore_index=True
                    )

                    daily_total = (
                        well_forecasts
                        .groupby("date", as_index=False)
                        ["forecast_oil_production_bopd"]
                        .sum()
                        .rename(columns={
                            "forecast_oil_production_bopd":
                            "total_forecast_bopd"
                        })
                        .sort_values("date")
                    )

                    average_daily = daily_total[
                        "total_forecast_bopd"
                    ].mean()

                    total_period = daily_total[
                        "total_forecast_bopd"
                    ].sum()

                    st.success(
                        f"تم التنبؤ لـ "
                        f"{well_forecasts['well'].nunique()} بئر."
                    )

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "متوسط إنتاج الحقل المتوقع",
                        f"{average_daily:,.1f} BOPD"
                    )

                    m2.metric(
                        "إجمالي إنتاج الفترة",
                        f"{total_period:,.1f} برميل"
                    )

                    m3.metric(
                        "الآبار المتوقّع لها",
                        f"{well_forecasts['well'].nunique()}"
                    )

                    fig_total = px.line(
                        daily_total,
                        x="date",
                        y="total_forecast_bopd",
                        markers=True,
                        title="إجمالي إنتاج الحقل المتوقع يوميًا"
                    )

                    fig_total.update_layout(
                        xaxis_title="التاريخ",
                        yaxis_title="إنتاج الحقل (BOPD)",
                        hovermode="x unified"
                    )

                    st.plotly_chart(
                        fig_total,
                        use_container_width=True
                    )

                    fig_wells = px.line(
                        well_forecasts,
                        x="date",
                        y="forecast_oil_production_bopd",
                        color="well",
                        title="مقارنة الإنتاج المتوقع بين آبار الحقل"
                    )

                    fig_wells.update_layout(
                        xaxis_title="التاريخ",
                        yaxis_title="إنتاج البئر (BOPD)",
                        hovermode="x unified"
                    )

                    st.plotly_chart(
                        fig_wells,
                        use_container_width=True
                    )

                    st.markdown("### إجمالي إنتاج الحقل حسب اليوم")

                    st.dataframe(
                        daily_total.round(2),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("### توقعات كل بئر")

                    st.dataframe(
                        well_forecasts.round(2),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "تنزيل توقعات كل الآبار CSV",
                        well_forecasts.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="field_well_forecasts.csv",
                        mime="text/csv",
                        key="download_field_wells"
                    )

                    st.download_button(
                        "تنزيل إجمالي إنتاج الحقل CSV",
                        daily_total.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="field_daily_forecast.csv",
                        mime="text/csv",
                        key="download_field_total"
                    )

                    if skipped_wells:
                        st.warning(
                            "تم تجاوز الآبار التالية لعدم كفاية "
                            "البيانات أو تعذر إنشاء التنبؤ: "
                            + "، ".join(skipped_wells)
                        )

                    st.caption(
                        "إجمالي إنتاج الحقل هو مجموع توقعات الآبار "
                        "التي نجح التنبؤ لها فقط. افتراض ثبات ظروف "
                        "التشغيل قد لا يعكس الظروف المستقبلية الفعلية."
                    )



elif page == "البيانات والتصدير":
    st.subheader("رفع البيانات ومعاينتها وفحص جودتها")

    st.markdown("""
    **الأعمدة المطلوبة:**
    - `date`: تاريخ القراءة.
    - `oil_production_bopd`: إنتاج النفط بوحدة برميل/يوم.

    **الأعمدة الاختيارية:**
    - `well`: اسم أو رقم البئر.
    - `field`: اسم الحقل.
    - `water_cut_pct`, `pressure_psi`, `choke_pct`, `uptime_pct`.
    """)

    upload = st.file_uploader(
        "ارفع ملف البيانات",
        type=["csv", "xlsx", "xls"],
        help="يمكنك رفع ملف CSV أو Excel."
    )

    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                candidate = pd.read_csv(upload)
            else:
                candidate = pd.read_excel(upload)

            candidate.columns = (
                candidate.columns.astype(str).str.strip()
            )

            st.markdown("### 1. معاينة البيانات")
            st.caption(
                f"عدد الصفوف: {len(candidate):,} | "
                f"عدد الأعمدة: {len(candidate.columns)}"
            )

            st.dataframe(
                candidate.head(20),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 2. تحديد الأعمدة")

            detected = detect_oil_columns(candidate)
            date_options = ["— اختر العمود —"] + list(candidate.columns)
            production_options = ["— اختر العمود —"] + list(candidate.columns)

            date_default = (
                date_options.index(detected["date"])
                if detected.get("date") in candidate.columns else 0
            )
            production_default = (
                production_options.index(detected["oil_production_bopd"])
                if detected.get("oil_production_bopd") in candidate.columns else 0
            )

            date_col = st.selectbox(
                "أي عمود يمثل التاريخ؟",
                date_options,
                index=date_default,
                key="detected_date_column"
            )

            production_col = st.selectbox(
                "أي عمود يمثل إنتاج النفط؟",
                production_options,
                index=production_default,
                key="detected_production_column"
            )

            well_col = st.selectbox(
                "عمود اسم البئر (اختياري)",
                ["بدون"] + list(candidate.columns),
                index=(
                    1 + list(candidate.columns).index(detected["well"])
                    if detected.get("well") in candidate.columns else 0
                ),
                key="detected_well_column"
            )

            field_col = st.selectbox(
                "عمود اسم الحقل (اختياري)",
                ["بدون"] + list(candidate.columns),
                index=(
                    1 + list(candidate.columns).index(detected["field"])
                    if detected.get("field") in candidate.columns else 0
                ),
                key="detected_field_column"
            )

            if date_col != "— اختر العمود —" and production_col != "— اختر العمود —":
                st.success("تم تحديد عمود التاريخ وعمود الإنتاج.")

                st.markdown("### 3. فحص جودة البيانات")

                quality = pd.DataFrame({
                    "العمود": candidate.columns,
                    "نوع البيانات": [
                        str(candidate[c].dtype) for c in candidate.columns
                    ],
                    "القيم المفقودة": [
                        int(candidate[c].isna().sum()) for c in candidate.columns
                    ],
                    "القيم الفريدة": [
                        int(candidate[c].nunique()) for c in candidate.columns
                    ]
                })

                st.dataframe(
                    quality,
                    use_container_width=True,
                    hide_index=True
                )

                parsed_dates = pd.to_datetime(
                    candidate[date_col], errors="coerce"
                )
                production = pd.to_numeric(
                    candidate[production_col], errors="coerce"
                )

                invalid_dates = int(parsed_dates.isna().sum())
                invalid_production = int(production.isna().sum())
                negative_production = int((production < 0).sum())
                duplicate_rows = int(candidate.duplicated().sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("تواريخ غير صالحة", invalid_dates)
                c2.metric("إنتاج غير رقمي أو مفقود", invalid_production)
                c3.metric("إنتاج سالب", negative_production)
                c4.metric("صفوف مكررة", duplicate_rows)

                if invalid_dates or invalid_production or negative_production:
                    st.warning(
                        "توجد قيم غير صالحة. سيتم استبعاد القراءات "
                        "ذات التاريخ أو الإنتاج غير الصالح، والإنتاج السالب."
                    )
                else:
                    st.success("اجتازت البيانات الفحوص الأساسية.")

                if st.button(
                    "اعتماد البيانات واستخدامها في المشروع",
                    type="primary"
                ):
                    cleaned = candidate.copy()

                    cleaned["date"] = parsed_dates
                    cleaned["oil_production_bopd"] = production

                    if well_col != "بدون":
                        cleaned["well"] = cleaned[well_col].astype(str)
                    elif "well" not in cleaned.columns:
                        cleaned["well"] = "Well-01"

                    if field_col != "بدون":
                        cleaned["field"] = cleaned[field_col].astype(str)
                    elif "field" not in cleaned.columns:
                        cleaned["field"] = "Imported Field"

                    cleaned = cleaned.dropna(
                        subset=["date", "oil_production_bopd"]
                    )
                    cleaned = cleaned[
                        cleaned["oil_production_bopd"] >= 0
                    ]

                    if cleaned.empty:
                        st.error("لا توجد قراءات صالحة بعد التنظيف.")
                    else:
                        st.session_state.uploaded_data = cleaned
                        st.session_state.use_uploaded_data = True
                        st.success("تم اعتماد البيانات بنجاح.")
                        st.rerun()

            else:
                st.info("اختر عمود التاريخ وعمود إنتاج النفط لعرض فحص الجودة.")
        except Exception as exc:
            st.error(
                f"تعذر قراءة الملف. تأكد من سلامته وتنسيقه: {exc}"
            )

    st.divider()
    st.markdown("### تنزيل البيانات الافتراضية")

    st.download_button(
        "تنزيل البيانات الافتراضية CSV",
        synthetic.to_csv(index=False).encode("utf-8-sig"),
        file_name="synthetic_oilfield_data.csv",
        mime="text/csv"
    )

    st.markdown("### تصدير تقرير PDF")

    if st.button("إنشاء تقرير PDF"):
        pdf = make_pdf_report(
            "Libya OilCast AI — Oil Production Forecast Report",
            metrics,
            pd.DataFrame()
        )

        if pdf:
            st.download_button(
                "تنزيل التقرير PDF",
                pdf,
                file_name="libya_oilcast_report.pdf",
                mime="application/pdf"
            )

    st.markdown("### البيانات المستخدمة حالياً")
    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

elif page == "عن المشروع":
    st.subheader("عن Libya OilCast AI")
    st.markdown("""
**الهدف:** تطوير نموذج برمجي قابل للتوسعة للتنبؤ بإنتاج النفط باستخدام أساليب تعلم الآلة، مع تقييم زمني ومقارنة النماذج.

**المكوّنات الحالية**
- توليد بيانات اصطناعية متعددة الآبار لأغراض اختبار الواجهة.
- هندسة خصائص زمنية (lags وrolling means).
- مقارنة Linear Regression وRandom Forest وExtra Trees وHist Gradient Boosting مع Persistence Baseline.
- تقسيم زمني للتدريب والاختبار.
- عرض المقاييس والرسوم وتصدير CSV/PDF.

**قبل استخدامه في مشروع بحثي نهائي**
1. استبدال البيانات الاصطناعية ببيانات موثقة وشرح مصدرها ووحداتها.
2. مراجعة طريقة التعامل مع الآبار المتوقفة والقراءات المفقودة والتكرارات.
3. مقارنة النماذج على نفس فترات الاختبار، وإضافة walk-forward validation.
4. إضافة نطاقات عدم اليقين وتحليل حساسية وشرح هندسي للخصائص.
5. توثيق الإصدارات والنتائج وإعادة التجارب ببذرة ثابتة.

**ملاحظة:** هذه نسخة بحثية أولية وليست نظاماً تشغيلياً معتمداً. لا توجد أي دعوى بأن النموذج جديد أو حاصل على براءة اختراع؛ الجِدّة تحتاج مراجعة أدبيات وتجارب على بيانات موثوقة.
""")
    st.code("Python 3.10+  |  Streamlit  |  scikit-learn  |  pandas  |  Plotly", language="text")

st.sidebar.divider()
st.sidebar.caption("Research prototype • Verify imported data before interpreting results.")
