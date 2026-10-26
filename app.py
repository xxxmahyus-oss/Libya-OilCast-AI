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
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.25rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #155e75 65%, #0f766e 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {margin: 0; color: #d9f3f4;}
    .small-note {font-size: .85rem; color: #64748b;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px; border-radius: 12px;}
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
    for well, group in df.groupby("well"):
        f = add_features(group)
        feature_frames.append(f)
    if not feature_frames:
        return pd.DataFrame(), pd.DataFrame(), {}
    data = pd.concat(feature_frames, ignore_index=True)
    if len(data) < 35:
        return pd.DataFrame(), data, {}
    data = data.sort_values("date")
    split_date = data["date"].quantile(0.8)
    train = data[data["date"] <= split_date]
    test = data[data["date"] > split_date]
    if len(train) < 20 or len(test) < 5:
        cut = int(len(data)*0.8)
        train, test = data.iloc[:cut], data.iloc[cut:]
    X_train, y_train = train[FEATURES], train["oil_production_bopd"]
    X_test, y_test = test[FEATURES], test["oil_production_bopd"]
    rows, fitted = [], {}
    # Persistence baseline: next day's production equals the most recent known day.
    baseline = test["lag_1"].to_numpy()
    rows.append({"Model":"Persistence Baseline", "MAE":mean_absolute_error(y_test, baseline),
                 "RMSE":np.sqrt(mean_squared_error(y_test, baseline)),
                 "R²":r2_score(y_test, baseline) if y_test.nunique()>1 else np.nan})
    for name, model in model_bank(seed).items():
        model.fit(X_train, y_train)
        pred = np.maximum(0, model.predict(X_test))
        rows.append({"Model":name, "MAE":mean_absolute_error(y_test, pred),
                     "RMSE":np.sqrt(mean_squared_error(y_test, pred)),
                     "R²":r2_score(y_test, pred) if y_test.nunique()>1 else np.nan})
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    if best_name == "Persistence Baseline":
        best_name = min([x for x in fitted], key=lambda x: metrics.loc[metrics["Model"]==x, "RMSE"].iloc[0])
    best_model = fitted[best_name]
    test_predictions = test[["date","field","well","oil_production_bopd"]].copy()
    test_predictions["predicted_oil_production_bopd"] = np.maximum(0, best_model.predict(test[FEATURES]))
    test_predictions["model"] = best_name
    return metrics, test_predictions, {"models":fitted, "best_name":best_name, "features":FEATURES, "feature_data":data}

def forecast_one_well(group, model, horizon):
    hist = group.sort_values("date").copy().reset_index(drop=True)
    if len(hist) < 31:
        return pd.DataFrame()
    result = []
    for step in range(horizon):
        last_date = hist["date"].max()
        new_date = last_date + pd.Timedelta(days=1)
        y = hist["oil_production_bopd"].astype(float)
        new = {
            "date":new_date,
            "field":hist["field"].iloc[-1],
            "well":hist["well"].iloc[-1],
            "oil_production_bopd":np.nan,
            "water_cut_pct":float(hist["water_cut_pct"].iloc[-1]),
            "pressure_psi":float(hist["pressure_psi"].iloc[-1]),
            "choke_pct":float(hist["choke_pct"].iloc[-1]),
            "uptime_pct":float(hist["uptime_pct"].iloc[-1]),
        }
        # Feature vector uses only information available up to the forecast origin.
        feat = {
            "day_index":int((new_date-hist["date"].min()).days),
            "lag_1":float(y.iloc[-1]), "lag_7":float(y.iloc[-7]),
            "lag_14":float(y.iloc[-14]), "lag_30":float(y.iloc[-30]),
            "rolling_mean_7":float(y.iloc[-7:].mean()),
            "rolling_mean_30":float(y.iloc[-30:].mean()),
            "water_cut_pct":new["water_cut_pct"], "pressure_psi":new["pressure_psi"],
            "choke_pct":new["choke_pct"], "uptime_pct":new["uptime_pct"],
            "month":new_date.month
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([feat])[FEATURES])[0]))
        new["oil_production_bopd"] = pred
        result.append({"date":new_date,"field":new["field"],"well":new["well"],
                       "forecast_oil_production_bopd":round(pred,2)})
        hist = pd.concat([hist, pd.DataFrame([new])], ignore_index=True)
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
page = st.sidebar.radio("التنقل", ["نظرة عامة", "تحليل الإنتاج", "تدريب ومقارنة النماذج", "التنبؤ المستقبلي", "البيانات والتصدير", "عن المشروع"])
st.sidebar.divider()
st.sidebar.subheader("إعدادات البيانات الافتراضية")
n_wells = st.sidebar.slider("عدد الآبار الافتراضية", 3, 20, 8)
n_days = st.sidebar.slider("عدد أيام التاريخ", 180, 1200, 900, step=30)
seed = st.sidebar.number_input("Seed للبيانات", min_value=1, max_value=999999, value=42)
st.sidebar.caption("البيانات الافتراضية محاكاة تعليمية وليست بيانات حقيقية من ليبيا.")

synthetic = make_synthetic_data(n_wells, n_days, int(seed))
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
use_uploaded = st.sidebar.toggle("استخدام بيانات CSV المرفوعة", value=False)
if use_uploaded and st.session_state.uploaded_data is not None:
    raw = st.session_state.uploaded_data.copy()
    data_source = "Imported CSV"
else:
    raw = synthetic.copy()
    data_source = "Synthetic demo data"

df = prepare_well_data(raw)
fields = sorted(df["field"].dropna().unique().tolist())
selected_field = st.sidebar.selectbox("الحقل", ["كل الحقول"] + fields)
filtered = df if selected_field == "كل الحقول" else df[df["field"] == selected_field]
wells = sorted(filtered["well"].dropna().unique().tolist())
selected_well = st.sidebar.selectbox("البئر", ["كل الآبار"] + wells)
view_df = filtered if selected_well == "كل الآبار" else filtered[filtered["well"] == selected_well]
st.sidebar.caption(f"مصدر البيانات الحالي: {data_source}")

st.markdown(f"""
<div class="hero">
  <h1>🛢️ {APP_NAME}</h1>
  <p>Machine Learning-Based Forecasting of Oil Production in Libyan Oilfields</p>
  <p>منصة بحثية لتصور بيانات الإنتاج ومقارنة نماذج التنبؤ.</p>
</div>
""", unsafe_allow_html=True)

st.warning("تنبيه بحثي: البيانات الافتراضية مصطنعة لأغراض تطوير واختبار البرمجية فقط. لا تستخدم التوقعات لاتخاذ قرارات تشغيلية ميدانية دون التحقق من بيانات حقيقية ومراجعة مهندس مختص.")

# Cache evaluation for current dataset
with st.spinner("جاري تجهيز خصائص السلاسل الزمنية وتقييم النماذج..."):
    metrics, test_preds, evaluation = evaluate_models(df, int(seed))

# ---------- Pages ----------
if page == "نظرة عامة":
    total_prod = view_df.groupby("date")["oil_production_bopd"].sum()
    latest = view_df.sort_values("date").groupby("well").tail(1)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("عدد الآبار", f"{view_df['well'].nunique():,}")
    c2.metric("متوسط الإنتاج اليومي", f"{view_df['oil_production_bopd'].mean():,.1f} BOPD")
    c3.metric("إجمالي آخر يوم", f"{total_prod.iloc[-1]:,.1f} BOPD" if len(total_prod) else "—")
    c4.metric("أيام البيانات", f"{view_df['date'].nunique():,}")
    left,right = st.columns([1.6,1])
    with left:
        trend = view_df.groupby("date", as_index=False)["oil_production_bopd"].sum()
        fig = px.line(trend, x="date", y="oil_production_bopd", title="اتجاه الإنتاج اليومي — الآبار المختارة")
        fig.update_layout(xaxis_title="التاريخ", yaxis_title="إنتاج النفط (برميل/يوم)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_well = view_df.groupby("well", as_index=False)["oil_production_bopd"].mean().sort_values("oil_production_bopd", ascending=False)
        fig2 = px.bar(by_well, x="well", y="oil_production_bopd", title="متوسط الإنتاج حسب البئر")
        fig2.update_layout(xaxis_title="البئر", yaxis_title="متوسط BOPD")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("آخر القراءات")
    st.dataframe(view_df.sort_values("date", ascending=False).head(15), use_container_width=True, hide_index=True)

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
    if metrics.empty:
        st.error("البيانات غير كافية. نحتاج عادةً 60 قراءة يومية على الأقل لكل بئر، ويفضل أكثر.")
    else:
        st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
        best = metrics.iloc[0]
        c1,c2,c3 = st.columns(3)
        c1.metric("أفضل نموذج حسب RMSE", str(best["Model"]))
        c2.metric("RMSE", f"{best['RMSE']:.2f}")
        c3.metric("MAE", f"{best['MAE']:.2f}")
        fig = px.bar(metrics, x="Model", y="RMSE", title="مقارنة RMSE — الأقل أفضل")
        fig.update_layout(xaxis_title="النموذج", yaxis_title="RMSE (BOPD)")
        st.plotly_chart(fig, use_container_width=True)
        if not test_preds.empty:
            available_wells = sorted(test_preds["well"].unique())
            pred_well = st.selectbox("اختر بئراً لعرض الاختبار", available_wells)
            p = test_preds[test_preds["well"]==pred_well].sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p["date"], y=p["oil_production_bopd"], mode="lines", name="الإنتاج الفعلي"))
            fig.add_trace(go.Scatter(x=p["date"], y=p["predicted_oil_production_bopd"], mode="lines", name="توقع النموذج"))
            fig.update_layout(title=f"فعلي مقابل متوقع — {pred_well}", xaxis_title="التاريخ", yaxis_title="BOPD", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        st.info("يُقسَّم التاريخ ترتيبياً إلى تدريب واختبار؛ لا تُستخدم عشوائية الصفوف. مع ذلك، نتائج البيانات الاصطناعية لا تمثل دقة متوقعة على حقول حقيقية.")
        st.download_button("تنزيل جدول المقاييس CSV", metrics.to_csv(index=False).encode("utf-8-sig"),
                           file_name="model_metrics.csv", mime="text/csv")

elif page == "التنبؤ المستقبلي":
    st.subheader("تنبؤ إنتاج البئر للأيام القادمة")
    if not evaluation or not evaluation.get("models"):
        st.error("تعذر تدريب النماذج. تأكد أن لكل بئر بيانات كافية.")
    else:
        candidate_models = list(evaluation["models"].keys())
        default_model = evaluation["best_name"] if evaluation["best_name"] in candidate_models else candidate_models[0]
        col1,col2 = st.columns(2)
        with col1:
            chosen_well = st.selectbox("البئر المراد التنبؤ له", wells if wells else sorted(df["well"].unique()))
        with col2:
            horizon = st.slider("أفق التنبؤ (يوم)", 7, 90, 30)
        chosen_model = st.selectbox("نموذج التنبؤ", candidate_models, index=candidate_models.index(default_model))
        history = df[df["well"]==chosen_well].sort_values("date")
        if len(history) < 31:
            st.error("هذا البئر يحتاج إلى 31 قراءة يومية على الأقل لتكوين خصائص التأخير.")
        else:
            forecast = forecast_one_well(history, evaluation["models"][chosen_model], horizon)
            fig = go.Figure()
            recent = history.tail(120)
            fig.add_trace(go.Scatter(x=recent["date"], y=recent["oil_production_bopd"], mode="lines", name="التاريخ الفعلي"))
            fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast_oil_production_bopd"], mode="lines+markers", name="التوقع"))
            fig.update_layout(title=f"توقع الإنتاج — {chosen_well} باستخدام {chosen_model}",
                              xaxis_title="التاريخ", yaxis_title="Oil Production (BOPD)", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            total = forecast["forecast_oil_production_bopd"].sum()
            avg = forecast["forecast_oil_production_bopd"].mean()
            m1,m2,m3 = st.columns(3)
            m1.metric("متوسط الإنتاج المتوقع", f"{avg:,.1f} BOPD")
            m2.metric("مجموع القيم اليومية المتوقعة", f"{total:,.1f} برميل")
            m3.metric("فترة التنبؤ", f"{horizon} يوم")
            st.dataframe(forecast, use_container_width=True, hide_index=True)
            st.download_button("تنزيل التنبؤ CSV", forecast.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"forecast_{chosen_well.lower()}.csv", mime="text/csv")
            st.caption("هذا توقع أحادي الخطوة تكراري؛ عدم اليقين يتراكم كلما طال أفق التنبؤ، والخصائص التشغيلية المستقبلية تُثبَّت عند آخر قراءة.")

elif page == "البيانات والتصدير":
    st.subheader("رفع بيانات حقيقية أو تنزيل بيانات العرض")
    st.markdown("""
يُفضّل أن يحتوي CSV على الأعمدة التالية:
- `date`: التاريخ بصيغة مثل `2025-01-31`
- `oil_production_bopd`: إنتاج النفط بوحدة برميل/يوم
- `well`: معرّف البئر
- `field`: اسم الحقل (اختياري)
- `water_cut_pct`, `pressure_psi`, `choke_pct`, `uptime_pct` (اختيارية؛ تُملأ بقيم افتراضية إن غابت)
""")
    upload = st.file_uploader("ارفع ملف CSV", type=["csv"])
    if upload is not None:
        try:
            candidate = pd.read_csv(upload)
            st.write("معاينة الملف", candidate.head())
            required = {"date","oil_production_bopd"}
            if not required.issubset(candidate.columns):
                st.error("الأعمدة date و oil_production_bopd مطلوبة بالأسماء نفسها. عدّل أسماء الأعمدة في CSV ثم ارفعه من جديد.")
            else:
                if st.button("اعتماد الملف في الجلسة"):
                    st.session_state.uploaded_data = candidate
                    st.success("تم اعتماد البيانات لهذه الجلسة. فعّل خيار استخدام بيانات CSV في الشريط الجانبي.")
        except Exception as e:
            st.error(f"تعذر قراءة CSV: {e}")
    st.divider()
    st.markdown("### تنزيل البيانات الافتراضية")
    st.download_button("تنزيل البيانات الافتراضية CSV", synthetic.to_csv(index=False).encode("utf-8-sig"),
                       file_name="synthetic_oilfield_data.csv", mime="text/csv")
    st.markdown("### تصدير تقرير PDF")
    if st.button("إنشاء تقرير PDF"):
        pdf = make_pdf_report("Libya OilCast AI — Oil Production Forecast Report", metrics, pd.DataFrame())
        if pdf:
            st.download_button("تنزيل التقرير PDF", pdf, file_name="libya_oilcast_report.pdf", mime="application/pdf")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

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
