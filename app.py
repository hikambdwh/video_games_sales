import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import pickle


st.set_page_config(
    page_title="Video Game Sales Dashboard",
    page_icon="🎮",
    layout="wide"
)


st.markdown(
    """
    <style>
    /* Background utama */
    .stApp {
        background: radial-gradient(circle at top left, #172554 0, #020617 40%, #000000 100%);
        color: #e5e7eb;
    }

    section.main > div {
        padding-top: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #020617 40%, #111827 100%);
        border-right: 1px solid #1f2937;
    }

    /* Judul utama */
    .title-text {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #fbbf24;
        text-shadow: 0 0 12px rgba(251,191,36,0.8), 0 0 28px rgba(59,130,246,0.7);
    }

    .subtitle-text {
        font-size: 0.95rem;
        color: #9ca3af;
    }

    /* Kartu metric */
    .metric-card {
        background: radial-gradient(circle at top, rgba(59,130,246,0.25), rgba(15,23,42,0.95));
        border-radius: 0.75rem;
        padding: 0.9rem 1.1rem;
        border: 1px solid rgba(148,163,184,0.3);
        box-shadow: 0 0 20px rgba(59,130,246,0.25);
    }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #9ca3af;
        letter-spacing: 0.08em;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e5e7eb;
    }

    /* Judul section */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }

    /* Biar background chart transparan */
    .vega-embed svg, .vega-embed canvas {
        background: transparent !important;
    }

    /* Kartu hasil prediksi */
    .prediction-card {
        margin-top: 1rem;
        padding: 1.1rem 1.3rem;
        border-radius: 0.75rem;
        background: radial-gradient(circle at top, rgba(22,163,74,0.4), rgba(15,23,42,0.95));
        border: 1px solid rgba(34,197,94,0.8);
        box-shadow: 0 0 24px rgba(34,197,94,0.6);
    }

    .prediction-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #bbf7d0;
        letter-spacing: 0.08em;
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 800;
        color: #dcfce7;
    }

    .prediction-sub {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load dataset vgsales.csv.
    Kolom yang diharapkan (dataset Kaggle vgsales):
    ['Rank','Name','Platform','Year','Genre','Publisher',
     'NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']
    """
    csv_path = Path("dataset/vgsales.csv")

    if not csv_path.exists():
        st.error(
            "File 'dataset/vgsales.csv' tidak ditemukan.\n\n"
            "Pastikan kamu sudah meletakkan dataset di folder `dataset/`."
        )
        st.stop()

    df = pd.read_csv(csv_path)

    # Bersihkan kolom Year
    if "Year" in df.columns:
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)

    return df


def _find_model_path() -> Path | None:
    """
    Cari file model di beberapa lokasi umum:
    - model/video_game_sales.pkl
    - video_game_sales.pkl (root)
    """
    candidates = [
        Path("model/video_game_sales.pkl"),
        Path("video_game_sales.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = _find_model_path()
    if model_path is None:
        return None, "File model 'video_game_sales.pkl' tidak ditemukan. " \
                     "Letakkan di folder `model/` atau di root yang sama dengan app.py."

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        # Supaya Streamlit tidak crash kalau versi sklearn/numpy beda
        return None, f"Gagal memuat model dari '{model_path}': {e}"


df = load_data()
rf_model, model_error = load_model()


st.sidebar.markdown(
    "Sesuaikan filter di bawah untuk mengeksplorasi data penjualan video game."
)

min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider(
    "Rentang Tahun",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

genre_options = ["Semua Genre"] + sorted(df["Genre"].dropna().unique())
genre_selected = st.sidebar.selectbox("Pilih Genre", genre_options)

st.sidebar.markdown("---")
st.sidebar.image(
    "assets/vgsales_dashboard.png",
    caption="Dashboard Tableau (referensi)",
    use_column_width=True,
)

# Filter data sesuai pilihan
df_filtered = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]
if genre_selected != "Semua Genre":
    df_filtered = df_filtered[df_filtered["Genre"] == genre_selected]

# HEADER
st.markdown('<div class="title-text">Video Game Sales Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">'
    'Analisis & prediksi penjualan video game global berdasarkan genre dan wilayah '
    '(NA, EU, JP, Other) menggunakan dataset <b>vgsales</b>.'
    '</div>',
    unsafe_allow_html=True
)
st.markdown("")


col1, col2, col3, col4 = st.columns(4)

total_global = df_filtered["Global_Sales"].sum()
total_games = len(df_filtered)
avg_sales = df_filtered["Global_Sales"].mean() if total_games > 0 else 0
top_genre = (
    df_filtered.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False).index[0]
    if not df_filtered.empty else "-"
)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Total Penjualan Global</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_global:,.1f} juta kopi</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Jumlah Game</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{total_games:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Rata-rata Penjualan per Game</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{avg_sales:,.2f} juta</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Genre Terlaris</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{top_genre}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

# BAR CHART
st.markdown('<div class="section-title">Penjualan Global berdasarkan Genre</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">'
    'Total penjualan global per genre pada rentang tahun yang dipilih.'
    '</div>',
    unsafe_allow_html=True
)

genre_sales = (
    df_filtered.groupby("Genre")["Global_Sales"]
    .sum()
    .reset_index()
    .sort_values(by="Global_Sales", ascending=False)
)

if genre_sales.empty:
    st.info("Tidak ada data untuk filter yang dipilih.")
else:
    genre_palette = [
        "#22c55e", "#0ea5e9", "#a855f7", "#eab308",
        "#f97316", "#fb7185", "#38bdf8", "#2dd4bf"
    ]

    bar = (
        alt.Chart(genre_sales)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Genre:N", sort="-y", title="Genre"),
            y=alt.Y("Global_Sales:Q", title="Total Penjualan Global (juta)"),
            color=alt.Color("Genre:N", legend=None,
                            scale=alt.Scale(range=genre_palette)),
            tooltip=[
                alt.Tooltip("Genre:N", title="Genre"),
                alt.Tooltip("Global_Sales:Q", title="Penjualan Global (juta)", format=",.2f"),
            ],
        )
        .properties(height=380)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#e5e7eb",
            titleColor="#e5e7eb",
            gridColor="#1f2937",
        )
    )

    st.altair_chart(bar, use_container_width=True)


# LINE CHART – Tren Penjualan Global per Tahun

st.markdown('<div class="section-title">Tren Penjualan Global</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">'
    'Pergerakan total penjualan global per tahun.'
    '</div>',
    unsafe_allow_html=True
)

yearly_global = (
    df_filtered.groupby("Year")["Global_Sales"]
    .sum()
    .reset_index()
    .sort_values("Year")
)

if not yearly_global.empty:
    line_global = (
        alt.Chart(yearly_global)
        .mark_line(point=True, interpolate="monotone")
        .encode(
            x=alt.X("Year:O", title="Tahun"),
            y=alt.Y("Global_Sales:Q", title="Penjualan Global (juta)"),
            color=alt.value("#38bdf8"),
            tooltip=[
                alt.Tooltip("Year:O", title="Tahun"),
                alt.Tooltip("Global_Sales:Q", title="Penjualan Global (juta)", format=",.2f"),
            ],
        )
        .properties(height=360)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#e5e7eb",
            titleColor="#e5e7eb",
            gridColor="#111827",
        )
    )

    st.altair_chart(line_global, use_container_width=True)
else:
    st.info("Tidak ada data tren global untuk filter yang dipilih.")


# LINE CHART – Penjualan per Region (NA, JP, EU)

st.markdown('<div class="section-title">Tren Penjualan per Region</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">'
    'Perbandingan penjualan per tahun untuk North America, Jepang, dan Eropa.'
    '</div>',
    unsafe_allow_html=True
)

region_cols = ["NA_Sales", "EU_Sales", "JP_Sales"]
available_regions = [c for c in region_cols if c in df_filtered.columns]

if available_regions:
    yearly_region = (
        df_filtered.groupby("Year")[available_regions]
        .sum()
        .reset_index()
        .sort_values("Year")
    )

    col_na, col_jp, col_eu = st.columns(3)

    if "NA_Sales" in available_regions:
        na_chart = (
            alt.Chart(yearly_region)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X("Year:O", title="Tahun"),
                y=alt.Y("NA_Sales:Q", title="NA Sales (juta)"),
                color=alt.value("#f97316"),
                tooltip=[
                    alt.Tooltip("Year:O", title="Tahun"),
                    alt.Tooltip("NA_Sales:Q", title="NA Sales (juta)", format=",.2f"),
                ],
            )
            .properties(title="North America", height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor="#e5e7eb",
                titleColor="#e5e7eb",
                gridColor="#111827",
            )
        )
        with col_na:
            st.altair_chart(na_chart, use_container_width=True)

    if "JP_Sales" in available_regions:
        jp_chart = (
            alt.Chart(yearly_region)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X("Year:O", title="Tahun"),
                y=alt.Y("JP_Sales:Q", title="JP Sales (juta)"),
                color=alt.value("#a855f7"),
                tooltip=[
                    alt.Tooltip("Year:O", title="Tahun"),
                    alt.Tooltip("JP_Sales:Q", title="JP Sales (juta)", format=",.2f"),
                ],
            )
            .properties(title="Jepang", height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor="#e5e7eb",
                titleColor="#e5e7eb",
                gridColor="#111827",
            )
        )
        with col_jp:
            st.altair_chart(jp_chart, use_container_width=True)

    if "EU_Sales" in available_regions:
        eu_chart = (
            alt.Chart(yearly_region)
            .mark_line(point=True, interpolate="monotone")
            .encode(
                x=alt.X("Year:O", title="Tahun"),
                y=alt.Y("EU_Sales:Q", title="EU Sales (juta)"),
                color=alt.value("#22c55e"),
                tooltip=[
                    alt.Tooltip("Year:O", title="Tahun"),
                    alt.Tooltip("EU_Sales:Q", title="EU Sales (juta)", format=",.2f"),
                ],
            )
            .properties(title="Eropa", height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor="#e5e7eb",
                titleColor="#e5e7eb",
                gridColor="#111827",
            )
        )
        with col_eu:
            st.altair_chart(eu_chart, use_container_width=True)
else:
    st.info("Kolom penjualan region (NA, EU, JP) tidak ditemukan di dataset.")


# PREDIKSI GLOBAL SALES MENGGUNAKAN MODEL

st.markdown('<div class="section-title">Prediksi Global Sales</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">'
    'Gunakan model Random Forest yang sudah kamu latih '
    'untuk memprediksi <b>Global_Sales</b> berdasarkan NA, EU, JP, dan Other Sales.'
    '</div>',
    unsafe_allow_html=True
)

if rf_model is None:
    st.warning(model_error)
else:
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            na_input = st.number_input(
                "NA_Sales (juta)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Perkiraan penjualan di North America dalam satuan juta."
            )
            eu_input = st.number_input(
                "EU_Sales (juta)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Perkiraan penjualan di Eropa dalam satuan juta."
            )
        with c2:
            jp_input = st.number_input(
                "JP_Sales (juta)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Perkiraan penjualan di Jepang dalam satuan juta."
            )
            other_input = st.number_input(
                "Other_Sales (juta)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                help="Perkiraan penjualan di region lain (selain NA/EU/JP) dalam satuan juta."
            )

        submitted = st.form_submit_button("🎯 Prediksi Penjualan Global")

    if submitted:
        input_df = pd.DataFrame(
            {
                "NA_Sales": [na_input],
                "EU_Sales": [eu_input],
                "JP_Sales": [jp_input],
                "Other_Sales": [other_input],
            }
        )

        try:
            prediction = float(rf_model.predict(input_df)[0])
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Prediksi Global Sales</div>
                    <div class="prediction-value">{prediction:,.2f} juta kopi</div>
                    <div class="prediction-sub">
                        Berdasarkan input NA/EU/JP/Other Sales yang kamu berikan.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")


# FOOTER

st.markdown("---")
st.markdown(
    '<div style="font-size:0.75rem; color:#6b7280;">'
    '🎮 Dibuat dengan Streamlit • Tema neon bertema video game • '
    'Model: RandomForestRegressor (video_game_sales.pkl) • Dataset: vgsales'
    '</div>',
    unsafe_allow_html=True
)
