import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="다양한 환경 변동과 나도수영의 생장률 분석",
    layout="wide"
)

# 한글 폰트
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")

# ===============================
# 유틸 함수
# ===============================
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def variation_rate(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    mean = series.mean()
    if mean == 0:
        return np.nan
    return (series.max() - series.min()) / mean

def filter_by_period_safe(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    filtered = df[(df["time"] >= start) & (df["time"] <= end)]
    return filtered if not filtered.empty else df

# ===============================
# 데이터 로딩
# ===============================
@st.cache_data
def load_environment_data():
    env = {}
    data_dir = Path("data")
    for file in data_dir.iterdir():
        if file.suffix == ".csv" and "환경데이터" in normalize_text(file.name):
            school = normalize_text(file.name).replace("_환경데이터.csv", "")
            env[school] = pd.read_csv(file)
    return env

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    for file in data_dir.iterdir():
        if file.suffix == ".xlsx" and "생육결과데이터" in normalize_text(file.name):
            xls = pd.ExcelFile(file, engine="openpyxl")
            return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
    return {}

env_data = load_environment_data()
growth_data = load_growth_data()

if not env_data or not growth_data:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# ===============================
# 메타 정보
# ===============================
PERIODS = {
    "동산고": ("2024-06-19", "2024-07-17"),
    "송도고": ("2024-05-19", "2024-07-10"),
    "하늘고": ("2024-05-30", "2024-07-08"),
    "아라고": ("2024-05-26", "2024-06-24")
}

st.title("다양한 환경 변동과 나도수영의 생장률 분석")

tab1, tab2, tab3 = st.tabs(["실험 개요", "환경 데이터 분석", "결과 분석"])

# ==================================================
# Tab 2 : 환경 데이터 분석 (습도 분할)
# ==================================================
with tab2:
    rows = []
    for school in env_data:
        env = filter_by_period_safe(env_data[school], *PERIODS[school])
        gdf = growth_data[school]

        rows.append([
            school,
            variation_rate(env["temperature"]),
            variation_rate(env["humidity"]),
            env["humidity"].mean(),
            variation_rate(env["ph"]),
            variation_rate(env["ec"]),
            gdf["생중량(g)"].mean()
        ])

    vdf = pd.DataFrame(rows, columns=[
        "학교", "온도 변동률", "습도 변동률", "습도 평균",
        "pH 변동률", "EC 변동률", "평균 생중량"
    ])

    fig2 = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "온도 변동률 vs 생중량",
            "습도 변동률 vs 생중량",
            "pH 변동률 vs 생중량",
            "습도 절대값 vs 생중량",
            "EC 변동률 vs 생중량"
        ],
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": True}, None]
        ]
    )

    fig2.add_bar(x=vdf["학교"], y=vdf["온도 변동률"], row=1, col=1)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=1, col=1)

    fig2.add_bar(x=vdf["학교"], y=vdf["습도 변동률"], row=1, col=2)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=1, col=2)

    fig2.add_bar(x=vdf["학교"], y=vdf["pH 변동률"], row=2, col=1)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=2, col=1)

    fig2.add_bar(x=vdf["학교"], y=vdf["습도 평균"], row=2, col=2)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=2, col=2)

    fig2.add_bar(x=vdf["학교"], y=vdf["EC 변동률"], row=3, col=1)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=3, col=1)

    fig2.update_layout(font=PLOTLY_FONT, height=1000)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# Tab 3 : 결과 분석 (요인별 상관계수 + 해석)
# ==================================================
with tab3:
    factors = ["temperature", "humidity", "ph", "ec"]
    factor_names = ["온도", "습도", "pH", "EC"]

    corr_means = []
    for f in factors:
        vals = []
        for school in env_data:
            env = filter_by_period_safe(env_data[school], *PERIODS[school])
            gdf = growth_data[school]
            n = min(len(env), len(gdf))
            if n >= 2:
                vals.append(np.corrcoef(env[f][:n], gdf["생중량(g)"][:n])[0, 1])
        corr_means.append(np.nanmean(vals))

    corr_df = pd.DataFrame({
        "환경 요인": factor_names,
        "평균 상관계수": corr_means
    })

    fig3 = go.Figure(
        data=[go.Bar(x=corr_df["환경 요인"], y=corr_df["평균 상관계수"])]
    )
    fig3.update_layout(
        title="환경 요인별 생중량과의 평균 상관계수",
        font=PLOTLY_FONT,
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
**상관관계 해석**

상관계수는 두 변수 간의 관계 방향과 강도를 나타내는 지표이다.  
양의 상관관계는 환경 요인이 증가할수록 생중량이 증가하는 경향을 의미하며,
음의 상관관계는 환경 요인이 증가할수록 생중량이 감소하는 경향을 의미한다.

요인별 분석 결과, 환경 변수에 따라 생중량과의 관계 강도는 서로 다르게 나타났다.
특히 **EC의 경우**, 네 학교를 종합해 분석했을 때 변동률 기준으로 비교적
강한 음의 상관관계가 도출되었다.

다만 EC 절대값이 특히 높았던 **아라고**의 경우,
EC 변동률은 낮았음에도 불구하고 생장률이 낮게 관측되었다.
이는 EC 단일 요인보다는 **환경 변화의 맥락과 복합 요인**을 함께 고려해야 함을 시사한다.
""")

    buffer = io.BytesIO()
    corr_df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    st.download_button(
        "상관계수 요약 XLSX 다운로드",
        data=buffer,
        file_name="환경요인별_상관계수_분석.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
