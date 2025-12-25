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

def filter_by_period_safe(df, start, end):
    df = df.copy()
    if "time" not in df.columns:
        return df
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    filtered = df[(df["time"] >= start) & (df["time"] <= end)]
    return filtered if not filtered.empty else df

# ===============================
# 데이터 로딩
# ===============================
@st.cache_data
def load_environment_data():
    env = {}
    for f in Path("data").iterdir():
        if f.suffix == ".csv" and "환경데이터" in normalize_text(f.name):
            env[normalize_text(f.name).replace("_환경데이터.csv", "")] = pd.read_csv(f)
    return env

@st.cache_data
def load_growth_data():
    for f in Path("data").iterdir():
        if f.suffix == ".xlsx" and "생육결과데이터" in normalize_text(f.name):
            xls = pd.ExcelFile(f, engine="openpyxl")
            return {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names}
    return {}

env_data = load_environment_data()
growth_data = load_growth_data()

if not env_data or not growth_data:
    st.error("데이터 로딩 실패")
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
# Tab 2 : 환경 데이터 분석 (습도 분리)
# ==================================================
with tab2:
    rows = []
    for s in env_data:
        env = filter_by_period_safe(env_data[s], *PERIODS[s])
        g = growth_data[s]
        rows.append([
            s,
            variation_rate(env["temperature"]),
            variation_rate(env["humidity"]),
            env["humidity"].mean(),
            variation_rate(env["ph"]),
            variation_rate(env["ec"]),
            g["생중량(g)"].mean()
        ])

    vdf = pd.DataFrame(rows, columns=[
        "학교","온도 변동률","습도 변동률","습도 평균","pH 변동률","EC 변동률","평균 생중량"
    ])

    fig2 = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "온도 변동률 vs 생중량",
            "습도 변동률 vs 생중량",
            "습도 절대값 vs 생중량",
            "pH 변동률 vs 생중량",
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

    fig2.add_bar(x=vdf["학교"], y=vdf["습도 평균"], row=2, col=2)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=2, col=2)

    fig2.add_bar(x=vdf["학교"], y=vdf["pH 변동률"], row=2, col=1)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=2, col=1)

    fig2.add_bar(x=vdf["학교"], y=vdf["EC 변동률"], row=3, col=1)
    fig2.add_scatter(x=vdf["학교"], y=vdf["평균 생중량"], secondary_y=True, row=3, col=1)

    fig2.update_layout(font=PLOTLY_FONT, height=1000)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# Tab 3 : 결과 분석 (요인별 상관계수)
# ==================================================
with tab3:
    factors = ["temperature","humidity","ph","ec"]
    factor_kor = ["온도","습도","pH","EC"]

    corr_result = []
    for f in factors:
        vals = []
        for s in env_data:
            env = filter_by_period_safe(env_data[s], *PERIODS[s])
            g = growth_data[s]
            n = min(len(env), len(g))
            if n >= 2:
                vals.append(np.corrcoef(env[f][:n], g["생중량(g)"][:n])[0,1])
        corr_result.append(np.nanmean(vals))

    corr_df = pd.DataFrame({"환경요인": factor_kor, "평균 상관계수": corr_result})

    fig3 = go.Figure(go.Bar(
        x=corr_df["환경요인"],
        y=corr_df["평균 상관계수"]
    ))
    fig3.update_layout(
        title="환경 요인별 생중량과의 평균 상관계수",
        font=PLOTLY_FONT,
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
**상관관계 해석**  
상관계수는 두 변수 간의 관계 방향과 강도를 나타낸다.  
양의 상관관계는 환경 요인이 증가할수록 생중량이 증가하는 경향을,
음의 상관관계는 환경 요인이 증가할수록 생중량이 감소하는 경향을 의미한다.

분석 결과, 환경 요인별로 생중량과의 상관 강도는 서로 다르게 나타났다.  
특히 **EC의 경우**, 네 학교 데이터를 종합했을 때 변동률 기준으로 비교적 강한
음의 상관관계가 관측되었다.  
다만 EC 절대값이 특히 높았던 아라고의 경우,
변동률은 낮았음에도 생장률이 낮게 나타나는 예외적 경향이 확인되었다.

이는 단일 요인보다는 **환경 변화의 맥락을 함께 고려해야 함**을 시사한다.
""")
