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
    df = df.dropna(subset=["time"])
    start, end = pd.to_datetime(start), pd.to_datetime(end)
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
            school = normalize_text(f.name).replace("_환경데이터.csv", "")
            env[school] = pd.read_csv(f)
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
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# ===============================
# 기간 정보
# ===============================
PERIODS = {
    "동산고": ("2024-06-19", "2024-07-17"),
    "송도고": ("2024-05-19", "2024-07-10"),
    "하늘고": ("2024-05-30", "2024-07-08"),
    "아라고": ("2024-05-26", "2024-06-24")
}

# ===============================
# UI
# ===============================
st.title("다양한 환경 변동과 나도수영의 생장률 분석")
tab1, tab2, tab3 = st.tabs(["실험 개요", "환경 데이터 분석", "결과 분석"])

# ==================================================
# Tab 1 : 실험 개요 (복원)
# ==================================================
with tab1:
    st.subheader("학교별 평균 환경 조건 비교")

    rows = []
    for school in env_data:
        env = filter_by_period_safe(env_data[school], *PERIODS[school])
        rows.append([
            school,
            env["temperature"].mean(),
            env["humidity"].mean(),
            env["ph"].mean(),
            env["ec"].mean()
        ])

    avg_df = pd.DataFrame(
        rows,
        columns=["학교", "온도", "습도", "pH", "EC"]
    )

    fig1 = go.Figure()
    for col in ["온도", "습도", "pH", "EC"]:
        fig1.add_bar(
            x=avg_df["학교"],
            y=avg_df[col],
            name=col
        )

    fig1.update_layout(
        barmode="group",
        title="학교별 환경 지표 평균값",
        font=PLOTLY_FONT,
        height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
본 실험은 네 개 학교에서 재배된 나도수영의 생육 데이터를 기반으로,
각 학교의 환경 조건과 생장 결과를 비교·분석하는 것을 목표로 한다.

환경 요인은 온도, 습도, pH, EC로 설정하였으며,
단일 값 비교의 한계를 고려하여 이후 분석에서는 **변동률 및 변화량**을 중심으로 접근하였다.
""")

# ==================================================
# Tab 2 : 환경 데이터 분석 (이전과 동일)
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

    def add_pair(x, y1, y2, r, c):
        fig2.add_bar(x=x, y=y1, row=r, col=c)
        fig2.add_scatter(x=x, y=y2, secondary_y=True, row=r, col=c)

    add_pair(vdf["학교"], vdf["온도 변동률"], vdf["평균 생중량"], 1, 1)
    add_pair(vdf["학교"], vdf["습도 변동률"], vdf["평균 생중량"], 1, 2)
    add_pair(vdf["학교"], vdf["pH 변동률"], vdf["평균 생중량"], 2, 1)
    add_pair(vdf["학교"], vdf["습도 평균"], vdf["평균 생중량"], 2, 2)
    add_pair(vdf["학교"], vdf["EC 변동률"], vdf["평균 생중량"], 3, 1)

    fig2.update_layout(font=PLOTLY_FONT, height=1000)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# Tab 3 : 결과 분석 (요소별 해설 추가)
# ==================================================
with tab3:
    factors = ["temperature", "humidity", "ph", "ec"]
    names = ["온도", "습도", "pH", "EC"]
    corrs = []

    for f in factors:
        vals = []
        for school in env_data:
            env = filter_by_period_safe(env_data[school], *PERIODS[school])
            gdf = growth_data[school]
            n = min(len(env), len(gdf))
            if n >= 2:
                vals.append(np.corrcoef(env[f][:n], gdf["생중량(g)"][:n])[0, 1])
        corrs.append(np.nanmean(vals))

    corr_df = pd.DataFrame({"환경 요인": names, "평균 상관계수": corrs})

    fig3 = go.Figure([go.Bar(x=names, y=corrs)])
    fig3.update_layout(
        title="환경 요인별 생중량과의 평균 상관계수",
        font=PLOTLY_FONT,
        height=450
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
### 요소별 해석

**온도**  
온도는 생장에 필수적인 요인이지만, 네 학교 모두 극단적인 온도 차이가 크지 않아
생중량과의 상관관계는 비교적 약하게 나타났다.
이는 나도수영이 온도 변화에 대해 일정 수준의 적응성을 가진 종임을 시사한다.

**습도**  
습도는 증산 작용과 직결되는 요인으로,
변동률 기준 분석에서 생중량과의 관계가 더 뚜렷하게 나타났다.
이는 평균 습도보다 **환경의 안정성**이 생장에 더 중요함을 의미한다.

**pH**  
pH는 직접적인 에너지원은 아니지만,
양분 흡수 효율에 영향을 주는 간접 요인이다.
학교 간 pH 변화 폭이 작아 상관관계는 제한적으로 나타났다.

**EC**  
EC는 단순 절대값 비교 시
1, 2, 4, 8과 같은 이상적인 단계가 아닌
약 0.7, 1, 4, 7.8 수준으로 분포하였다.
이로 인해 EC 단일 변수만으로 생장을 설명하기는 어려웠으며,
변동률을 통해 접근했을 때 상대적으로 의미 있는 경향이 관측되었다.

이는 극지 환경에 적응한 나도수영이
절대적인 환경 값보다는 **환경 변화에 대한 대응 능력**에 의해
생장 특성이 달라질 수 있음을 시사한다.
""")
