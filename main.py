import streamlit as st
import pandas as pd
from pathlib import Path
import unicodedata
import io
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="다양한 환경 변동과 나도수영의 생장률 분석",
    layout="wide"
)

# 한글 폰트 (Streamlit)
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
def normalize(name: str) -> str:
    return unicodedata.normalize("NFC", name)

def variation_rate(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return (series.max() - series.min()) / series.mean()

# ===============================
# 데이터 로딩
# ===============================
@st.cache_data
def load_environment_data():
    env = {}
    data_dir = Path("data")
    if not data_dir.exists():
        return env

    for file in data_dir.iterdir():
        if file.suffix.lower() == ".csv":
            name = normalize(file.name)
            try:
                df = pd.read_csv(file)
                school = name.replace("_환경데이터.csv", "")
                env[school] = df
            except Exception:
                continue
    return env

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    target = None
    for file in data_dir.iterdir():
        if file.suffix.lower() == ".xlsx" and "생육결과데이터" in normalize(file.name):
            target = file
            break
    if target is None:
        return {}

    xls = pd.ExcelFile(target, engine="openpyxl")
    result = {}
    for sheet in xls.sheet_names:
        try:
            result[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue
    return result

with st.spinner("데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

if not env_data or not growth_data:
    st.error("데이터를 불러올 수 없습니다. data 폴더를 확인하세요.")
    st.stop()

# ===============================
# 메타 정보
# ===============================
EC_INFO = {
    "송도고": 1.0,
    "하늘고": 2.0,
    "아라고": 4.0,
    "동산고": 8.0
}

schools = ["전체"] + list(EC_INFO.keys())
selected_school = st.sidebar.selectbox("학교 선택", schools)

# ===============================
# 제목
# ===============================
st.title("🌱 다양한 환경 변동과 나도수영의 생장률 분석")

tab1, tab2, tab3 = st.tabs(["실험 개요", "환경 데이터 분석", "결과 분석"])

# ==================================================
# Tab 1 : 실험 개요
# ==================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.markdown(
        """
        본 연구는 학교별로 상이한 환경 조건에서 재배된 **극지식물(나도수영)**의
        생육 결과를 비교하여,
        **환경 요소의 변동성(변동률)이 생중량에 미치는 영향**을 정량적으로 분석하는 것을 목표로 한다.
        """
    )

    avg_rows = []
    for school, df in env_data.items():
        avg_rows.append([
            school,
            df["temperature"].mean(),
            df["humidity"].mean(),
            df["ph"].mean(),
            df["ec"].mean()
        ])

    avg_df = pd.DataFrame(
        avg_rows,
        columns=["학교", "온도", "습도", "pH", "EC"]
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["평균 온도", "평균 습도", "평균 pH", "평균 EC"]
    )

    fig.add_bar(x=avg_df["학교"], y=avg_df["온도"], row=1, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["습도"], row=1, col=2)
    fig.add_bar(x=avg_df["학교"], y=avg_df["pH"], row=2, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["EC"], row=2, col=2)

    fig.update_layout(font=PLOTLY_FONT, height=700)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# Tab 2 : 환경 데이터 분석
# ==================================================
with tab2:
    st.subheader("환경 변동률과 생중량 비교")

    rows = []
    for school, env_df in env_data.items():
        gdf = growth_data.get(school)
        if gdf is None:
            continue
        rows.append([
            school,
            variation_rate(env_df["temperature"]),
            variation_rate(env_df["humidity"]),
            variation_rate(env_df["ph"]),
            variation_rate(env_df["ec"]),
            gdf["생중량(g)"].mean()
        ])

    vdf = pd.DataFrame(
        rows,
        columns=["학교", "온도 변동률", "습도 변동률", "pH 변동률", "EC 변동률", "평균 생중량"]
    )

    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "온도 변동률 vs 생중량",
            "습도 변동률 vs 생중량",
            "pH 변동률 vs 생중량",
            "EC 변동률 vs 생중량"
        ],
        specs=[[{"secondary_y": True}]*2]*2
    )

    indicators = ["온도 변동률", "습도 변동률", "pH 변동률", "EC 변동률"]
    positions = [(1,1),(1,2),(2,1),(2,2)]

    for ind, (r,c) in zip(indicators, positions):
        fig2.add_bar(x=vdf["학교"], y=vdf[ind], row=r, col=c, name=ind)
        fig2.add_scatter(
            x=vdf["학교"],
            y=vdf["평균 생중량"],
            mode="lines+markers",
            row=r, col=c,
            secondary_y=True,
            name="생중량"
        )

    fig2.update_layout(font=PLOTLY_FONT, height=800)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# Tab 3 : 결과 분석
# ==================================================
with tab3:
    st.subheader("환경 변동률과 생중량 간 상관계수")

    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "하늘고", "동산고",
            "아라고", "송도고"
        ]
    )

    school_positions = {
        "하늘고": (1,1),
        "동산고": (1,2),
        "아라고": (2,1),
        "송도고": (2,2)
    }

    for school, (r,c) in school_positions.items():
        env_df = env_data.get(school)
        gdf = growth_data.get(school)
        if env_df is None or gdf is None:
            continue

        corr = [
            np.corrcoef(env_df["temperature"][:len(gdf)], gdf["생중량(g)"])[0,1],
            np.corrcoef(env_df["humidity"][:len(gdf)], gdf["생중량(g)"])[0,1],
            np.corrcoef(env_df["ph"][:len(gdf)], gdf["생중량(g)"])[0,1],
            np.corrcoef(env_df["ec"][:len(gdf)], gdf["생중량(g)"])[0,1]
        ]

        fig3.add_bar(
            x=["온도", "습도", "pH", "EC"],
            y=corr,
            row=r, col=c
        )

    fig3.update_layout(font=PLOTLY_FONT, height=800)
    st.plotly_chart(fig3, use_container_width=True)

    buffer = io.BytesIO()
    vdf.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    st.download_button(
        "분석 요약 XLSX 다운로드",
        data=buffer,
        file_name="환경변동률_생중량_분석.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
