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
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def variation_rate(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2 or series.mean() == 0:
        return np.nan
    return (series.max() - series.min()) / series.mean()

def filter_by_period(df, start, end):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df[(df["time"] >= start) & (df["time"] <= end)]

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
            fname = normalize_text(file.name)
            if "환경데이터" in fname:
                try:
                    df = pd.read_csv(file)
                    school = fname.replace("_환경데이터.csv", "")
                    env[school] = df
                except Exception:
                    continue
    return env

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    target = None

    for file in data_dir.iterdir():
        if file.suffix.lower() == ".xlsx" and "생육결과데이터" in normalize_text(file.name):
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
    st.error("데이터를 불러올 수 없습니다. data 폴더와 파일명을 확인하세요.")
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

PERIODS = {
    "동산고": ("2024-06-19", "2024-07-17"),
    "송도고": ("2024-05-19", "2024-07-10"),
    "하늘고": ("2024-05-30", "2024-07-08"),
    "아라고": ("2024-05-26", "2024-06-24")
}

schools = ["전체"] + list(EC_INFO.keys())
selected_school = st.sidebar.selectbox("학교 선택", schools)

# ===============================
# 제목
# ===============================
st.title("다양한 환경 변동과 나도수영의 생장률 분석")

tab1, tab2, tab3 = st.tabs(["실험 개요", "환경 데이터 분석", "결과 분석"])

# ==================================================
# Tab 1 : 실험 개요
# ==================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.markdown(
        """
        **목적**  
        본 연구는 극지식물 *나도수영*의 생육을 단일 환경 요인(EC 농도)만으로 설명하기 어렵다는
        실험적 한계에서 출발하였다. 실제 실험에서 설정된 EC 조건은
        1, 2, 4, 8이 아닌 약 **0.7, 1, 4, 7.8 수준**으로 완전히 분리되지 않았으며,
        이로 인해 EC 단독 요인의 영향 분석에는 제한이 존재한다.

        따라서 본 연구에서는 온도, 습도, pH, EC 등 **다수의 환경 변수를 동시에 고려**하고,
        절대값이 아닌 **환경 변동량(변동률)**을 사용하여 생육 반응을 분석하였다.

        변동률을 사용하는 이유는 다음과 같다.  
        1. 나도수영은 극지 환경에 적응한 식물로, 절대 조건보다 **환경 변화에 대한 반응성**이 중요할 가능성이 있다.  
        2. 주어진 제한된 데이터에서 **추가적인 해석 정보를 최대한 도출**하기 위함이다.
        """
    )

    avg_rows = []
    for school, df in env_data.items():
        start, end = PERIODS[school]
        df_f = filter_by_period(df, start, end)
        avg_rows.append([
            school,
            df_f["temperature"].mean(),
            df_f["humidity"].mean(),
            df_f["ph"].mean(),
            df_f["ec"].mean()
        ])

    avg_df = pd.DataFrame(
        avg_rows,
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
        title="학교별 환경 지표 평균 비교",
        font=PLOTLY_FONT,
        height=600
    )
    st.plotly_chart(fig1, use_container_width=True)

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

        start, end = PERIODS[school]
        env_df = filter_by_period(env_df, start, end)

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
        fig2.add_bar(x=vdf["학교"], y=vdf[ind], row=r, col=c)
        fig2.add_scatter(
            x=vdf["학교"],
            y=vdf["평균 생중량"],
            mode="lines+markers",
            secondary_y=True,
            row=r, col=c
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
        subplot_titles=["하늘고", "동산고", "아라고", "송도고"]
    )

    positions = {
        "하늘고": (1,1),
        "동산고": (1,2),
        "아라고": (2,1),
        "송도고": (2,2)
    }

    for school, (r,c) in positions.items():
        env_df = env_data[school]
        gdf = growth_data[school]

        start, end = PERIODS[school]
        env_df = filter_by_period(env_df, start, end)

        n = min(len(env_df), len(gdf))

        corr = [
            np.corrcoef(env_df["temperature"][:n], gdf["생중량(g)"][:n])[0,1],
            np.corrcoef(env_df["humidity"][:n], gdf["생중량(g)"][:n])[0,1],
            np.corrcoef(env_df["ph"][:n], gdf["생중량(g)"][:n])[0,1],
            np.corrcoef(env_df["ec"][:n], gdf["생중량(g)"][:n])[0,1]
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
        label="분석 요약 XLSX 다운로드",
        data=buffer,
        file_name="환경변동률_생중량_분석.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
