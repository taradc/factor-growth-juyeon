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
def normalize(name: str) -> str:
    return unicodedata.normalize("NFC", name)

def variation_rate(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return (series.max() - series.min()) / series.mean()

# ===============================
# 기간 정보
# ===============================
PERIODS = {
    "동산고": ("2024-06-19", "2024-07-17"),
    "송도고": ("2024-05-19", "2024-07-10"),
    "하늘고": ("2024-05-30", "2024-07-08"),
    "아라고": ("2024-05-26", "2024-06-24"),
}

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
                df["time"] = pd.to_datetime(df["time"])
                school = name.replace("_환경데이터.csv", "")
                if school in PERIODS:
                    start, end = PERIODS[school]
                    df = df[(df["time"] >= start) & (df["time"] <= end)]
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
# 사이드바
# ===============================
schools = ["전체"] + list(env_data.keys())
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
        본 연구는 학교별로 상이한 환경 조건에서 재배된 극지식물(나도수영)의
        생육 결과를 비교하여 환경 요인이 생중량에 미치는 영향을 분석하는 것을 목표로 한다.

        **연구 목적**
        - EC만을 기준으로 생육을 설명하는 데에는 한계가 있음  
          (명목상 EC 1·2·4·8이었으나 실제 측정값은 약 0.7, 1, 4, 7.8 수준)
        - 단일 변수(EC) 중심 해석은 현실 환경을 충분히 반영하지 못함
        - 다른 조에서도 언급되었듯, 다양한 환경 변수의 동시 고려가 필요함

        **변동량(변화율)을 사용하는 이유**
        1. 나도수영은 극지 환경에 적응한 식물로, 절대값보다 환경 변화에 대한 내성이 중요함  
        2. 제한된 데이터 조건에서 평균보다 더 많은 정보를 끌어내기 위해 변동성 지표를 활용함  
        3. 환경의 ‘안정성’이 생육에 미치는 영향을 정량적으로 비교하기 위함
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

    fig = go.Figure()
    indicators = ["온도", "습도", "pH", "EC"]

    for ind in indicators:
        fig.add_bar(
            x=avg_df["학교"],
            y=avg_df[ind],
            name=ind
        )

    fig.update_layout(
        barmode="group",
        title="학교별 환경 지표 평균 비교",
        font=PLOTLY_FONT,
        height=600
    )

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
            gdf["생중량(g)]()
