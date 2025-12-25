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
    df = df.sort_values("time")

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    filtered = df[(df["time"] >= start) & (df["time"] <= end)]

    # 🔴 기간 필터링 결과가 비면 → 전체 사용
    if filtered.empty:
        return df

    return filtered

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
                df = pd.read_csv(file)
                school = fname.replace("_환경데이터.csv", "")
                env[school] = df

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
        result[sheet] = pd.read_excel(xls, sheet_name=sheet)

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

st.sidebar.selectbox("학교 선택", ["전체"] + list(EC_INFO.keys()))

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
    st.markdown("""
**목적**  
본 연구는 극지식물 *나도수영*의 생육을 단일 환경 요인(EC 농도)만으로 설명하기 어렵다는
실험적 한계에서 출발하였다. 실제 EC 조건은 1·2·4·8이 아닌
약 **0.7·1·4·7.8 수준**으로 완전히 분리되지 않았다.

이에 따라 본 연구는 온도, 습도, pH, EC의 **변동률**을 중심으로
환경 변화에 대한 생육 반응을 분석하였다.

1. 극지식물은 절대 조건보다 환경 변화에 대한 적응 반응이 중요하다  
2. 제한된 데이터에서 최대한의 해석 정보를 도출한다  
3. 학교별 실험 기간 차이를 고려하여 분석한다
""")

    avg_rows = []
    for school, df in env_data.items():
        df_f = filter_by_period_safe(df, *PERIODS[school])
        avg_rows.append([
            school,
            df_f["temperature"].mean(),
            df_f["humidity"].mean(),
            df_f["ph"].mean(),
            df_f["ec"].mean()
        ])

    avg_df = pd.DataFrame(avg_rows, columns=["학교", "온도", "습도", "pH", "EC"])

    fig1 = go.Figure()
    for col in ["온도", "습도", "pH", "EC"]:
        fig1.add_bar(x=avg_df["학교"], y=avg_df[col], name=col)

    fig1.update_layout(
        barmode="group",
        title="학교별 환경 지표 평균",
        font=PLOTLY_FONT,
        height=600
    )
    st.plotly_chart(fig1, use_container_width=True)

# ==================================================
# Tab 2 : 환경 데이터 분석
# ==================================================
with tab2:
    rows = []
    for school, env_df in env_data.items():
        gdf = growth_data.get(school)
        if gdf is None:
            continue

        env_df = filter_by_period_safe(env_df, *PERIODS[school])

        rows.append([
            school,
            variation_rate(env_df["temperature"]),
            variation_rate(env_df["humidity"]),
            variation_rate(env_df["ph"]),
            variation_rate(env_df["ec"]),
            gdf["생중량(g)"].mean()
        ])

    vdf = pd.DataFrame(rows, columns=[
        "학교", "온도 변동률", "습도 변동률", "pH 변동률", "EC 변동률", "평균 생중량"
    ])

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

    for i, col in enumerate(["온도 변동률", "습도 변동률", "pH 변동률", "EC 변동률"]):
        r, c = divmod(i, 2)
        fig2.add_bar(x=vdf["학교"], y=vdf[col], row=r+1, col=c+1)
        fig2.add_scatter(
            x=vdf["학교"], y=vdf["평균 생중량"],
            mode="lines+markers", secondary_y=True,
            row=r+1, col=c+1
        )

    fig2.update_layout(font=PLOTLY_FONT, height=800)
    st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# Tab 3 : 결과 분석
# ==================================================
with tab3:
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["하늘고", "동산고", "아라고", "송도고"]
    )

    pos = {"하늘고": (1,1), "동산고": (1,2), "아라고": (2,1), "송도고": (2,2)}

    for school, (r,c) in pos.items():
        env_df = filter_by_period_safe(env_data[school], *PERIODS[school])
        gdf = growth_data[school]

        n = min(len(env_df), len(gdf))
        if n < 2:
            corr = [np.nan]*4
        else:
            corr = [
                np.corrcoef(env_df[k].iloc[:n], gdf["생중량(g)"].iloc[:n])[0,1]
                for k in ["temperature", "humidity", "ph", "ec"]
            ]

        fig3.add_bar(x=["온도","습도","pH","EC"], y=corr, row=r, col=c)

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
