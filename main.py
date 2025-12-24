import streamlit as st
import pandas as pd
from pathlib import Path
import unicodedata
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------
# 기본 설정
# -------------------------------
st.set_page_config(
    page_title="극지식물 최적 EC 농도 연구",
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

# -------------------------------
# 유틸: 한글 파일명 안전 비교
# -------------------------------
def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", name)

# -------------------------------
# 데이터 로딩
# -------------------------------
@st.cache_data
def load_environment_data():
    data_dir = Path("data")
    env_data = {}

    if not data_dir.exists():
        return env_data

    for file in data_dir.iterdir():
        if file.suffix.lower() == ".csv":
            norm_name = normalize_name(file.name)
            try:
                df = pd.read_csv(file)
                school = norm_name.replace("_환경데이터.csv", "")
                env_data[school] = df
            except Exception:
                continue

    return env_data


@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    growth_file = None

    for file in data_dir.iterdir():
        if file.suffix.lower() == ".xlsx":
            if "생육결과데이터" in normalize_name(file.name):
                growth_file = file
                break

    if growth_file is None:
        return {}

    xls = pd.ExcelFile(growth_file, engine="openpyxl")
    growth_data = {}

    for sheet in xls.sheet_names:
        try:
            growth_data[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            continue

    return growth_data


with st.spinner("데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

if not env_data or not growth_data:
    st.error("데이터 파일을 찾을 수 없습니다. data 폴더 구조를 확인하세요.")
    st.stop()

# -------------------------------
# 메타 정보
# -------------------------------
EC_INFO = {
    "송도고": 1.0,
    "하늘고": 2.0,
    "아라고": 4.0,
    "동산고": 8.0
}

COLOR_MAP = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728"
}

# -------------------------------
# 사이드바
# -------------------------------
schools = ["전체"] + list(EC_INFO.keys())
selected_school = st.sidebar.selectbox("학교 선택", schools)

# -------------------------------
# 제목
# -------------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# =========================================================
# Tab 1 : 실험 개요
# =========================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.markdown(
        """
        본 연구는 **극지식물의 생육에 영향을 미치는 EC(Electrical Conductivity) 농도**를
        학교별로 다르게 설정하여,
        **생육 결과를 비교 분석하고 최적 EC 농도를 도출**하는 것을 목표로 한다.
        """
    )

    summary_rows = []
    total_plants = 0
    for school, ec in EC_INFO.items():
        count = len(growth_data.get(school, []))
        total_plants += count
        summary_rows.append([school, ec, count])

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["학교명", "EC 목표", "개체수"]
    )

    st.dataframe(summary_df, use_container_width=True)

    avg_temp = pd.concat(env_data.values())["temperature"].mean()
    avg_hum = pd.concat(env_data.values())["humidity"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 개체수", total_plants)
    col2.metric("평균 온도 (°C)", f"{avg_temp:.1f}")
    col3.metric("평균 습도 (%)", f"{avg_hum:.1f}")
    col4.metric("최적 EC", "2.0 (하늘고)")

# =========================================================
# Tab 2 : 환경 데이터
# =========================================================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    avg_data = []
    for school, df in env_data.items():
        avg_data.append([
            school,
            df["temperature"].mean(),
            df["humidity"].mean(),
            df["ph"].mean(),
            df["ec"].mean()
        ])

    avg_df = pd.DataFrame(
        avg_data,
        columns=["학교", "온도", "습도", "pH", "EC"]
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "평균 온도", "평균 습도",
            "평균 pH", "목표 EC vs 실측 EC"
        ]
    )

    fig.add_bar(x=avg_df["학교"], y=avg_df["온도"], row=1, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["습도"], row=1, col=2)
    fig.add_bar(x=avg_df["학교"], y=avg_df["pH"], row=2, col=1)

    fig.add_bar(
        x=list(EC_INFO.keys()),
        y=list(EC_INFO.values()),
        name="목표 EC",
        row=2, col=2
    )
    fig.add_bar(
        x=avg_df["학교"],
        y=avg_df["EC"],
        name="실측 EC",
        row=2, col=2
    )

    fig.update_layout(font=PLOTLY_FONT, height=700)
    st.plotly_chart(fig, use_container_width=True)

    if selected_school != "전체":
        df = env_data[selected_school]
        fig_ts = px.line(
            df,
            x="time",
            y=["temperature", "humidity", "ec"],
            title=f"{selected_school} 환경 시계열"
        )
        fig_ts.add_hline(y=EC_INFO[selected_school], line_dash="dash")
        fig_ts.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("환경 데이터 원본"):
        if selected_school == "전체":
            for school, df in env_data.items():
                st.write(school)
                st.dataframe(df)
        else:
            st.dataframe(env_data[selected_school])

        buffer = io.BytesIO()
        pd.concat(env_data.values()).to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            "CSV 다운로드",
            data=buffer,
            file_name="환경데이터_전체.csv",
            mime="text/csv"
        )

# =========================================================
# Tab 3 : 생육 결과
# =========================================================
with tab3:
    st.subheader("EC별 생육 결과 비교")

    growth_summary = []
    for school, df in growth_data.items():
        growth_summary.append([
            school,
            EC_INFO.get(school),
            df["생중량(g)"].mean(),
            df["잎 수(장)"].mean(),
            df["지상부 길이(mm)"].mean(),
            len(df)
        ])

    gdf = pd.DataFrame(
        growth_summary,
        columns=["학교", "EC", "생중량", "잎수", "지상부길이", "개체수"]
    )

    best = gdf.loc[gdf["생중량"].idxmax()]

    st.metric("🥇 최고 평균 생중량 EC", f"{best['EC']} (하늘고)")

    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=["생중량", "잎 수", "지상부 길이", "개체수"]
    )

    fig2.add_bar(x=gdf["EC"], y=gdf["생중량"], row=1, col=1)
    fig2.add_bar(x=gdf["EC"], y=gdf["잎수"], row=1, col=2)
    fig2.add_bar(x=gdf["EC"], y=gdf["지상부길이"], row=2, col=1)
    fig2.add_bar(x=gdf["EC"], y=gdf["개체수"], row=2, col=2)

    fig2.update_layout(font=PLOTLY_FONT, height=700)
    st.plotly_chart(fig2, use_container_width=True)

    all_growth = pd.concat(growth_data, names=["학교"]).reset_index(level=0)
    fig_box = px.box(
        all_growth,
        x="학교",
        y="생중량(g)",
        color="학교"
    )
    fig_box.update_layout(font=PLOTLY_FONT)
    st.plotly_chart(fig_box, use_container_width=True)

    fig_scatter1 = px.scatter(
        all_growth,
        x="잎 수(장)",
        y="생중량(g)",
        color="학교"
    )
    fig_scatter2 = px.scatter(
        all_growth,
        x="지상부 길이(mm)",
        y="생중량(g)",
        color="학교"
    )
    fig_scatter1.update_layout(font=PLOTLY_FONT)
    fig_scatter2.update_layout(font=PLOTLY_FONT)

    st.plotly_chart(fig_scatter1, use_container_width=True)
    st.plotly_chart(fig_scatter2, use_container_width=True)

    with st.expander("생육 데이터 원본"):
        for school, df in growth_data.items():
            st.write(school)
            st.dataframe(df)

        buffer = io.BytesIO()
        all_growth.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button(
            "XLSX 다운로드",
            data=buffer,
            file_name="생육결과_전체.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
