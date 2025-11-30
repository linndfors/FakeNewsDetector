import streamlit as st
import streamlit.components.v1 as components
import time
import pandas as pd
from src.preprocessor import clean_text
from src.inference import FakeNewsModel
from src.database import init_db, log_request, get_stats

st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="🕵️",
    layout="wide"
)

st.markdown("""
<style>
    .explanation-box {
        background-color: #f0f2f6;
        border-left: 5px solid #4B8BBE;
        padding: 15px;
        border-radius: 5px;
        color: #31333F;
        font-size: 16px;
    }
    .stAlert {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

init_db()

@st.cache_resource
def get_model_pipeline():
    return FakeNewsModel()

ai_engine = get_model_pipeline()

st.sidebar.title("📊 Статистика")
stats_data = get_stats()
if stats_data:
    df_stats = pd.DataFrame(stats_data, columns=["Label", "Count"])
    st.sidebar.bar_chart(df_stats.set_index("Label"))
else:
    st.sidebar.text("Даних поки немає")

st.title("🕵️ AI Fake News Detector")
st.markdown("Вставте текст новини, щоб перевірити його на достовірність та отримати пояснення.")

user_input = st.text_area("Текст новини", height=200, placeholder="Вставте сюди текст статті...")

if st.button("🔍 Аналізувати", use_container_width=True):
    if user_input:
        start_time = time.time()
        
        with st.spinner("⏳ ШІ аналізує семантику та контекст..."):
            cleaned_text = clean_text(user_input)
            
            label, confidence, probs = ai_engine.predict(cleaned_text)
            processing_time = (time.time() - start_time) * 1000
            
            log_request(cleaned_text, label, confidence, processing_time)

        st.divider()
        
        if label == "FAKE":
            header_color = "red"
            result_text = "⚠️ ОБЕРЕЖНО: Висока ймовірність фейку"
            icon = "🚨"
        else:
            header_color = "green"
            result_text = "✅ ДОВІРА: Текст виглядає правдивим"
            icon = "🛡️"

        st.markdown(f"<h2 style='text-align: center; color: {header_color};'>{icon} {result_text}</h2>", unsafe_allow_html=True)
        
        col_metrics1, col_metrics2 = st.columns(2)
        col_metrics1.metric("Впевненість моделі", f"{confidence:.2%}")
        col_metrics2.metric("Час аналізу", f"{processing_time:.0f} ms")
        
        st.progress(confidence, text="Рівень впевненості алгоритму")
    else:
        st.warning("Будь ласка, введіть текст для аналізу.")