import re
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, REQUIRED_COLUMNS


def convert_to_csv_url(sheet_url: str) -> str:
    """
    يحوّل أي رابط جوجل شيت إلى رابط تصدير CSV صحيح ديناميكياً
    """
    id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not id_match:
        st.error("❌ تعذّر استخراج معرّف الشيت من الرابط. تأكد من صحة الرابط.")
        st.stop()
    sheet_id = id_match.group(1)

    gid_match = re.search(r"[#&?]gid=(\d+)", sheet_url)
    gid = gid_match.group(1) if gid_match else "0"

    # تعديل المسار ليعود بصيغة التصدير الفعلية (export) بناءً على المتغيرات المستخرجة
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data(show_spinner="جاري الاتصال بقاعدة البيانات...")
def fetch_data(url: str):
    """يجلب البيانات من جوجل شيت وينظّفها (فراغات، صفوف فارغة، تكرارات)."""
    try:
        data_frame = pd.read_csv(url, encoding="utf-8")

        for col in REQUIRED_COLUMNS:
            if col in data_frame.columns:
                data_frame[col] = data_frame[col].astype(str).str.strip()

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in data_frame.columns]
        if missing_cols:
            st.error(f"❌ الأعمدة التالية مفقودة في السطر الأول من الشيت: {missing_cols}")
            st.write("الأعمدة المتوفرة حالياً:", list(data_frame.columns))
            return None

        data_frame = data_frame.dropna(subset=["Question"])
        data_frame = data_frame[data_frame["Question"].str.len() > 0]
        data_frame = data_frame.drop_duplicates(subset=["Question"]).reset_index(drop=True)

        if data_frame.empty:
            st.error("❌ جدول البيانات فارغ بعد التنظيف. تأكد من وجود بيانات صالحة في الشيت.")
            return None

        return data_frame
    except Exception as error:
        st.error(f"❌ فشل الاتصال بجوجل شيت. تأكد أن الرابط متاح لـ 'أي شخص لديه الرابط': {error}")
        return None


@st.cache_resource(show_spinner="جاري تحميل نموذج الذكاء الاصطناعي لأول مرة (قد يستغرق دقيقة)...")
def load_ai_model():
    """يحمّل نموذج التضمين الدلالي متعدد اللغات."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as error:
        st.error(f"❌ فشل تحميل نموذج الذكاء الاصطناعي. تأكد من اتصال جهازك بالإنترنت: {error}")
        return None


@st.cache_data(show_spinner="جاري التجهيز...")
def generate_embeddings(_model, _df):
    """يولّد متجهات دلالية مطبّعة (L2) لكل الأسئلة المخزنة، وفق بروتوكول E5 (بادئة passage:)."""
    try:
        questions_list = [f"passage: {q}" for q in _df["Question"].tolist()]
        embeddings = _model.encode(
            questions_list,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings
    except Exception as error:
        st.error(f"❌ حدث خطأ أثناء معالجة نصوص الأسئلة: {error}")
        return None


def load_css(css_path: str) -> str:
    """يقرأ ملف CSS خارجي كنص خام ليتم حقنه في الصفحة."""
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()
