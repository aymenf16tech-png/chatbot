import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# إعداد الصفحة أول شيء في السكربت
st.set_page_config(page_title="المساعد العلمي", page_icon="🤖", layout="centered")

st.title("🤖 مساعد الأسئلة والأجوبة الدلالي")
st.write("اكتب سؤالك في القائمة الجانبية، وسيقوم البوت بالبحث في قاعدة البيانات الدلالية عن أقرب إجابة.")

# --- سطر الاتصال بجوجل شيت ---
# 🔗 استبدل الرابط أدناه برابط ملفك الحقيقي
SHEET_URL = "https://docs.google.com/spreadsheets/d/1OHR6udxUs5XKAASLB2TqDzVLV0u5Qce5/edit?usp=sharing&ouid=105622729338359124172&rtpof=true&sd=true"

# خطوة تتبع 1: تحويل الرابط لـ CSV
try:
    csv_url = SHEET_URL.replace('/edit?usp=sharing', '/export?format=csv')
    csv_url = csv_url.replace('/edit#gid=', '/export?format=csv&gid=')
except Exception as e:
    st.error(f"خطأ في صيغة الرابط المكتوب: {e}")
    st.stop()

# خطوة تتبع 2: جلب البيانات من جوجل شيت
@st.cache_data(show_spinner="جاري الاتصال بجوجل شيت وجلب جدول البيانات...")
def fetch_data(url):
    try:
        data_frame = pd.read_csv(url, encoding="utf-8")
        return data_frame
    except Exception as error:
        st.error(f"❌ فشل الاتصال بجوجل شيت. تأكد من أن الرابط متاح لـ 'أي شخص لديه الرابط': {error}")
        return None

df = fetch_data(csv_url)

if df is not None:
    # التحقق من الأعمدة
    required_columns = ['Question', 'Answer', 'Category']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ الأعمدة التالية مفقودة في السطر الأول من الشيت: {missing_cols}")
        st.write("الأعمدة المتوفرة حالياً في ملفك هي:", list(df.columns))
        st.stop()
else:
    st.stop()

# خطوة تتبع 3: تحميل نموذج الذكاء الاصطناعي
@st.cache_resource(show_spinner="جاري تحميل نموذج الذكاء الاصطناعي لأول مرة (قد يستغرق دقيقة)...")
def load_ai_model():
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as error:
        st.error(f"❌ فشل تحميل نموذج الذكاء الاصطناعي. تأكد من اتصال جهازك بالإنترنت: {error}")
        return None

model = load_ai_model()

if model is None:
    st.stop()

# خطوة تتبع 4: توليد المتجهات
@st.cache_data(show_spinner="جاري تجهيز متجهات الأسئلة المخزنة...")
def generate_embeddings(_model, _df):
    try:
        questions_list = _df['Question'].tolist()
        embeddings = _model.encode(questions_list, convert_to_tensor=True)
        return embeddings
    except Exception as error:
        st.error(f"❌ حدث خطأ أثناء معالجة نصوص الأسئلة: {error}")
        return None

question_embeddings = generate_embeddings(model, df)

if question_embeddings is None:
    st.stop()


# --- 🎯 بناء عناصر واجهة المستخدم في القائمة الجانبية (Sidebar) ---
st.sidebar.header("📥 إدخال البيانات")

# صندوق النص في القائمة الجانبية
user_query = st.sidebar.text_area(
    "📝 اكتب سؤالك هنا:", 
    placeholder="مثال: كيف يمكنني تعديل الرمز السري؟",
    height=120
)

# زر الإرسال في القائمة الجانبية
submit_button = st.sidebar.button("🔍 إرسال السؤال والمقارنة", use_container_width=True)


# --- معالجة الضغط وعرض النتائج في المنتصف ---
st.divider()

if submit_button:
    if not user_query.strip():
        st.warning("⚠️ الرجاء كتابة سؤال أولاً في القائمة الجانبية قبل الضغط على زر الإرسال.")
    else:
        with st.spinner("جاري مقارنة المعنى الدلالي لسؤالك..."):
            query_embedding = model.encode(user_query, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]
            
            best_match_idx = cosine_scores.argmax().item()
            score = cosine_scores[best_match_idx].item()
            similarity_percentage = round(score * 100, 2)
            
            matched_question = df.iloc[best_match_idx]['Question']
            answer = df.iloc[best_match_idx]['Answer']
            category = df.iloc[best_match_idx]['Category']

        # 🛑 فحص شرط نسبة التطابق (70%)
        if similarity_percentage < 70.0:
            st.error("❌ عذراً، لم أجد سؤالاً قريباً من طرحك في قاعدة البيانات المتاحة.")
            st.info(f"💡 أقرب تطابق وجدته كان بنسبة **{similarity_percentage}%** فقط، وهو غير كافٍ لضمان دقة الإجابة.")
        else:
            # عرض النتائج إذا تجاوزت نسبة 70% بنجاح
            st.success("✅ تم العثور على أقرب تطابق بنجاح!")
            
            # عرض البيانات الأساسية
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 نسبة التشابه الدلالي", value=f"{similarity_percentage}%")
            with col2:
                st.metric(label="📁 تصنيف السؤال", value=category)
                
            st.markdown("### 🔍 السؤال الأقرب في الملف:")
            st.info(matched_question)
            
            st.markdown("### 💡 الإجابة المستحضرة:")
            st.write(answer)
else:
    st.info("👈 الرجاء كتابة سؤالك في القائمة الجانبية على اليسار/اليمين ثم الضغط على زر **'إرسال السؤال والمقارنة'** لبدء البحث.")