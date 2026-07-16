import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(page_title="المساعد الذكي", page_icon="🤖", layout="centered")

# --- 2. حقن كود CSS لإجبار المتصفح على التنسيق من اليمين إلى اليسار (RTL) ---
st.markdown(
    """
    <style>
    /* جعل اتجاه الصفحة بالكامل من اليمين إلى اليسار */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        direction: RTL;
        text-align: right;
    }
    /* تنسيق حقول الإدخال لتكتب من اليمين */
    input, textarea, .stTextArea textarea {
        direction: RTL !important;
        text-align: right !important;
    }
    /* محاذاة العناوين والنصوص */
    h1, h2, h3, h4, h5, h6, p, span, label {
        text-align: right !important;
        direction: RTL !important;
    }
    /* ضبط اتجاه أيقونات التنبيهات */
    [data-testid="stNotification"] {
        direction: RTL;
        text-align: right;
    }
    /* ضبط اتجاه الصناديق والمقاييس */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        text-align: right !important;
        direction: RTL !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h2 style='text-align: center;'>🤖 مساعد الفريق العلمي الدلالي</h2>", unsafe_allow_html=True)

# --- 3. رابط الاتصال بجوجل شيت ---
# 🔗 استبدل الرابط أدناه برابط ملفك الحقيقي (تأكد أن الرابط متاح للعرض العام)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1OHR6udxUs5XKAASLB2TqDzVLV0u5Qce5/edit?usp=sharing&ouid=105622729338359124172&rtpof=true&sd=true"

# تحويل الرابط تلقائيًا إلى صيغة تصدير CSV
try:
    csv_url = SHEET_URL.replace('/edit?usp=sharing', '/export?format=csv')
    csv_url = csv_url.replace('/edit#gid=', '/export?format=csv&gid=')
except Exception as e:
    st.error(f"خطأ في صيغة الرابط المكتوب: {e}")
    st.stop()


# --- 4. جلب البيانات من جوجل شيت وتجهيزها ---
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
    # التحقق من وجود الأعمدة المطلوبة
    required_columns = ['Question', 'Answer', 'Category']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ الأعمدة التالية مفقودة في السطر الأول من الشيت: {missing_cols}")
        st.write("الأعمدة المتوفرة حالياً في ملفك هي:", list(df.columns))
        st.stop()
else:
    st.stop()


# --- 5. تحميل نموذج الذكاء الاصطناعي E5 ---
@st.cache_resource(show_spinner="جاري تحميل نموذج الذكاء الاصطناعي لأول مرة (قد يستغرق دقيقة)...")
def load_ai_model():
    try:
        # استدعاء النموذج الفائق والمناسب لـ Streamlit
        model = SentenceTransformer('intfloat/multilingual-e5-small')
        return model
    except Exception as error:
        st.error(f"❌ فشل تحميل نموذج الذكاء الاصطناعي. تأكد من اتصال جهازك بالإنترنت: {error}")
        return None

model = load_ai_model()

if model is None:
    st.stop()


# --- 6. توليد المتجهات الدلالية ---
@st.cache_data(show_spinner="جاري تجهيز متجهات الأسئلة المخزنة...")
def generate_embeddings(_model, _df):
    try:
        # نموذج E5 يتطلب إضافة "passage: " قبل النصوص المخزنة لرفع دقة البحث
        questions_list = [f"passage: {q}" for q in _df['Question'].tolist()]
        embeddings = _model.encode(questions_list, convert_to_tensor=True)
        return embeddings
    except Exception as error:
        st.error(f"❌ حدث خطأ أثناء معالجة نصوص الأسئلة: {error}")
        return None

question_embeddings = generate_embeddings(model, df)

if question_embeddings is None:
    st.stop()


# --- 7. عناصر واجهة المستخدم في منتصف الصفحة تماماً ---
st.write("---")

user_query = st.text_area(
    "📝 اكتب سؤالك هنا:", 
    placeholder="مثال: كيف يمكنني تعديل الرمز السري؟",
    height=100
)

submit_button = st.button("🔍 إرسال السؤال والمقارنة", use_container_width=True)


# --- 8. معالجة البحث وعرض النتائج ---
st.write("---")

if submit_button:
    if not user_query.strip():
        st.warning("⚠️ الرجاء كتابة سؤال أولاً قبل الضغط على زر الإرسال.")
    else:
        with st.spinner("جاري مقارنة المعنى الدلالي لسؤالك..."):
            # نموذج E5 يتطلب إضافة "query: " قبل سؤال المستخدم الجديد
            query_embedding = model.encode(f"query: {user_query}", convert_to_tensor=True)
            
            # حساب تشابه جيب التمام
            cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]
            
            best_match_idx = cosine_scores.argmax().item()
            score = cosine_scores[best_match_idx].item()
            similarity_percentage = round(score * 100, 2)
            
            matched_question = df.iloc[best_match_idx]['Question']
            answer = df.iloc[best_match_idx]['Answer']
            category = df.iloc[best_match_idx]['Category']

        # 🛑 تطبيق شرط الحماية: حد أدنى للتطابق 70%
        if similarity_percentage < 70.0:
            st.error("❌ عذراً، لم أجد سؤالاً قريباً من طرحك في قاعدة البيانات المتاحة.")
            st.info(f"💡 أقرب تطابق وجدته كان بنسبة **{similarity_percentage}%** فقط، وهو غير كافٍ لضمان دقة الإجابة.")
        else:
            # عرض النتائج في حال تجاوز النسبة المطلوبة بنجاح
            st.success("✅ تم العثور على أقرب تطابق بنجاح!")
            
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
    st.info("💡 الرجاء كتابة سؤالك في الصندوق أعلاه ثم الضغط على زر **'إرسال السؤال والمقارنة'** لبدء البحث.")