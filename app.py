import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. تهيئة الصفحة وضبط الاتجاه بالكامل
st.set_page_config(page_title="محرك البحث الذكي", layout="centered")

# حقن كود CSS قوي يجبر المتصفح بالكامل على التحول لـ RTL
st.markdown(
    """
    <style>
    /* تطبيق اتجاه RTL على حاوية التطبيق بأكملها وجسم الصفحة */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        direction: RTL !important;
        text-align: right !important;
    }
    
    /* ضبط المدخلات، النصوص المرافقة لها، والعناصر النائبة */
    div[data-testid="stTextInput"] {
        direction: RTL !important;
        text-align: right !important;
    }
    
    input {
        direction: RTL !important;
        text-align: right !important;
    }
    
    /* ضبط محاذاة كافة العناوين والنصوص العادية */
    h1, h2, h3, h4, h5, h6, p, span, label {
        text-align: right !important;
        direction: RTL !important;
    }
    
    /* محاذاة الأزرار وتأثيراتها البصرية */
    div.stButton > button {
        display: block;
        margin-right: auto !important; /* لدفع الزر إلى جهة اليمين */
        margin-left: 0 !important;
        direction: RTL !important;
    }
    
    /* ضبط محاذاة صناديق التنبيهات (النجاح، التحذير، الخطأ) */
    [data-testid="stNotification"] {
        direction: RTL !important;
        text-align: right !important;
    }
    
    .stAlert {
        direction: RTL !important;
        text-align: right !important;
    }
    
    /* ضبط الأيقونات المرافقة للتنبيهات لتظهر على اليمين */
    [data-testid="stNotification"] > div {
        flex-direction: row-reverse !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==========================================
# ⚠️ ضع رابط ملف جوجل شيت الخاص بك هنا بأمان (مخفي تماماً عن المستخدمين)
# تأكد أن صلاحية الرابط هي "Anyone with the link can view"
# ==========================================
SECRET_GSHEET_URL = "https://docs.google.com/spreadsheets/d/1OHR6udxUs5XKAASLB2TqDzVLV0u5Qce5/edit?usp=sharing&ouid=105622729338359124172&rtpof=true&sd=true"

# 1. تحميل نموذج الذكاء الاصطناعي لفهم معاني اللغة العربية السياقية
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# 2. دالة لقراءة البيانات في الخلفية وتحويل الرابط إلى صيغة CSV
@st.cache_data(ttl=600)  # لتسريع البحث وحفظ البيانات مؤقتاً لمدة 10 دقائق
def load_data_from_gsheet(sheet_url):
    try:
        csv_url = sheet_url.split('/edit')[0] + '/gviz/tq?tqx=out:csv'
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"حدث خطأ أثناء الاتصال بالملف: {e}")
        return None

# واجهة تطبيق Streamlit النظيفة والخالية من الروابط
st.title("🔍 محرك البحث الذكي لأسئلة البوت العلمي")
st.write("اكتب سؤالك بالأسفل واضغط على الزر لمطابقته الفكرية والدلالية مع أسئلة قاعدة البيانات.")
st.markdown("---")

# جلب البيانات من الشيت في الخلفية (مخفي تماماً)
df = load_data_from_gsheet(SECRET_GSHEET_URL)

if df is not None:
    # تنظيف أسماء الأعمدة للتأكد من عدم وجود مسافات حول الكلمات
    df.columns = [col.strip() for col in df.columns]
    
    # أسماء الأعمدة المعتمدة والمطابقة لملفك
    col_question = "Question"
    col_answer = "Answer"
    col_category = "Category"
    
    # التحقق من وجود عمودي السؤال والجواب
    if col_question in df.columns and col_answer in df.columns:
        # إزالة أي أسطر فارغة في عمود الأسئلة
        df = df.dropna(subset=[col_question])
        
        # صندوق إدخال السؤال (الوحيد الذي يظهر للمستخدم في المتصفح)
        user_question = st.text_input("اكتب سؤالك هنا:", placeholder="مثال: هل لازم أنصح البنت لو خطبها شخص غير مناسب؟")
        
        # زر الإرسال والمقارنة
        search_button = st.button("إرسال السؤال ومقارنته 🚀")
        
        # تنفيذ عملية المقارنة عند الضغط على الزر
        if search_button and user_question:
            with st.spinner("جاري تحليل فكرة السؤال ومقارنتها..."):
                # تحويل الأسئلة في الملف وسؤال المستخدم إلى Embeddings تعبر عن المعنى
                questions_list = df[col_question].astype(str).tolist()
                questions_embeddings = model.encode(questions_list, show_progress_bar=False)
                user_embedding = model.encode([user_question])
                
                # حساب جيب تمام الزاوية (Cosine Similarity) للتشابه المعنوي
                similarities = cosine_similarity(user_embedding, questions_embeddings)[0]
                
                # العثور على المؤشر وأعلى نسبة تشابه
                max_sim_idx = similarities.argmax()
                highest_similarity_score = similarities[max_sim_idx] * 100
                
                st.markdown(f"### 📊  نسبة التطابق الفكري المحسوبة: ` {highest_similarity_score:.2f}` % ")
                st.markdown("---")
                
                # شرط الـ 70% للتحقق من وجود السؤال في قاعدة البيانات
                if highest_similarity_score >= 70:
                    matched_question = df.iloc[max_sim_idx][col_question]
                    matched_answer = df.iloc[max_sim_idx][col_answer]
                    matched_category = df.iloc[max_sim_idx][col_category] if col_category in df.columns else None
                    
                    # عرض صنف السؤال إذا كان متوفراً في السطر
                    if pd.notna(matched_category):
                        st.info(f"📁 **تصنيف السؤال :**  {matched_category}")
                        
                    st.subheader("📌 السؤال المطابق الذي تم العثور عليه:")
                    st.markdown(f"> {matched_question}")
                    
                    st.subheader("📝 الإجابة المطابقة المعتمدة:")
                    if pd.notna(matched_answer):
                        st.success(matched_answer)
                    else:
                        st.warning("عُثر على سؤال مطابق بنسبة عالية، ولكن خانة الإجابة المقابلة له فارغة في الملف.")
                else:
                    st.error("❌ الملف لا يحتوي على سؤال مطابق لهذا السؤال (نسبة التطابق أقل من 70%)")
                    
        elif search_button and not user_question:
            st.warning("⚠️ يرجى كتابة السؤال أولاً في الخانة المخصصة قبل الضغط على زر الإرسال.")
    else:
        st.error(f"خطأ: يرجى التأكد من أن ملفك يحتوي على الأعمدة التالية تماماً وبنفس الأحرف الإنجليزية: '{col_question}' و '{col_answer}'")