import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. إعدادات الصفحة وعنوان الواجهة ---
st.set_page_config(page_title="المساعد العلمي ", page_icon="🤖", layout="centered")
st.markdown("<h2 style='text-align: center;'>🤖 مساعد الأسئلة والأجوبة الدلالي</h2>", unsafe_allow_html=True)
st.write("اطرح أي سؤال وسيقوم البوت بالبحث عن أقرب سؤال مخزن وإعطائك تفاصيل الإجابة ونسبة التطابق.")

# --- 2. تحميل البيانات من جوجل شيت (مع التخزين المؤقت لتسريع الأداء) ---
@st.cache_resource
def load_resources():
    try:
        # تحميل نموذج الذكاء الاصطناعي
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # 🔗 ضع رابط جوجل شيت الخاص بك هنا بين القوسين
        sheet_url = "https://docs.google.com/spreadsheets/d/1OHR6udxUs5XKAASLB2TqDzVLV0u5Qce5/edit?usp=sharing&ouid=105622729338359124172&rtpof=true&sd=true"
        
        # تحويل الرابط تلقائياً ليعمل كملف CSV قابل للتنزيل والقراءة برمجياً
        csv_url = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
        csv_url = csv_url.replace('/edit#gid=', '/export?format=csv&gid=') # لدعم الصفحات الفرعية إن وجدت
        
        # قراءة البيانات مباشرة من الإنترنت
        df = pd.read_csv(csv_url, encoding="utf-8")
        
        # التحقق من وجود الأعمدة المطلوبة بنفس حالة الأحرف بالضبط
        required_columns = ['Question', 'Answer', 'Category']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"❌ خطأ: العمود '{col}' غير موجود في جوجل شيت. الأعمدة الحالية هي: {list(df.columns)}")
                return None, None, None
                
        # توليد المتجهات للأسئلة المخزنة
        stored_embeddings = model.encode(df['Question'].tolist(), convert_to_tensor=True)
        return model, df, stored_embeddings
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء الاتصال بجوجل شيت وقراءة البيانات: {e}")
        return None, None, None

# استدعاء الموارد المشحونة
model, df, question_embeddings = load_resources()

# إيقاف السكربت إذا لم يتم تحميل الموارد بنجاح لتجنب أخطاء التعريف (is not defined)
if model is None or df is None or question_embeddings is None:
    st.warning("⚠️ يرجى التأكد من صلاحيات رابط جوجل شيت وأسماء الأعمدة لتشغيل البوت.")
    st.stop()

# --- 3. بناء واجهة المستخدم ---
user_query = st.text_input("📝 اكتب سؤالك هنا:", placeholder="مثال: أريد تعديل الرمز السري للحساب...")

if user_query:
    with st.spinner("جاري البحث في قاعدة البيانات الدلالية..."):
        # تحويل سؤال المستخدم لمتجه
        query_embedding = model.encode(user_query, convert_to_tensor=True)
        
        # حساب تشابه جيب التمام
        cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]
        
        # تحديد النتيجة الفضلى
        best_match_idx = cosine_scores.argmax().item()
        score = cosine_scores[best_match_idx].item()
        
        # استخراج البيانات المقابلة
        matched_question = df.iloc[best_match_idx]['Question']
        answer = df.iloc[best_match_idx]['Answer']
        category = df.iloc[best_match_idx]['Category']
        similarity_percentage = round(score * 100, 2)
        
    # --- 4. عرض النتائج بشكل منظم ---
    st.success("تم العثور على أقرب تطابق!")
    
    # استخدام نظام الأعمدة لعرض التصنيف ودرجة القرب بشكل جذاب
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🎯 درجة القرب والتطابق", value=f"{similarity_percentage}%")
    with col2:
        st.metric(label="📁 تصنيف السؤال", value=category)
        
    # عرض السؤال المخزن والإجابة داخل صناديق مميزة
    st.markdown("### 🔍 السؤال الأقرب في الملف:")
    st.info(matched_question)
    
    st.markdown("### 💡 الإجابة المستحضرة:")
    st.write(answer)

    # تنبيه إضافي إذا كانت نسبة التطابق منخفضة
    if similarity_percentage < 50.0:
        st.warning("⚠️ ملاحظة: نسبة التطابق منخفضة نسبيًا، قد لا يكون هذا هو السؤال الدقيق الذي تبحث عنه.")