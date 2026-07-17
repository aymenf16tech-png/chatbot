import streamlit as st
from sentence_transformers import util

from config import (
    SHEET_URL,
    HARD_THRESHOLD,
    CONFIDENT_THRESHOLD,
    TOP_K,
    CSS_FILE_PATH,
)
from utils import (
    convert_to_csv_url,
    fetch_data,
    load_ai_model,
    generate_embeddings,
    load_css,
)

# ============================================================
# 1) إعدادات الصفحة + حقن ملف CSS الخارجي
# ============================================================
st.set_page_config(page_title="المساعد الذكي", page_icon="🤖", layout="centered")
st.markdown(f"<style>{load_css(CSS_FILE_PATH)}</style>", unsafe_allow_html=True)

st.markdown("<div class='app-title'>🤖 المساعد الذكي للفريق العلمي</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>بحث دلالي فوري في قاعدة الأسئلة والأجوبة</div>", unsafe_allow_html=True)

# ============================================================
# 2) تجهيز البيانات والنموذج (مرة واحدة بفضل التخزين المؤقت)
# ============================================================
csv_url = convert_to_csv_url(SHEET_URL)
df = fetch_data(csv_url)

if df is None:
    st.stop()

model = load_ai_model()
if model is None:
    st.stop()

question_embeddings = generate_embeddings(model, df)
if question_embeddings is None:
    st.stop()

# تهيئة حالة الجلسة (Session State) لحفظ نتائج البحث وتجنب اختفائها عند التفاعل
if "search_results" not in st.session_state:
    st.session_state.search_results = None

# ============================================================
# 3) واجهة الإدخال (باستخدام st.form لتحسين الأداء)
# ============================================================
st.write("")

with st.form(key="search_form", clear_on_submit=False):
    user_query = st.text_area(
        "📝 اكتب السؤال هنا:",
        placeholder="مثال: كيف يمكنني تعديل الرمز السري؟",
        height=160,
    )
    submit_button = st.form_submit_button("  🔍 ابحث عن الإجابة  ")

st.write("---")

# ============================================================
# 4) معالجة البحث الدلالي وحفظ النتائج
# ============================================================
if submit_button:
    clean_query = user_query.strip()
    if not clean_query:
        st.warning("⚠️ الرجاء كتابة السؤال أولاً قبل الضغط على زر البحث.")
        st.session_state.search_results = None
    else:
        with st.spinner("جاري تحليل السؤال ومقارنته دلالياً..."):
            # ترميز السؤال وتطبيعه دلالياً
            query_embedding = model.encode(
                f"query: {clean_query}",
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            # حساب التشابه وجلب أفضل النتائج
            cosine_scores = util.cos_sim(query_embedding, question_embeddings)[0]
            k = min(TOP_K, len(df))
            top_results = cosine_scores.topk(k)
            
            # تخزين النتائج في الـ Session State لمنع اختفائها
            st.session_state.search_results = {
                "top_indices": [int(i) for i in top_results.indices],
                "top_scores": [round(float(s) * 100, 2) for s in top_results.values],
                "k": k
            }

# ============================================================
# 5) عرض النتائج المستمرة
# ============================================================
if st.session_state.search_results:
    results = st.session_state.search_results
    best_match_idx = results["top_indices"][0]
    similarity_percentage = results["top_scores"][0]
    k = results["k"]

    # الحالة الأولى: نسبة التشابه ضعيفة جداً
    if similarity_percentage < HARD_THRESHOLD:
        st.error("❌ لا يوجد سؤال مطابق أو قريب لهذا السؤال في قاعدة البيانات المتاحة.")
        st.info(
            f"💡 أقرب تطابق تم العثور عليه كانت نسبته **{similarity_percentage}%**، "
            f"وهي أقل من الحد الأدنى المطلوب ({HARD_THRESHOLD:.0f}%)."
        )

    # الحالة الثانية: تطابق مؤكد وعالي الدقة
    elif similarity_percentage >= CONFIDENT_THRESHOLD:
        matched_question = df.iloc[best_match_idx]["Question"]
        answer = df.iloc[best_match_idx]["Answer"]
        category = df.iloc[best_match_idx]["Category"]

        st.success("✅ تم العثور على سؤال مطابق بدقة عالية!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🎯 نسبة التشابه الدلالي", value=f"{similarity_percentage}%")
        with col2:
            st.metric(label="📁 تصنيف السؤال", value=category)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">🔍 السؤال الأقرب في الملف:</div>
                <div class="result-text">{matched_question}</div>
            </div>
            <div class="result-card">
                <div class="result-label">💡 الإجابة:</div>
                <div class="result-text">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # الحالة الثالثة: منطقة عدم اليقين (عرض عدة خيارات محتملة)
    else:
        st.warning(
            f"🤔 لم يتم العثور على تطابق مؤكد (أعلى نسبة {similarity_percentage}%). "
            f"إليك أقرب {k} أسئلة محتملة، اختر الأقرب لسؤالك:"
        )

        for rank, (idx, sc) in enumerate(zip(results["top_indices"], results["scores"]), start=1):
            if sc < HARD_THRESHOLD:
                continue
            q = df.iloc[idx]["Question"]
            a = df.iloc[idx]["Answer"]
            c = df.iloc[idx]["Category"]
            
            # الآن لن تختفي النتائج عند فتح الـ expander
            with st.expander(f"{rank}. {q}   —   نسبة التشابه: {sc}%"):
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">📁 التصنيف: {c}</div>
                        <div class="result-label">💡 الإجابة:</div>
                        <div class="result-text">{a}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    if not submit_button:
        st.info("💡 اكتب سؤالك في الصندوق أعلاه ثم اضغط **'ابحث عن الإجابة'** لبدء البحث.")