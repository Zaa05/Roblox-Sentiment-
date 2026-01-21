import streamlit as st
import joblib
import re
import nltk
import numpy as np

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ===============================
# FIX NLTK DOWNLOAD
# ===============================
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("bernoulli_nb.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Cek model info
st.sidebar.markdown("### ℹ️ Info Model")
st.sidebar.write(f"Model type: {type(model).__name__}")
st.sidebar.write(f"Classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")

# ===============================
# PREPROCESSING
# ===============================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cache stopwords to avoid repeated calls
@st.cache_data
def get_stopwords():
    return set(stopwords.words('indonesian'))

stop_words = get_stopwords()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

label_map = {
    0: "Negatif 😡",
    1: "Netral 😐",
    2: "Positif 😊"
}

# ===============================
# DIAGNOSTIC FUNCTIONS
# ===============================
def analyze_prediction(text):
    """Analisis lengkap prediksi"""
    clean_text = preprocess(text)
    vector = tfidf.transform([clean_text])
    prediction = model.predict(vector)[0]
    
    analysis = {
        'original': text,
        'cleaned': clean_text,
        'prediction': prediction,
        'label': label_map[prediction],
        'tokens': clean_text.split()
    }
    
    # Cek probabilitas jika ada
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector)[0]
        analysis['probabilities'] = {
            'negatif': proba[0],
            'netral': proba[1],
            'positif': proba[2]
        }
    
    # Cek apakah tokens ada di vocabulary
    vocab_status = {}
    for token in analysis['tokens']:
        vocab_status[token] = token in tfidf.vocabulary_
    analysis['vocab_status'] = vocab_status
    
    return analysis

# ===============================
# UI (FRONT END)
# ===============================
st.title("📊 Analisis Sentimen Roblox Indonesia")
st.write("Model: **Bernoulli Naive Bayes + TF-IDF**")

# Sidebar diagnostics
with st.sidebar:
    st.markdown("### 🔍 Diagnosa")
    
    if st.button("Test Model Sederhana"):
        test_cases = [
            ("jelek", "Harusnya Negatif (0)"),
            ("buruk", "Harusnya Negatif (0)"),
            ("payah", "Harusnya Negatif (0)"),
            ("biasa", "Harusnya Netral (1)"),
            ("bagus", "Harusnya Positif (2)"),
            ("baik", "Harusnya Positif (2)")
        ]
        
        for text, expected in test_cases:
            analysis = analyze_prediction(text)
            st.write(f"**'{text}'**")
            st.write(f"  → Prediksi: {analysis['prediction']}")
            st.write(f"  → Expected: {expected}")
            st.write(f"  → Cleaned: {analysis['cleaned']}")
            st.write("---")

text_input = st.text_area(
    "Masukkan komentar Platform X:",
    placeholder="Contoh: Game roblox makin seru setelah update terbaru",
    height=120
)

if st.button("Prediksi Sentimen", type="primary"):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        try:
            # Analisis lengkap
            analysis = analyze_prediction(text_input)
            
            # Tampilkan hasil
            st.markdown("---")
            st.subheader("🔮 Hasil Analisis")
            
            # Prediction dengan warna
            if analysis['prediction'] == 0:
                st.error(f"### {analysis['label']}")
            elif analysis['prediction'] == 1:
                st.warning(f"### {analysis['label']}")
            else:
                st.success(f"### {analysis['label']}")
            
            # Detail analysis
            with st.expander("📊 Detail Analisis Lengkap"):
                st.write(f"**Teks asli:** {analysis['original']}")
                st.write(f"**Teks diproses:** {analysis['cleaned']}")
                st.write(f"**Tokens:** {analysis['tokens']}")
                
                st.write("**Vocabulary Check:**")
                for token, in_vocab in analysis['vocab_status'].items():
                    status = "✅ ADA" if in_vocab else "❌ TIDAK ADA"
                    st.write(f"  - {token}: {status}")
                
                if 'probabilities' in analysis:
                    st.write("**Probabilitas:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Negatif", f"{analysis['probabilities']['negatif']:.1%}")
                    with col2:
                        st.metric("Netral", f"{analysis['probabilities']['netral']:.1%}")
                    with col3:
                        st.metric("Positif", f"{analysis['probabilities']['positif']:.1%}")
            
            # Warning jika banyak tokens tidak ada di vocabulary
            missing_tokens = [t for t, status in analysis['vocab_status'].items() if not status]
            if missing_tokens:
                st.warning(f"⚠️ {len(missing_tokens)} token tidak ada dalam vocabulary model: {missing_tokens}")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
