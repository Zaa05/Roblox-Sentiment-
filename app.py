import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pandas as pd
import io
from io import BytesIO

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
model = joblib.load("mnb_model.pkl")
tfidf = joblib.load("mnb_tfidf.pkl")

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
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# LABEL MAPPING untuk 2 KELAS (TANPA NETRAL)
label_map = {
    0: "Negatif 😡",
    1: "Positif 😊"
}

# ===============================
# FUNGSI UNTUK PREDIKSI BATCH
# ===============================
def predict_batch(texts):
    """
    Prediksi sentimen untuk batch teks (2 kelas)
    """
    results = []
    
    for text in texts:
        try:
            # Preprocess teks
            clean_text = preprocess(text)
            
            # Transform ke features
            vector = tfidf.transform([clean_text])
            
            # Predict
            prediction = model.predict(vector)[0]
            
            # Get probabilities if available (2 kelas)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vector)[0]
                probabilities = {
                    'negatif': proba[0] if len(proba) > 0 else 0,
                    'positif': proba[1] if len(proba) > 1 else 0
                }
            
            results.append({
                'original_text': text,
                'cleaned_text': clean_text,
                'prediction': prediction,
                'sentiment_label': label_map.get(prediction, "Tidak Diketahui"),
                'probabilities': probabilities
            })
        except Exception as e:
            results.append({
                'original_text': text,
                'cleaned_text': 'ERROR',
                'prediction': -1,
                'sentiment_label': 'ERROR',
                'probabilities': None,
                'error': str(e)
            })
    
    return results

def process_uploaded_file(uploaded_file):
    """
    Proses file yang diupload (Excel atau CSV)
    """
    try:
        # Cek tipe file
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            return None, "Format file tidak didukung. Gunakan Excel (.xlsx, .xls) atau CSV (.csv)"
        
        # Cari kolom yang berisi komentar
        comment_columns = ['komentar', 'comment', 'text', 'teks', 'review', 'ulasan', 'tweet', 'post']
        
        found_column = None
        for col in df.columns:
            col_lower = col.lower()
            for keyword in comment_columns:
                if keyword in col_lower:
                    found_column = col
                    break
            if found_column:
                break
        
        # Jika tidak ditemukan, gunakan kolom pertama
        if not found_column:
            found_column = df.columns[0]
            st.warning(f"Kolom komentar tidak ditemukan. Menggunakan kolom pertama: '{found_column}'")
        
        return df, found_column
        
    except Exception as e:
        return None, f"Error membaca file: {str(e)}"

# ===============================
# DIAGNOSTIC FUNCTIONS
# ===============================
def analyze_prediction(text):
    """Analisis lengkap prediksi untuk 2 kelas"""
    clean_text = preprocess(text)
    vector = tfidf.transform([clean_text])
    prediction = model.predict(vector)[0]
    
    analysis = {
        'original': text,
        'cleaned': clean_text,
        'prediction': prediction,
        'label': label_map.get(prediction, "Tidak Diketahui"),
        'tokens': clean_text.split()
    }
    
    # Cek probabilitas jika ada (2 kelas)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector)[0]
        analysis['probabilities'] = {
            'negatif': proba[0] if len(proba) > 0 else 0,
            'positif': proba[1] if len(proba) > 1 else 0
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
st.write("Model: **Multinomial Naive Bayes + TF-IDF** (2 Kelas: Negatif & Positif)")

# Buat tab untuk berbagai fungsi
tab1, tab2 = st.tabs(["🔍 Analisis Tunggal", "📁 Analisis File"])

with tab1:
    # TAB 1: ANALISIS TUNGGAL
    text_input = st.text_area(
        "Masukkan komentar Platform X:",
        placeholder="Contoh: Game roblox makin seru setelah update terbaru",
        height=120,
        key="single_text"
    )
    
    if st.button("Prediksi Sentimen", type="primary", key="predict_single"):
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
                    st.success(f"### {analysis['label']}")
                else:
                    st.warning(f"### {analysis['label']}")
                
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
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Negatif", f"{analysis['probabilities']['negatif']:.1%}")
                        with col2:
                            st.metric("Positif", f"{analysis['probabilities']['positif']:.1%}")
                
                # Warning jika banyak tokens tidak ada di vocabulary
                missing_tokens = [t for t, status in analysis['vocab_status'].items() if not status]
                if missing_tokens:
                    st.warning(f"⚠️ {len(missing_tokens)} token tidak ada dalam vocabulary model: {missing_tokens}")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

with tab2:
    # TAB 2: ANALISIS FILE
    st.subheader("📁 Upload File untuk Analisis Batch")
    
    # Pilihan upload file
    uploaded_file = st.file_uploader(
        "Upload file Excel atau CSV",
        type=['xlsx', 'xls', 'csv'],
        help="File harus memiliki kolom berisi komentar (kolom akan otomatis dideteksi)"
    )
    
    if uploaded_file is not None:
        # Proses file
        df, comment_column = process_uploaded_file(uploaded_file)
        
        if df is not None:
            st.success(f"✅ File berhasil dibaca. Kolom komentar: **'{comment_column}'**")
            
            # Tampilkan preview data
            with st.expander("👁️ Preview Data"):
                st.dataframe(df.head(10), use_container_width=True)
                st.write(f"Jumlah data: {len(df)} baris")
                st.write(f"Kolom: {list(df.columns)}")
            
            # Pilih jumlah data untuk dianalisis
            st.subheader("⚙️ Konfigurasi Analisis")
            col1, col2 = st.columns(2)
            
            with col1:
                analyze_all = st.checkbox("Analisis semua data", value=True)
            
            with col2:
                if not analyze_all:
                    sample_size = st.slider(
                        "Jumlah sampel untuk dianalisis",
                        min_value=1,
                        max_value=min(1000, len(df)),
                        value=min(100, len(df))
                    )
                else:
                    sample_size = len(df)
            
            # Tombol untuk memulai analisis
            if st.button(" Mulai Analisis Batch", type="primary"):
                with st.spinner("Sedang menganalisis data..."):
                    try:
                        # Sampel data jika perlu
                        if analyze_all:
                            sample_df = df.copy()
                        else:
                            sample_df = df.sample(n=sample_size, random_state=42)
                        
                        # Ekstrak komentar
                        comments = sample_df[comment_column].fillna('').tolist()
                        
                        # Lakukan prediksi batch
                        results = predict_batch(comments)
                        
                        # Buat DataFrame hasil
                        results_df = pd.DataFrame(results)
                        
                        # Tambahkan kolom original untuk referensi
                        sample_df = sample_df.reset_index(drop=True)
                        results_df = pd.concat([sample_df, results_df], axis=1)
                        
                        # Tampilkan hasil
                        st.subheader("📊 Hasil Analisis Batch")
                        
                        # Statistik
                        sentiment_counts = results_df['sentiment_label'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            neg_count = sentiment_counts.get("Negatif 😡", 0)
                            st.metric("Negatif", neg_count, f"{neg_count/len(results_df)*100:.1f}%")
                        with col2:
                            pos_count = sentiment_counts.get("Positif 😊", 0)
                            st.metric("Positif", pos_count, f"{pos_count/len(results_df)*100:.1f}%")
                        
                        # Visualisasi distribusi
                        st.subheader("📈 Distribusi Sentimen")
                        
                        # Buat dataframe untuk chart
                        chart_data = pd.DataFrame({
                            'Sentimen': ['Negatif', 'Positif'],
                            'Jumlah': [neg_count, pos_count]
                        })
                        st.bar_chart(chart_data.set_index('Sentimen'))
                        
                        # Pie chart
                        if neg_count > 0 or pos_count > 0:
                            import plotly.express as px
                            
                            pie_data = pd.DataFrame({
                                'Sentimen': ['Negatif', 'Positif'],
                                'Jumlah': [neg_count, pos_count]
                            })
                            
                            fig = px.pie(pie_data, values='Jumlah', names='Sentimen',
                                        title='Distribusi Sentimen',
                                        color='Sentimen',
                                        color_discrete_map={'Negatif': 'red', 'Positif': 'green'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Tampilkan tabel hasil
                        st.subheader("📋 Detail Hasil Prediksi")
                        
                        # Pilih kolom untuk ditampilkan
                        display_cols = [comment_column, 'sentiment_label']
                        if 'probabilities' in results_df.columns:
                            # Ekstrak probabilitas ke kolom terpisah
                            results_df['prob_negatif'] = results_df['probabilities'].apply(
                                lambda x: f"{x['negatif']:.1%}" if x else "N/A"
                            )
                            results_df['prob_positif'] = results_df['probabilities'].apply(
                                lambda x: f"{x['positif']:.1%}" if x else "N/A"
                            )
                            
                            display_cols.extend(['prob_negatif', 'prob_positif'])
                        
                        # Tambahkan kolom cleaned text
                        display_cols.append('cleaned_text')
                        
                        # Tampilkan tabel dengan pilihan kolom
                        st.dataframe(results_df[display_cols], use_container_width=True)
                        
                        # Download hasil
                        st.subheader("💾 Download Hasil")
                        
                        # Convert to Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Hasil_Analisis')
                            
                            # Tambahkan sheet statistik
                            stats_df = pd.DataFrame({
                                'Statistik': ['Total Data', 'Negatif', 'Positif'],
                                'Jumlah': [len(results_df), neg_count, pos_count],
                                'Persentase': [
                                    '100%',
                                    f'{neg_count/len(results_df):.1%}',
                                    f'{pos_count/len(results_df):.1%}'
                                ]
                            })
                            stats_df.to_excel(writer, index=False, sheet_name='Statistik')
                            
                            # Tambahkan sheet untuk visualisasi data
                            summary_df = pd.DataFrame({
                                'Kategori': ['Negatif', 'Positif'],
                                'Jumlah': [neg_count, pos_count],
                                'Persentase': [
                                    f'{neg_count/len(results_df):.1%}',
                                    f'{pos_count/len(results_df):.1%}'
                                ]
                            })
                            summary_df.to_excel(writer, index=False, sheet_name='Summary')
                        
                        output.seek(0)
                        
                        # Tombol download
                        st.download_button(
                            label="📥 Download Hasil (Excel)",
                            data=output,
                            file_name=f"hasil_analisis_sentimen_{uploaded_file.name.split('.')[0]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        # Tombol download CSV
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Hasil (CSV)",
                            data=csv,
                            file_name=f"hasil_analisis_sentimen_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan selama analisis: {str(e)}")
        else:
            st.error(f"❌ {comment_column}")

# Sidebar diagnostics
with st.sidebar:
    st.markdown("### 🔍 Diagnosa")
    
    if st.button("Test Model Sederhana"):
        test_cases = [
            ("jelek", "Harusnya Negatif (0)"),
            ("buruk", "Harusnya Negatif (0)"),
            ("payah", "Harusnya Negatif (0)"),
            ("bagus", "Harusnya Positif (1)"),
            ("baik", "Harusnya Positif (1)"),
            ("keren", "Harusnya Positif (1)"),
            ("suka", "Harusnya Positif (1)"),
            ("benci", "Harusnya Negatif (0)")
        ]
        
        st.write("**Test 2 Kelas:**")
        for text, expected in test_cases:
            analysis = analyze_prediction(text)
            
            # Tampilkan dengan icon
            if analysis['prediction'] == 0:
                icon = "🔴"
            elif analysis['prediction'] == 1:
                icon = "🟢"
            else:
                icon = "⚫"
            
            st.write(f"{icon} **'{text}'**")
            st.write(f"  → Prediksi: {analysis['prediction']} ({analysis['label']})")
            st.write(f"  → Expected: {expected}")
            
            if 'probabilities' in analysis:
                st.write(f"  → Prob: N={analysis['probabilities']['negatif']:.1%}, P={analysis['probabilities']['positif']:.1%}")
            
            st.write("---")
    
    st.markdown("---")
    st.markdown("### 📁 Fitur File")
    st.info("""
    **Format file yang didukung:**
    - Excel (.xlsx, .xls)
    - CSV (.csv)
    
    **Kolom komentar akan otomatis dideteksi.**
    Nama kolom yang didukung:
    - komentar, comment, text, teks
    - review, ulasan, tweet, post
    
    Jika tidak ditemukan, kolom pertama akan digunakan.
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Konfigurasi Model")
    st.write("**Kelas:** 2 (Negatif & Positif)")
    st.write("**Teknik:** Binary Classification")
    st.write("**SMOTE:** Applied for class balance")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p> Dibangun dengan Streamlit | Model: Multinomial Naive Bayes ( TF-IDF + SMOTE ) </p>
</div>
""", unsafe_allow_html=True)
