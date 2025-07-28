# app.py (Geliştirilmiş Versiyon)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import sys
import re
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Agent import
from agents.data_analysis_agent import create_pandas_agent
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# Streamlit yapılandırması
st.set_page_config(
    page_title="AI Raporlama Asistanı", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Ana başlık
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🤖 Yapay Zeka Destekli Dinamik Raporlama Asistanı</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">Veri analizi, tahmin ve görselleştirme için akıllı asistanınız</p>
</div>
""", unsafe_allow_html=True)

# Session state başlatma
def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'agent_response' not in st.session_state:
        st.session_state.agent_response = None
    if 'generated_plots' not in st.session_state:
        st.session_state.generated_plots = []
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

initialize_session_state()

# Gelişmiş kod çalıştırma fonksiyonu
def execute_agent_code_advanced(code_string, df):
    """Agent kodunu gelişmiş şekilde çalıştır"""
    
    # Matplotlib ayarları
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    exec_globals = {
        'df': df,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'px': px,
        'go': go,
        'np': np,
        'st': st,
        'datetime': datetime,
        'timedelta': timedelta
    }
    
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        plt.close('all')  # Önceki figürleri temizle
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code_string, exec_globals)
        
        # Matplotlib figürlerini yakala
        matplotlib_figs = []
        for i in plt.get_fignums():
            fig = plt.figure(i)
            matplotlib_figs.append(fig)
        
        return {
            'success': True,
            'stdout': output_buffer.getvalue(),
            'stderr': error_buffer.getvalue(),
            'matplotlib_figs': matplotlib_figs,
            'exec_globals': exec_globals
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'stdout': output_buffer.getvalue(),
            'stderr': error_buffer.getvalue(),
            'matplotlib_figs': []
        }

# Gelişmiş kod çıkarma fonksiyonu
def extract_code_from_response(response_text):
    """Agent yanıtından kodu çıkar"""
    code_blocks = []
    
    # Python kod bloklarını bul
    python_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
    code_blocks.extend(python_blocks)
    
    # Kod bloğu yoksa satır satır ara
    if not code_blocks:
        lines = response_text.split('\n')
        code_lines = []
        
        for line in lines:
            if any(keyword in line for keyword in ['plt.', 'sns.', 'px.', 'fig', 'ax', 'df[']):
                code_lines.append(line)
        
        if code_lines:
            code_blocks.append('\n'.join(code_lines))
    
    return code_blocks

# Sidebar - Veri yükleme
with st.sidebar:
    st.header("📁 1. Veri Kaynağı")
    
    source_type = st.radio(
        "Veri Kaynağı Seçin:",
        ["CSV/Excel Dosyası", "Örnek Veri Seti", "SQL Bağlantısı (Yakında)"]
    )
    
    if source_type == "CSV/Excel Dosyası":
        uploaded_file = st.file_uploader(
            "Dosya Seçin", 
            type=['csv', 'xlsx', 'xls'],
            help="CSV veya Excel dosyası yükleyebilirsiniz"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Dosya yükleniyor..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    else:
                        df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                
                st.markdown('<div class="success-message">✅ Dosya başarıyla yüklendi!</div>', 
                           unsafe_allow_html=True)
                
                # Veri özeti
                st.subheader("📊 Veri Özeti")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Satır Sayısı", len(df))
                with col2:
                    st.metric("Sütun Sayısı", len(df.columns))
                
                # Sütun bilgileri
                with st.expander("🔍 Sütun Detayları"):
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        null_count = df[col].isnull().sum()
                        st.write(f"**{col}:** {dtype} ({null_count} eksik)")
                        
            except Exception as e:
                st.error(f"❌ Dosya okuma hatası: {e}")
    
    elif source_type == "Örnek Veri Seti":
        st.subheader("📋 Hazır Veri Setleri")
        
        dataset_choice = st.selectbox(
            "Veri seti seçin:",
            ["Satış Verileri", "E-ticaret Verileri", "Finansal Veriler"]
        )
        
        if st.button("🚀 Veri Setini Yükle"):
            if dataset_choice == "Satış Verileri":
                # Satış verileri oluştur
                np.random.seed(42)
                dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
                categories = ['Elektronik', 'Giyim', 'Kitap', 'Ev & Bahçe', 'Spor', 'Kozmetik']
                regions = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Adana']
                
                data = []
                for date in dates:
                    for _ in range(np.random.randint(1, 5)):
                        data.append({
                            'Tarih': date,
                            'Kategori': np.random.choice(categories),
                            'Bölge': np.random.choice(regions),
                            'Satış_Tutarı': np.random.randint(100, 10000),
                            'Adet': np.random.randint(1, 50),
                            'Kar': np.random.randint(10, 2000),
                            'Müşteri_Tipi': np.random.choice(['Bireysel', 'Kurumsal'])
                        })
                
                st.session_state.df = pd.DataFrame(data)
                
            elif dataset_choice == "E-ticaret Verileri":
                # E-ticaret verileri
                np.random.seed(42)
                products = ['Laptop', 'Telefon', 'Tablet', 'Kulaklık', 'Kamera', 'Saat']
                
                data = []
                for i in range(1000):
                    data.append({
                        'Ürün': np.random.choice(products),
                        'Fiyat': np.random.randint(500, 15000),
                        'İndirim_Oranı': np.random.randint(0, 50),
                        'Değerlendirme': np.random.uniform(3.0, 5.0),
                        'Satış_Adedi': np.random.randint(1, 100),
                        'Stok': np.random.randint(0, 500),
                        'Tarih': pd.date_range('2024-01-01', periods=1000, freq='H')[i]
                    })
                
                st.session_state.df = pd.DataFrame(data)
            
            st.success("✅ Örnek veri seti yüklendi!")
            st.experimental_rerun()
    
    else:
        st.info("🔧 SQL bağlantısı özelliği yakında eklenecek.")

# Ana içerik
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Veri önizleme
    st.header("👀 2. Veri Önizleme")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📋 Veri Tablosu")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Temel İstatistikler")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write(df[numeric_cols].describe())
    
    with col3:
        st.subheader("🔍 Veri Kalitesi")
        total_rows = len(df)
        missing_data = df.isnull().sum().sum()
        completeness = ((total_rows * len(df.columns) - missing_data) / (total_rows * len(df.columns))) * 100
        
        st.metric("Veri Bütünlüğü", f"{completeness:.1f}%")
        st.metric("Eksik Değer", missing_data)
        st.metric("Duplicate", df.duplicated().sum())

    # Analiz bölümü
    st.header("🎯 3. Akıllı Veri Analizi")
    
    # Analiz kategorileri
    analysis_type = st.selectbox(
        "🔍 Analiz Türü Seçin:",
        [
            "Genel Analiz", 
            "Tahmin ve Forecasting", 
            "Görselleştirme", 
            "İstatistiksel Analiz",
            "Trend Analizi"
        ]
    )
    
    # Örnek sorular kategoriye göre
    if analysis_type == "Tahmin ve Forecasting":
        example_questions = [
            "2025 yılı satış tahminini yap",
            "Gelecek 6 ay için trend analizi yap",
            "En çok satan ürünün gelecek aylardaki performansını tahmin et",
            "Kategori bazında gelecek dönem satış projeksiyonu",
            "Mevsimsel analiz ile gelecek yıl tahmini"
        ]
    elif analysis_type == "Görselleştirme":
        example_questions = [
            "Kategorilere göre satış dağılımını bar grafiği ile göster",
            "Zaman içinde satış trendini çizgi grafiği ile göster",
            "Bölgelere göre performansı pasta grafiği ile göster",
            "Ürün kategorileri arası korelasyonu scatter plot ile göster",
            "Aylık satış dağılımını histogram ile analiz et"
        ]
    elif analysis_type == "İstatistiksel Analiz":
        example_questions = [
            "En çok satan kategoriyi bul",
            "Ortalama satış tutarını hesapla",
            "Satış performansının istatistiksel özeti",
            "Outlier (aykırı değer) analizi yap",
            "Korelasyon matrisi oluştur"
        ]
    elif analysis_type == "Trend Analizi":
        example_questions = [
            "Son 6 ayın trend analizi",
            "Mevsimsel satış paternlerini analiz et",
            "Yıllar arası büyüme oranını hesapla",
            "En hızlı büyüyen kategorileri bul",
            "Düşüş trendi gösteren ürünleri tespit et"
        ]
    else:  # Genel Analiz
        example_questions = [
            "Veri setinin genel özetini çıkar",
            "En önemli bulguları listele",
            "Eksik veriler hakkında bilgi ver",
            "Temel istatistikleri göster",
            "Veri kalitesi analizi yap"
        ]
    
    # Örnek sorular gösterimi
    with st.expander(f"💡 {analysis_type} Örnek Soruları"):
        for i, question in enumerate(example_questions, 1):
            if st.button(f"{i}. {question}", key=f"example_{i}"):
                st.session_state.user_question = question
    
    # Soru girişi
    user_question = st.text_area(
        "🤔 Sorunuzu buraya yazın:",
        value=st.session_state.get('user_question', ''),
        height=120,
        placeholder=f"Örneğin: {example_questions[0]}",
        help="Detaylı sorular daha iyi sonuçlar verir"
    )
    
    # Analiz butonu
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analiz Et", 
            type="primary", 
            use_container_width=True,
            disabled=not user_question.strip()
        )
    
    # Analiz işlemi
    if analyze_button and user_question.strip():
        # Soruyu geçmişe ekle
        st.session_state.analysis_history.append({
            'question': user_question,
            'timestamp': datetime.now(),
            'type': analysis_type
        })
        
        with st.spinner("🤖 AI analiz yapıyor ve sonuçları hazırlıyor..."):
            try:
                # Agent'ı oluştur
                pandas_agent_executor = create_pandas_agent(df)
                
                # Geliştirilmiş prompt
                enhanced_question = f"""
                Analiz Türü: {analysis_type}
                Soru: {user_question}
                
                Lütfen aşağıdaki gereksinimleri karşıla:
                
                1. EĞER TAHMİN/FORECASTING SORGUSU İSE:
                   - Zaman serisi analizi yap
                   - Trend hesapla (artan/azalan/sabit)
                   - Gelecek değerleri tahmin et
                   - Matematiksel modelleme kullan
                   - Sonuçları görselleştir
                
                2. EĞER GÖRSELLEŞTİRME SORGUSU İSE:
                   - matplotlib veya seaborn kullan
                   - Uygun grafik türünü seç
                   - plt.tight_layout() kullan (plt.show() değil)
                   - Türkçe başlık ve etiketler ekle
                   - Renklendirme ve stil uygula
                
                3. GENEL GEREKSINIMLER:
                   - Sonuçları detaylı açıkla
                   - Önemli bulguları özetle
                   - Sayısal değerleri belirt
                   - Türkçe yanıt ver
                
                Veri sütunları: {', '.join(df.columns.tolist())}
                """
                
                # Agent'ı çalıştır
                response = pandas_agent_executor.invoke(enhanced_question)
                st.session_state.agent_response = response
                
            except Exception as e:
                st.error(f"❌ Analiz sırasında hata oluştu: {e}")
                st.write("**Hata Detayları:**")
                st.code(str(e))
                
                # Hata durumunda basit analiz öner
                st.info("💡 Daha basit bir soru deneyin veya veri formatını kontrol edin.")

# Sonuçları göster
if st.session_state.agent_response is not None:
    st.header("📊 Analiz Sonuçları")
    
    response = st.session_state.agent_response
    
    # Yanıtı çıkar
    if isinstance(response, dict):
        agent_output = response.get('output', 'Sonuç bulunamadı.')
    else:
        agent_output = str(response)
    
    # Ana yanıt
    st.subheader("🤖 AI Asistanının Yanıtı")
    st.write(agent_output)
    
    # Kod çıkarma ve çalıştırma
    code_blocks = extract_code_from_response(agent_output)
    
    if code_blocks:
        st.subheader("📈 Görselleştirme Sonuçları")
        
        for i, code_block in enumerate(code_blocks):
            with st.expander(f"Kod Bloğu {i+1}", expanded=True):
                st.code(code_block, language='python')
                
                # Kodu çalıştır
                result = execute_agent_code_advanced(code_block, df)
                
                if result['success']:
                    # Matplotlib figürlerini göster
                    for fig in result['matplotlib_figs']:
                        st.pyplot(fig, use_container_width=True)
                    
                    # Stdout çıktısı varsa göster
                    if result['stdout'].strip():
                        st.subheader("📋 Çıktı")
                        st.text(result['stdout'])
                        
                else:
                    st.error(f"❌ Kod çalıştırma hatası: {result.get('error', 'Bilinmeyen hata')}")
                    if result['stdout']:
                        st.write("**Stdout:**")
                        st.text(result['stdout'])
                    if result['stderr']:
                        st.write("**Stderr:**")
                        st.text(result['stderr'])
    
    # Gelişmiş özellikler
    with st.expander("🔧 Gelişmiş Ayarlar"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Sonuçları Temizle"):
                st.session_state.agent_response = None
                st.session_state.generated_plots = []
                st.experimental_rerun()
        
        with col2:
            if st.button("📥 Sonuçları İndir"):
                # Basit metin indirme
                st.download_button(
                    label="📄 Metin olarak indir",
                    data=agent_output,
                    file_name=f"analiz_sonucu_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        
        # Ham yanıt göster
        st.subheader("🔍 Ham Agent Yanıtı")
        st.json(response if isinstance(response, dict) else {"output": str(response)})

# Sidebar - Analiz geçmişi
if st.session_state.analysis_history:
    with st.sidebar:
        st.header("📝 Analiz Geçmişi")
        
        for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Son 5 analiz
            with st.expander(f"{item['type']} - {item['timestamp'].strftime('%H:%M')}"):
                st.write(f"**Soru:** {item['question'][:100]}...")
                if st.button(f"🔄 Tekrar Çalıştır", key=f"rerun_{i}"):
                    st.session_state.user_question = item['question']
                    st.experimental_rerun()

elif st.session_state.df is None:
    # Veri yüklememiş kullanıcı için rehber
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Hoş Geldiniz!</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Başlamak için sol taraftaki panelden bir veri dosyası yükleyin 
            veya örnek veri seti ile deneyin.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Özellikler tanıtımı
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Akıllı Analiz
        - Doğal dille soru sorun
        - Otomatik veri keşfi
        - İstatistiksel analiz
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Tahmin Modelleri
        - Zaman serisi analizi
        - Trend tahmini
        - Gelecek dönem projeksiyonu
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Görselleştirme
        - Otomatik grafik oluşturma
        - İnteraktif çizelgeler
        - Özelleştirilebilir görünüm
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>💡 Kullanım İpuçları:</strong></p>
    <p>• Tahmin soruları için "gelecek", "tahmin", "2025" gibi anahtar kelimeler kullanın</p>
    <p>• Grafik için "göster", "çiz", "görselleştir" ifadelerini ekleyin</p>
    <p>• Detaylı sorular daha iyi sonuçlar verir</p>
</div>
""", unsafe_allow_html=True)