# app.py (Tamamen Düzeltilmiş Versiyon)

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

# Basit analiz alternatifi - agent hata verdiğinde
def create_simple_analysis(df, question, analysis_type):
    """Agent hata verdiğinde SORU TÜRÜNE GÖRE basit analiz yap"""
    try:
        question_lower = question.lower()
        
        if analysis_type == "Tahmin ve Forecasting" or any(word in question_lower for word in ['tahmin', 'gelecek', 'forecast', 'trend']):
            # SADECE TAHMİN SORULARI İÇİN
            time_cols = []
            for col in df.columns:
                if 'tarih' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                    time_cols.append(col)
                elif df[col].dtype == 'datetime64[ns]':
                    time_cols.append(col)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if time_cols and numeric_cols:
                # Tahmin analizi kodu (önceki gibi)
                time_col = time_cols[0]
                num_col = numeric_cols[0]
                
                df_copy = df.copy()
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
                df_copy = df_copy.sort_values(time_col)
                
                recent_data = df_copy.tail(12)
                if len(recent_data) >= 3:
                    x = np.arange(len(recent_data))
                    y = recent_data[num_col].values
                    slope = np.polyfit(x, y, 1)[0]
                    
                    trend = "artan" if slope > 0 else "azalan" if slope < 0 else "sabit"
                    
                    future_predictions = []
                    for i in range(1, 7):
                        pred = slope * (len(recent_data) + i) + np.polyfit(x, y, 1)[1]
                        future_predictions.append(pred)
                    
                    return f"""
**📈 TREND ANALİZİ VE TAHMİN:**

**Mevcut Durum:**
- Analiz edilen değişken: {num_col}
- Veri aralığı: {df_copy[time_col].min().strftime('%Y-%m')} - {df_copy[time_col].max().strftime('%Y-%m')}
- Trend yönü: **{trend.upper()}**
- Ortalama değer: {recent_data[num_col].mean():.2f}

**Gelecek 6 Ay Tahmini:**
- 1. Ay: {future_predictions[0]:.2f}
- 2. Ay: {future_predictions[1]:.2f}
- 3. Ay: {future_predictions[2]:.2f}
- 4. Ay: {future_predictions[3]:.2f}
- 5. Ay: {future_predictions[4]:.2f}
- 6. Ay: {future_predictions[5]:.2f}

**Analiz Özeti:**
- Son dönemde **{trend}** eğilim gözlemleniyor
- 6 aylık ortalama tahmin: {np.mean(future_predictions):.2f}
- Değişim oranı: {((future_predictions[-1] - recent_data[num_col].iloc[-1]) / recent_data[num_col].iloc[-1] * 100):.1f}%
"""
            else:
                return "Tahmin analizi için uygun zaman serisi verisi bulunamadı."
        
        elif analysis_type == "Genel Analiz" or any(word in question_lower for word in ['analiz', 'özet', 'genel', 'bilgi']):
            # GENEL ANALİZ SORULARI İÇİN
            return f"""
**📋 GENEL VERİ ANALİZİ:**

**Veri Seti Özellikleri:**
- Toplam kayıt sayısı: {len(df):,}
- Sütun sayısı: {len(df.columns)}
- Eksik değer oranı: %{(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}

**Sayısal Sütunlar ({len(df.select_dtypes(include=[np.number]).columns)} adet):**
{', '.join(df.select_dtypes(include=[np.number]).columns.tolist())}

**Kategorik Sütunlar ({len(df.select_dtypes(include=['object', 'category']).columns)} adet):**
{', '.join(df.select_dtypes(include=['object', 'category']).columns.tolist())}

**Temel İstatistikler:**
- En çok satış: {df.select_dtypes(include=[np.number]).max().max():.2f}
- En az değer: {df.select_dtypes(include=[np.number]).min().min():.2f}
- Ortalama: {df.select_dtypes(include=[np.number]).mean().mean():.2f}

💡 **Öneriler:**
- Grafik analizi için: "Kategorilere göre pasta grafiği göster"
- Trend analizi için: "Gelecek dönem tahminini yap"
- Detaylı analiz için: "En çok satan ürünü bul"
"""
        
        else:
            # DİĞER TÜM SORULAR İÇİN - BASIT YANITLAR
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_cols and numeric_cols:
                # En basit analiz
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                grouped = df.groupby(cat_col)[num_col].sum()
                top_category = grouped.idxmax()
                
                return f"""
**📊 HİZLI ANALİZ:**

**Soru:** "{question}"

**Temel Bulgular:**
- En yüksek değere sahip kategori: **{top_category}**
- Bu kategorinin değeri: {grouped.max():.2f}
- Toplam kategori sayısı: {len(grouped)}
- Genel ortalama: {grouped.mean():.2f}

**Veri Özeti:**
- {len(df)} kayıt analiz edildi
- {len(categorical_cols)} kategorik, {len(numeric_cols)} sayısal sütun

💡 **Daha detaylı analiz için grafik talep edebilirsiniz.**
"""
            else:
                return f"""
**ℹ️ Soru:** "{question}"

**Veri Seti Bilgileri:**
- {len(df)} satır, {len(df.columns)} sütun
- Mevcut sütunlar: {', '.join(df.columns.tolist())}

**💡 Öneriler:**
- "Kategorilere göre grafik göster"
- "Trend analizi yap"  
- "En yüksek değerleri bul"
"""
            
    except Exception as e:
        return f"Analiz sırasında hata oluştu: {str(e)}"

# Gelişmiş kod çalıştırma fonksiyonu - AKILLI KOD DEĞİŞTİRME
def execute_agent_code_advanced(code_string, df):
    """Agent kodunu gelişmiş şekilde çalıştır - kod değiştirme ile"""
    
    # Matplotlib ayarları
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
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
        plt.close('all')
        
        # Kod çalıştır
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

# KULLANICI İSTEĞİNE GÖRE KOD DEĞİŞTİRME
def modify_code_based_on_request(code_string, user_question):
    """Kullanıcının isteğine göre kodu değiştir"""
    
    # Pasta grafiği istenmişse bar kodunu pasta ile değiştir
    if any(word in user_question.lower() for word in ['pasta', 'pie', 'dağılım']):
        if 'plt.bar(' in code_string or 'ax.bar(' in code_string:
            # Bar kodunu pasta ile değiştir
            lines = code_string.split('\n')
            new_lines = []
            
            for line in lines:
                if 'plt.bar(' in line or 'ax.bar(' in line:
                    # Bar satırını pasta ile değiştir
                    if 'grouped_data' in code_string:
                        new_lines.append("plt.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%', startangle=90)")
                    else:
                        new_lines.append("# Bar kodu pasta grafiği ile değiştirildi")
                elif 'xlabel' in line or 'ylabel' in line:
                    # Pasta grafiğinde x/y label gerekmez
                    new_lines.append("# " + line)
                else:
                    new_lines.append(line)
            
            return '\n'.join(new_lines)
    
    # Çizgi grafiği istenmişse
    elif any(word in user_question.lower() for word in ['çizgi', 'line', 'trend']):
        if 'plt.bar(' in code_string:
            code_string = code_string.replace('plt.bar(', 'plt.plot(')
            code_string = code_string.replace(', color=', ', marker="o", linewidth=2, color=')
    
    return code_string

# Kod çıkarma fonksiyonu - SESSIZ
def extract_code_from_response(response_text):
    """Agent yanıtından kodu çıkar ama kullanıcıya gösterme"""
    code_blocks = []
    
    # Python kod bloklarını bul
    python_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
    code_blocks.extend(python_blocks)
    
    # Eğer kod bloğu yoksa, farklı formatları dene
    if not code_blocks:
        # Basit ``` formatı
        general_blocks = re.findall(r'```\n(.*?)```', response_text, re.DOTALL)
        for block in general_blocks:
            if any(keyword in block for keyword in ['plt.', 'sns.', 'px.', 'df[', 'import']):
                code_blocks.append(block)
    
    # Eğer hala kod yoksa, matplotlib içeren satırları ara
    if not code_blocks:
        lines = response_text.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in ['plt.', 'sns.', 'px.', 'fig', 'ax', 'df[', 'import matplotlib', 'import pandas']):
                code_lines.append(line)
        
        if code_lines:
            code_blocks.append('\n'.join(code_lines))
    
    return code_blocks

# Yanıttan kod bloklarını tamamen temizle
def clean_response_from_code(response_text):
    """AI yanıtından kod bloklarını ve kod benzeri satırları tamamen temizle"""
    
    # Python kod bloklarını kaldır
    cleaned_text = re.sub(r'```python\n.*?```', '', response_text, flags=re.DOTALL)
    
    # Diğer kod bloklarını kaldır
    cleaned_text = re.sub(r'```.*?```', '', cleaned_text, flags=re.DOTALL)
    
    # Satır satır temizleme
    lines = cleaned_text.split('\n')
    clean_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Kod benzeri satırları filtrele
        if not any(keyword in line for keyword in [
            'import ', 'plt.', 'df[', 'df.', 'pd.', 'np.',
            '=', 'pandas', 'matplotlib', 'seaborn',
            'plt.', 'sns.', 'px.', 'go.', 'fig,', 'ax',
            '.groupby(', '.plot(', '.show()', '.savefig(',
            'pd.DataFrame', 'np.random', 'plt.figure'
        ]):
            # Sadece anlamlı açıklama satırları
            if line_stripped and not line_stripped.startswith('#'):
                clean_lines.append(line)
    
    # Temizlenmiş metni birleştir
    cleaned_text = '\n'.join(clean_lines)
    
    # Çok fazla boş satırları temizle
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

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
            st.rerun()
    
    else:
        st.info("🔧 SQL bağlantısı özelliği yakında eklenecek.")

# Ana içerik
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Veri önizleme - SADECE VERİ KALİTESİ
    st.header("👀 2. Veri Önizleme")
    
    # Sadece veri kalitesi metrikleri - 4 sütun
    col1, col2, col3, col4 = st.columns(4)
    
    total_rows = len(df)
    missing_data = df.isnull().sum().sum()
    completeness = ((total_rows * len(df.columns) - missing_data) / (total_rows * len(df.columns))) * 100
    duplicate_count = df.duplicated().sum()
    
    with col1:
        st.metric("📊 Toplam Satır", f"{total_rows:,}")
    
    with col2:
        st.metric("📋 Sütun Sayısı", f"{len(df.columns)}")
    
    with col3:
        st.metric("✅ Veri Bütünlüğü", f"{completeness:.1f}%")
    
    with col4:
        st.metric("🔍 Eksik Değer", f"{missing_data}")
    
    # İsteğe bağlı: Daha detaylı bilgi için expander
    with st.expander("🔧 Detaylı Veri Bilgisi"):
        st.write(f"**Duplicate Kayıt:** {duplicate_count}")
        st.write(f"**Sayısal Sütunlar:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Kategorik Sütunlar:** {len(df.select_dtypes(include=['object', 'category']).columns)}")
        if df.select_dtypes(include=['datetime']).columns.tolist():
            st.write(f"**Tarih Sütunları:** {', '.join(df.select_dtypes(include=['datetime']).columns.tolist())}")

    # Analiz bölümü
    st.header("🎯 3. Akıllı Veri Analizi")
    
    # Analiz kategorileri - GENEL ANALİZ KALDIRILDI
    analysis_type = st.selectbox(
        "🔍 Analiz Türü Seçin:",
        [
            "Tahmin ve Forecasting", 
            "Görselleştirme",
            "İstatistiksel Analiz",
            "Trend Analizi"
        ]
    )
    
    # Örnek sorular kategoriye göre - GÜNCELLENDİ
    if analysis_type == "Tahmin ve Forecasting":
        example_questions = [
            "En çok satan ürünün gelecek aylardaki performansını tahmin et",
            "2025 yılı satış tahminini yap"
        ]
    elif analysis_type == "Görselleştirme":
        example_questions = [
            "Kategorilere göre satış dağılımını bar grafiği ile göster",
            "Zaman içinde satış trendini çizgi grafiği ile göster",
            "Bölgelere göre performansı pasta grafiği ile göster"
        ]
    elif analysis_type == "İstatistiksel Analiz":
        example_questions = [
            "En çok satan kategoriyi bul",
            "Ortalama satış tutarını hesapla"
        ]
    elif analysis_type == "Trend Analizi":
        example_questions = [
            "Yıllar arası büyüme oranını hesapla",
            "En hızlı büyüyen kategorileri bul"
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
                
                # Geliştirilmiş prompt - ULTRA GÜÇLENDİRİLDİ
                enhanced_question = f"""
                Analiz Türü: {analysis_type}
                Kullanıcı Sorusu: "{user_question}"
                
                ÇOK ÖNEMLİ: SORUYU DİKKATLE ANALİZ ET!
                
                1. SORU ANALİZİ:
                   Kullanıcı hangi sütun/kategoriye göre analiz istiyor?
                   - "Bölgelere göre" → Bölge/Şehir sütununu kullan
                   - "Kategorilere göre" → Kategori sütununu kullan
                   - "Müşterilere göre" → Müşteri sütununu kullan
                   - "Ürünlere göre" → Ürün sütununu kullan
                
                2. GRAFİK TÜRÜ ANALİZİ:
                   Kullanıcı hangi grafik türü istiyor?
                   - "pasta grafiği" → MUTLAKA plt.pie() kullan
                   - "bar grafiği" → plt.bar() kullan
                   - "çizgi grafiği" → plt.plot() kullan
                
                3. YANIT DİLİ: MUTLAKA TÜRKÇE
                
                4. EĞER TAHMİN/FORECASTING SORGUSU İSE:
                   - Zaman serisi analizi yap
                   - Trend hesapla (artan/azalan/sabit)
                   - Gelecek değerleri tahmin et
                   - Matematiksel modelleme kullan
                   - Sonuçları görselleştir
                   - Analizi Türkçe açıkla
                
                5. EĞER GÖRSELLEŞTİRME SORGUSU İSE:
                   - Kullanıcının istediği SÜTUNU kullan
                   - Kullanıcının istediği GRAFİK TÜRÜNÜ kullan
                   - DOĞRU SÜTUN SEÇİMİ ÇOK ÖNEMLİ!
                   - plt.tight_layout() kullan (plt.show() ASLA KULLANMA!)
                   - Türkçe başlık ve etiketler ekle
                   - Grafik boyutunu ayarla: plt.figure(figsize=(10, 6))
                
                6. GENEL GEREKSINIMLER:
                   - Kullanıcının sorusunu KELİME KELİME analiz et
                   - YANLIŞLIK YAPMA - doğru sütunu seç
                   - Sonuçları detaylı olarak Türkçe açıkla
                   - Sayısal değerleri belirt
                   - Python kodunu ```python ile başlat
                
                MEVCUT VERİ SÜTUNLARI:
                {', '.join(df.columns.tolist())}
                
                KRITIK: 
                - Kullanıcının istediği sütunu MUTLAKA kullan
                - Yanlış sütun seçme!
                - Grafik türünü doğru seç
                - Tüm açıklamaları Türkçe yap
                """
                
                # Agent'ı çalıştır - DAHA KARARLI HATA YÖNETİMİ
                try:
                    # Önce basit bir test sorusu ile agent'ın çalışıp çalışmadığını kontrol et
                    test_response = pandas_agent_executor.invoke("Veri setinin ilk 3 satırını göster")
                    
                    # Test başarılıysa asıl soruyu sor
                    response = pandas_agent_executor.invoke(enhanced_question)
                    st.session_state.agent_response = response
                    
                except Exception as agent_error:
                    # Agent parsing hatası durumunda - SORU TÜRÜNE GÖRE AKILLI ANALİZ
                    if analysis_type == "Tahmin ve Forecasting":
                        simple_response = create_simple_analysis(df, user_question, analysis_type)
                    elif any(word in user_question.lower() for word in ['grafik', 'görsel', 'chart', 'göster', 'çiz', 'pasta', 'pie', 'bar', 'çubuk']):
                        # Görselleştirme soruları için bypass sistemi kullan
                        simple_response = "Grafik analizi hazırlanıyor..."
                    else:
                        # Diğer sorular için genel analiz
                        simple_response = create_simple_analysis(df, user_question, "Genel Analiz")
                    
                    st.session_state.agent_response = {"output": simple_response}
                
            except Exception as e:
                st.error(f"❌ Analiz sırasında hata oluştu: {e}")
                st.write("**Hata Detayları:**")
                st.code(str(e))
                
                # Hata durumunda basit analiz öner
                st.info("💡 Daha basit bir soru deneyin veya veri formatını kontrol edin.")

# Sonuçları göster - KOD BLOKLARI TAMAMEN GİZLİ
if st.session_state.agent_response is not None:
    st.header("📊 Analiz Sonuçları")
    
    response = st.session_state.agent_response
    
    # Yanıtı çıkar
    if isinstance(response, dict):
        agent_output = response.get('output', 'Sonuç bulunamadı.')
    else:
        agent_output = str(response)
    
    # Kod bloklarını tamamen temizle
    clean_response = clean_response_from_code(agent_output)
    
    # Eğer temizledikten sonra çok az metin kaldıysa, daha iyi temizleme yap
    if len(clean_response.strip()) < 50:
        # Alternatif temizleme - satır satır kontrol et
        lines = agent_output.split('\n')
        clean_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if not in_code_block:
                # Kod benzeri satırları filtrele
                if not any(keyword in line for keyword in ['import ', 'plt.', 'df[', 'df.', '=', 'pandas', 'matplotlib']):
                    if line.strip():  # Boş olmayan satırlar
                        clean_lines.append(line)
        
        clean_response = '\n'.join(clean_lines).strip()
    
    # AI'nın temiz yanıtını göster
    st.subheader("🤖 AI Asistanının Yorumu")
    if clean_response:
        st.write(clean_response)
    else:
        st.write("Analiz tamamlandı. Grafik aşağıda gösterilmektedir.")
    
    # SON ÇARE: AGENT BYPASS - DİREKT GRAFİK SİSTEMİ
    if any(word in user_question.lower() for word in ['grafik', 'görsel', 'chart', 'göster', 'çiz', 'pasta', 'pie', 'bar', 'çubuk', 'dağılım']):
        
        # Kullanıcının hangi sütunu istediğini analiz et
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Sütun seçimi - SORU ANALİZİ
        target_cat_col = None
        target_num_col = None
        
        query_lower = user_question.lower()
        
        # Kategorik sütun seçimi
        if 'bölge' in query_lower or 'şehir' in query_lower:
            for col in categorical_cols:
                if any(word in col.lower() for word in ['bölge', 'şehir', 'region', 'city']):
                    target_cat_col = col
                    break
        elif 'kategori' in query_lower:
            for col in categorical_cols:
                if 'kategori' in col.lower() or 'category' in col.lower():
                    target_cat_col = col
                    break
        elif 'müşteri' in query_lower:
            for col in categorical_cols:
                if 'müşteri' in col.lower() or 'customer' in col.lower():
                    target_cat_col = col
                    break
        elif 'ürün' in query_lower:
            for col in categorical_cols:
                if 'ürün' in col.lower() or 'product' in col.lower():
                    target_cat_col = col
                    break
        
        # Sayısal sütun seçimi
        if 'satış' in query_lower or 'tutar' in query_lower:
            for col in numeric_cols:
                if any(word in col.lower() for word in ['satış', 'tutar', 'sales', 'revenue']):
                    target_num_col = col
                    break
        elif 'adet' in query_lower or 'miktar' in query_lower:
            for col in numeric_cols:
                if any(word in col.lower() for word in ['adet', 'miktar', 'quantity']):
                    target_num_col = col
                    break
        elif 'kar' in query_lower:
            for col in numeric_cols:
                if 'kar' in col.lower() or 'profit' in col.lower():
                    target_num_col = col
                    break
        
        # Varsayılan seçimler
        if not target_cat_col and categorical_cols:
            target_cat_col = categorical_cols[0]
        if not target_num_col and numeric_cols:
            target_num_col = numeric_cols[0]
        
        # Grafik oluştur
        if target_cat_col and target_num_col:
            st.subheader("📈 Görselleştirme")
            
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                
                # Veriyi grupla
                data_grouped = df.groupby(target_cat_col)[target_num_col].sum()
                
                # Grafik türü seçimi
                if any(word in query_lower for word in ['pasta', 'pie', 'dağılım']):
                    # PASTA GRAFİĞİ
                    colors = plt.cm.Set3(range(len(data_grouped)))
                    plt.pie(data_grouped.values, labels=data_grouped.index, autopct='%1.1f%%', 
                           startangle=90, colors=colors)
                    plt.title(f'{target_cat_col} Bazında {target_num_col} Dağılımı', fontsize=14, pad=20)
                    
                elif any(word in query_lower for word in ['çizgi', 'line', 'trend']):
                    # ÇİZGİ GRAFİĞİ
                    plt.plot(data_grouped.index, data_grouped.values, marker='o', linewidth=2, markersize=8, color='blue')
                    plt.title(f'{target_cat_col} Bazında {target_num_col} Trendi', fontsize=14)
                    plt.xlabel(target_cat_col)
                    plt.ylabel(target_num_col)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    
                else:
                    # BAR GRAFİĞİ
                    colors = plt.cm.viridis(range(len(data_grouped)))
                    plt.bar(data_grouped.index, data_grouped.values, color=colors, alpha=0.8)
                    plt.title(f'{target_cat_col} Bazında {target_num_col} Dağılımı', fontsize=14)
                    plt.xlabel(target_cat_col)
                    plt.ylabel(target_num_col)
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(plt, use_container_width=True)
                
                # Analiz özeti göster
                st.write(f"**📊 Analiz Özeti:**")
                st.write(f"- **Seçilen kategori:** {target_cat_col}")
                st.write(f"- **Analiz edilen değer:** {target_num_col}")
                st.write(f"- **En yüksek değer:** {data_grouped.idxmax()} ({data_grouped.max():,.2f})")
                st.write(f"- **Toplam değer:** {data_grouped.sum():,.2f}")
                st.write(f"- **Kategori sayısı:** {len(data_grouped)}")
                
            except Exception as e:
                st.error(f"Grafik oluşturulurken hata: {e}")
    
    # Kod çıkarma ve çalıştırma - İKİNCİL SİSTEM
    code_blocks = extract_code_from_response(agent_output)
    
    if code_blocks and not any(word in user_question.lower() for word in ['grafik', 'görsel', 'chart', 'göster', 'çiz', 'pasta', 'pie', 'bar', 'çubuk', 'dağılım']):
        st.subheader("📈 Ek Görselleştirme")
        
        for i, code_block in enumerate(code_blocks):
            # Kullanıcının isteğine göre kodu değiştir
            modified_code = modify_code_based_on_request(code_block, user_question)
            
            # Değiştirilmiş kodu çalıştır
            result = execute_agent_code_advanced(modified_code, df)
            
            if result['success']:
                # Matplotlib figürlerini göster
                for fig in result['matplotlib_figs']:
                    st.pyplot(fig, use_container_width=True)
                break  # İlk başarılı grafik yeterli
    
    else:
        # Eğer grafik kodu bulunamadıysa hiçbir şey gösterme - UYARISIZ
        pass  # Hiç uyarı yok
    
    # Gelişmiş özellikler
    with st.expander("🔧 Gelişmiş Ayarlar"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Sonuçları Temizle"):
                st.session_state.agent_response = None
                st.session_state.generated_plots = []
                st.rerun()
        
        with col2:
            if st.button("📥 Sonuçları İndir"):
                # Temiz metni indir
                st.download_button(
                    label="📄 Analiz Raporunu İndir",
                    data=clean_response,
                    file_name=f"analiz_raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

# Sidebar - Analiz geçmişi
if st.session_state.analysis_history:
    with st.sidebar:
        st.header("📝 Analiz Geçmişi")
        
# Sidebar - Analiz geçmişi
if st.session_state.analysis_history:
    with st.sidebar:
        st.header("📝 Analiz Geçmişi")
        
        for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Son 5 analiz
            with st.expander(f"{item['type']} - {item['timestamp'].strftime('%H:%M')}"):
                st.write(f"**Soru:** {item['question'][:100]}...")
                if st.button(f"🔄 Tekrar Çalıştır", key=f"rerun_{i}"):
                    st.session_state.user_question = item['question']
                    st.rerun()

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