# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import warnings
warnings.filterwarnings('ignore')

# --- Agent import'u ---
from agents.data_analysis_agent import create_pandas_agent
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# Streamlit yapılandırması
st.set_page_config(page_title="AI Raporlama Asistanı", page_icon="🤖", layout="wide")
st.title("🤖 Yapay Zeka Destekli Dinamik Raporlama Asistanı")
st.write("Veri setinizi yükleyin, ardından sorularınızı sormaya başlayın!")

# Session state başlatma
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent_response' not in st.session_state:
    st.session_state.agent_response = None
if 'generated_plots' not in st.session_state:
    st.session_state.generated_plots = []

# Grafik yakalama fonksiyonu
def capture_matplotlib_plots():
    """Matplotlib figürlerini yakalar ve Streamlit'e gösterir"""
    figs = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        figs.append(fig)
    return figs

# Kod çalıştırma fonksiyonu (grafik destekli)
def execute_agent_code(code_string, df):
    """Agent'ın ürettiği kodu güvenli şekilde çalıştırır ve grafikleri yakalar"""
    
    # Kod çalıştırma ortamını hazırla
    exec_globals = {
        'df': df,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'px': px,
        'go': go,
        'st': st
    }
    
    # Çıktıları yakala
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    
    try:
        # Mevcut figürleri temizle
        plt.close('all')
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code_string, exec_globals)
        
        # Matplotlib figürlerini yakala
        matplotlib_figs = capture_matplotlib_plots()
        
        # Çıktıları al
        stdout_output = output_buffer.getvalue()
        stderr_output = error_buffer.getvalue()
        
        return {
            'success': True,
            'stdout': stdout_output,
            'stderr': stderr_output,
            'matplotlib_figs': matplotlib_figs,
            'exec_globals': exec_globals
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'stdout': output_buffer.getvalue(),
            'stderr': error_buffer.getvalue(),
            'matplotlib_figs': [],
            'exec_globals': exec_globals
        }

# Sidebar - Veri yükleme
with st.sidebar:
    st.header("1. Veri Kaynağınızı Seçin")
    source_type = st.radio("Veri Kaynağı Tipi", ["CSV/Excel Dosyası Yükle", "SQL Veritabanına Bağlan (Yakında)"])
    
    if source_type == "CSV/Excel Dosyası Yükle":
        uploaded_file = st.file_uploader("Bir CSV veya Excel dosyası seçin", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.success("Dosya başarıyla yüklendi!")
                
                # Veri hakkında temel bilgiler
                st.write(f"**Satır sayısı:** {len(df)}")
                st.write(f"**Sütun sayısı:** {len(df.columns)}")
                st.write("**Sütun isimleri:**")
                for col in df.columns:
                    st.write(f"- {col}")
                    
            except Exception as e:
                st.error(f"Dosya okunurken bir hata oluştu: {e}")
    
    elif source_type == "SQL Veritabanına Bağlan (Yakında)":
        st.info("Bu özellik yakında eklenecektir.")

# Ana içerik
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Veri önizleme
    st.header("2. Verinize Göz Atın")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.write("**Veri Özeti:**")
        st.write(f"Toplam satır: {len(df)}")
        st.write(f"Toplam sütun: {len(df.columns)}")
        st.write("**Veri Tipleri:**")
        for col, dtype in df.dtypes.items():
            st.write(f"- {col}: {dtype}")

    # Analiz bölümü
    st.header("3. Analiz Sorunuzu Sorun")
    
    # Örnek sorular
    with st.expander("💡 Örnek Sorular"):
        st.write("""
        - Kategorilere göre toplam satış tutarını bir bar grafiği ile göster
        - Aylık satış trendini çizgi grafiği ile göster
        - En çok satan 10 ürünü görselleştir
        - Bölgelere göre satış dağılımını pasta grafiği ile göster
        - Satış ve kar arasındaki ilişkiyi scatter plot ile göster
        - Ürün kategorilerinin satış performansını box plot ile analiz et
        """)
    
    user_question = st.text_area(
        "Sorunuzu buraya yazın:", 
        height=100,
        placeholder="Örneğin: Kategorilere göre toplam satış tutarını bir bar grafiği ile göster"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analiz Et", type="primary")
    
    if analyze_button and user_question:
        with st.spinner("🤖 AI analiz yapıyor ve görselleştirme hazırlıyor..."):
            try:
                # Agent'ı oluştur ve çalıştır
                pandas_agent_executor = create_pandas_agent(df)
                
                # Geliştirilmiş prompt
                enhanced_question = f"""
                {user_question}
                
                Lütfen aşağıdaki gereksinimleri karşıla:
                1. Eğer görselleştirme isteniyorsa, matplotlib veya seaborn kullanarak grafik oluştur
                2. plt.show() yerine plt.tight_layout() kullan
                3. Grafik başlığı ve eksen etiketlerini Türkçe olarak ekle
                4. Sonuçları açıklayıcı bir metin ile birlikte sun
                5. Eğer veri analizi yapıyorsan, önemli bulgularını özetle
                """
                
                response = pandas_agent_executor.invoke(enhanced_question)
                st.session_state.agent_response = response
                
            except Exception as e:
                st.error(f"Analiz sırasında hata oluştu: {e}")
                st.write("**Hata detayları:**")
                st.code(str(e))

# Sonuçları göster
if st.session_state.agent_response is not None:
    st.header("📊 Analiz Sonuçları")
    
    response = st.session_state.agent_response
    
    # Ana yanıt
    if isinstance(response, dict):
        agent_output = response.get('output', 'Sonuç bulunamadı.')
    else:
        agent_output = str(response)
    
    st.write("**🤖 AI Asistanının Yanıtı:**")
    st.write(agent_output)
    
    # Eğer agent kod ürettiyse, onu çalıştırmaya çalış
    if 'plt.' in agent_output or 'sns.' in agent_output or 'matplotlib' in agent_output.lower():
        st.write("**📈 Görselleştirme:**")
        
        # Basit kod çıkarma (geliştirilmesi gerekebilir)
        try:
            # Agent'ın çıktısından kod parçalarını bul
            import re
            code_blocks = re.findall(r'```python\n(.*?)```', agent_output, re.DOTALL)
            
            if not code_blocks:
                # Alternatif kod arama
                lines = agent_output.split('\n')
                code_lines = [line for line in lines if any(keyword in line for keyword in ['plt.', 'sns.', 'px.', 'fig', 'ax'])]
                if code_lines:
                    code_to_execute = '\n'.join(code_lines)
                    
                    result = execute_agent_code(code_to_execute, df)
                    
                    if result['success']:
                        # Matplotlib figürlerini göster
                        for fig in result['matplotlib_figs']:
                            st.pyplot(fig)
                        
                        # Varsa stdout çıktısını göster
                        if result['stdout'].strip():
                            st.write("**Çıktı:**")
                            st.text(result['stdout'])
                    else:
                        st.warning("Görselleştirme oluşturulamadı. Kod çalıştırma hatası.")
                        st.code(result.get('error', 'Bilinmeyen hata'))
            else:
                # Kod bloklarını çalıştır
                for code_block in code_blocks:
                    result = execute_agent_code(code_block, df)
                    
                    if result['success']:
                        for fig in result['matplotlib_figs']:
                            st.pyplot(fig)
                        
                        if result['stdout'].strip():
                            st.write("**Çıktı:**")
                            st.text(result['stdout'])
                    else:
                        st.error(f"Kod çalıştırma hatası: {result.get('error', 'Bilinmeyen hata')}")
                        
        except Exception as e:
            st.warning(f"Görselleştirme işlenirken hata: {e}")
    
    # Gelişmiş özellikler
    with st.expander("🔧 Gelişmiş Özellikler"):
        st.write("**Ham Agent Yanıtı:**")
        st.json(response if isinstance(response, dict) else {"output": str(response)})
        
        if st.button("🗑️ Sonuçları Temizle"):
            st.session_state.agent_response = None
            st.session_state.generated_plots = []
            st.experimental_rerun()

elif st.session_state.df is None:
    st.info("👆 Başlamak için kenar çubuğundan bir veri dosyası yükleyin.")
    
    # Örnek veri seti önerisi
    with st.expander("📁 Örnek Veri Seti Yükle"):
        if st.button("Örnek Satış Verisi Oluştur"):
            import random
            import datetime
            
            # Örnek veri oluştur
            categories = ['Elektronik', 'Giyim', 'Kitap', 'Ev & Bahçe', 'Spor']
            regions = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya']
            
            data = []
            for i in range(100):
                data.append({
                    'Tarih': datetime.date(2024, random.randint(1, 12), random.randint(1, 28)),
                    'Kategori': random.choice(categories),
                    'Bölge': random.choice(regions),
                    'Satış_Tutarı': random.randint(100, 5000),
                    'Adet': random.randint(1, 20),
                    'Kar': random.randint(50, 1000)
                })
            
            st.session_state.df = pd.DataFrame(data)
            st.success("Örnek veri seti oluşturuldu!")
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("💡 **İpucu:** Daha iyi sonuçlar için sorularınızı net ve spesifik şekilde sorun. Grafik istiyorsanız bunu açıkça belirtin.")

