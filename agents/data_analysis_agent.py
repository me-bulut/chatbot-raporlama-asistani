# agents/data_analysis_agent.py (Geliştirilmiş Versiyon)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# Yapılandırma
load_dotenv()

class AdvancedDataAnalysisAgent:
    """Gelişmiş veri analizi ve tahmin özellikleri olan agent"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
    def _initialize_llm(self):
        """LLM'i başlat"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı!")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,  # Tahmin için biraz daha yaratıcı
            max_tokens=4000
        )
    
    def _detect_time_column(self):
        """Zaman sütununu otomatik tespit et"""
        time_columns = []
        for col in self.df.columns:
            if self.df[col].dtype == 'datetime64[ns]':
                time_columns.append(col)
            elif 'tarih' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(self.df[col])
                    time_columns.append(col)
                except:
                    pass
        return time_columns[0] if time_columns else None
    
    def _detect_numeric_columns(self):
        """Sayısal sütunları tespit et"""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _detect_categorical_columns(self):
        """Kategorik sütunları tespit et"""
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def create_prediction_tool(self):
        """Tahmin aracı oluştur"""
        def predict_values(query: str) -> str:
            try:
                # Basit trend analizi ve tahmin
                time_col = self._detect_time_column()
                numeric_cols = self._detect_numeric_columns()
                
                if not time_col or not numeric_cols:
                    return "Tahmin için zaman serisi verisi bulunamadı."
                
                # Zaman sütununu datetime'a çevir
                df_copy = self.df.copy()
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
                df_copy = df_copy.sort_values(time_col)
                
                # Her sayısal sütun için basit linear regression
                predictions = {}
                
                for col in numeric_cols:
                    if col in df_copy.columns:
                        # Son 12 ay veya mevcut verinin %80'i
                        recent_data = df_copy.tail(min(12, int(len(df_copy) * 0.8)))
                        
                        if len(recent_data) >= 3:
                            # Basit trend hesaplama
                            x = np.arange(len(recent_data))
                            y = recent_data[col].values
                            
                            # Linear regression katsayıları
                            slope = np.polyfit(x, y, 1)[0]
                            intercept = np.polyfit(x, y, 1)[1]
                            
                            # Gelecek dönem tahmini
                            next_period = len(recent_data)
                            prediction = slope * next_period + intercept
                            
                            # Trend analizi
                            if slope > 0:
                                trend = "artan"
                            elif slope < 0:
                                trend = "azalan"
                            else:
                                trend = "sabit"
                            
                            predictions[col] = {
                                'prediction': prediction,
                                'trend': trend,
                                'slope': slope,
                                'current_avg': recent_data[col].mean(),
                                'growth_rate': (slope / recent_data[col].mean() * 100) if recent_data[col].mean() != 0 else 0
                            }
                
                # Sonuçları formatla
                result = "📈 TAHMİN ANALİZİ:\n\n"
                for col, pred in predictions.items():
                    result += f"**{col}:**\n"
                    result += f"- Gelecek dönem tahmini: {pred['prediction']:.2f}\n"
                    result += f"- Trend: {pred['trend']} ({pred['growth_rate']:.1f}% büyüme oranı)\n"
                    result += f"- Mevcut ortalama: {pred['current_avg']:.2f}\n\n"
                
                return result
                
            except Exception as e:
                return f"Tahmin hesaplanırken hata oluştu: {str(e)}"
        
        return Tool(
            name="prediction_tool",
            description="Zaman serisi verilerinden gelecek dönem tahminleri yapar",
            func=predict_values
        )
    
    def create_visualization_tool(self):
        """Görselleştirme aracı oluştur"""
        def create_charts(query: str) -> str:
            try:
                # Matplotlib ayarları
                plt.style.use('default')
                plt.rcParams['figure.figsize'] = (12, 8)
                plt.rcParams['font.size'] = 10
                
                # Grafik türünü belirle
                query_lower = query.lower()
                
                if any(word in query_lower for word in ['bar', 'çubuk', 'sütun']):
                    return self._create_bar_chart(query)
                elif any(word in query_lower for word in ['line', 'çizgi', 'trend']):
                    return self._create_line_chart(query)
                elif any(word in query_lower for word in ['pie', 'pasta', 'dağılım']):
                    return self._create_pie_chart(query)
                elif any(word in query_lower for word in ['scatter', 'nokta', 'korelasyon']):
                    return self._create_scatter_plot(query)
                elif any(word in query_lower for word in ['tahmin', 'gelecek', 'forecast']):
                    return self._create_prediction_chart(query)
                else:
                    # Varsayılan olarak en uygun grafiği seç
                    return self._create_auto_chart(query)
                    
            except Exception as e:
                return f"Grafik oluşturulurken hata: {str(e)}"
        
        return Tool(
            name="visualization_tool",
            description="Veriler için uygun grafikleri ve görselleştirmeleri oluşturur",
            func=create_charts
        )
    
    def _create_bar_chart(self, query):
        """Bar chart oluştur"""
        categorical_cols = self._detect_categorical_columns()
        numeric_cols = self._detect_numeric_columns()
        
        if not categorical_cols or not numeric_cols:
            return "Bar chart için uygun veri bulunamadı."
        
        # İlk kategorik ve sayısal sütunu kullan
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        # Verileri grupla
        grouped_data = self.df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
        
        # Grafik oluştur
        plt.figure(figsize=(12, 8))
        bars = plt.bar(grouped_data.index, grouped_data.values)
        plt.title(f'{cat_col} Bazında {num_col} Dağılımı', fontsize=16, fontweight='bold')
        plt.xlabel(cat_col, fontsize=12)
        plt.ylabel(num_col, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Değerleri bar'ların üstüne yaz
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        return f"Bar chart oluşturuldu: {cat_col} vs {num_col}"
    
    def _create_line_chart(self, query):
        """Line chart oluştur"""
        time_col = self._detect_time_column()
        numeric_cols = self._detect_numeric_columns()
        
        if not time_col or not numeric_cols:
            return "Trend grafiği için zaman serisi verisi bulunamadı."
        
        # Zaman sütununu datetime'a çevir
        df_copy = self.df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        df_copy = df_copy.sort_values(time_col)
        
        plt.figure(figsize=(14, 8))
        
        # İlk sayısal sütun için çizgi grafiği
        num_col = numeric_cols[0]
        plt.plot(df_copy[time_col], df_copy[num_col], marker='o', linewidth=2, markersize=6)
        
        plt.title(f'{num_col} Zaman İçinde Değişim', fontsize=16, fontweight='bold')
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel(num_col, fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return f"Trend grafiği oluşturuldu: {time_col} vs {num_col}"
    
    def _create_prediction_chart(self, query):
        """Tahmin grafiği oluştur"""
        time_col = self._detect_time_column()
        numeric_cols = self._detect_numeric_columns()
        
        if not time_col or not numeric_cols:
            return "Tahmin grafiği için zaman serisi verisi bulunamadı."
        
        df_copy = self.df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        df_copy = df_copy.sort_values(time_col)
        
        num_col = numeric_cols[0]
        
        plt.figure(figsize=(14, 8))
        
        # Mevcut veri
        plt.plot(df_copy[time_col], df_copy[num_col], 
                marker='o', label='Gerçek Veriler', linewidth=2)
        
        # Basit tahmin (son 6 ayın ortalaması ile trend)
        recent_data = df_copy.tail(6)
        if len(recent_data) >= 3:
            x = np.arange(len(recent_data))
            y = recent_data[num_col].values
            slope, intercept = np.polyfit(x, y, 1)
            
            # Gelecek 3 dönem tahmini
            future_periods = 3
            last_date = df_copy[time_col].max()
            
            future_dates = []
            future_values = []
            
            for i in range(1, future_periods + 1):
                future_date = last_date + timedelta(days=30*i)  # Aylık tahmin
                future_value = slope * (len(recent_data) + i) + intercept
                future_dates.append(future_date)
                future_values.append(future_value)
            
            plt.plot(future_dates, future_values, 
                    marker='s', linestyle='--', color='red', 
                    label='Tahmin', linewidth=2)
        
        plt.title(f'{num_col} - Mevcut Veriler ve Gelecek Tahminler', fontsize=16, fontweight='bold')
        plt.xlabel('Tarih', fontsize=12)
        plt.ylabel(num_col, fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return f"Tahmin grafiği oluşturuldu: {num_col} için gelecek trend"
    
    def _create_auto_chart(self, query):
        """Otomatik uygun grafik seç"""
        categorical_cols = self._detect_categorical_columns()
        numeric_cols = self._detect_numeric_columns()
        time_col = self._detect_time_column()
        
        if time_col and numeric_cols:
            return self._create_line_chart(query)
        elif categorical_cols and numeric_cols:
            return self._create_bar_chart(query)
        else:
            return "Uygun grafik türü belirlenemedi."
    
    def create_agent(self):
        """Geliştirilmiş agent oluştur"""
        
        # Özel promptlar
        system_prompt = """
        Sen gelişmiş bir veri analiz uzmanısın. Aşağıdaki yeteneklerin var:
        
        1. TEMEL ANALİZ: Veri özetleme, istatistikler, gruplamalar
        2. TAHMİN: Zaman serisi analizi, trend tahminleri, gelecek dönem hesaplamaları
        3. GÖRSELLEŞTİRME: Bar, line, pie, scatter grafikleri
        
        TAHMİN SORULARI İÇİN:
        - Önce veriyi zaman serisine göre analiz et
        - Trend hesapla (artan/azalan/sabit)
        - Matematiksel tahmin yap
        - prediction_tool'u kullan
        - Sonucu görselleştir
        
        GRAFİK SORULARI İÇİN:
        - visualization_tool'u kullan
        - Uygun grafik türünü seç
        - Grafik oluştur ve açıkla
        
        Her zaman Türkçe yanıt ver ve sonuçları detaylı açıkla.
        """
        
        # Araçları hazırla
        tools = [
            self.create_prediction_tool(),
            self.create_visualization_tool()
        ]
        
        # Ana pandas agent'ı oluştur
        agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            extra_tools=tools,
            agent_kwargs={
                'system_message': system_prompt,
                'memory': self.memory
            }
        )
        
        return agent

def create_pandas_agent(df: pd.DataFrame):
    """Ana agent oluşturma fonksiyonu"""
    try:
        advanced_agent = AdvancedDataAnalysisAgent(df)
        return advanced_agent.create_agent()
    except Exception as e:
        # Fallback basit agent
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı!")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        return create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )

