# agents/data_analysis_agent.py

import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# --- 1. Yapılandırma ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY .env dosyasında bulunamadı.")


# --- 2. YENİ ve DAHA GÜÇLÜ AGENT OLUŞTURMA FONKSİYONU ---

def create_pandas_agent(df: pd.DataFrame):
    """
    LangChain'in bu iş için özel olarak optimize edilmiş, hazır 
    Pandas DataFrame Agent'ını oluşturur.
    """
    # LLM'i (beyni) tanımlıyoruz
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Bu, LangChain'in bu iş için özel olarak tasarladığı, hazır ve güçlü agent'tır.
    # Karmaşık prompt'ları ve araçları bizim yerimize kendisi yönetir.
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="zero-shot-react-description", # Standart ReAct ajanı
        verbose=True,                             # Agent'ın düşüncelerini terminalde göster
        handle_parsing_errors=True,               # Formatlama hatalarını kendisi düzeltsin
        allow_dangerous_code=True                 # Bu agent'ın Python kodu çalıştırmasına izin ver
    )
    
    return agent_executor
