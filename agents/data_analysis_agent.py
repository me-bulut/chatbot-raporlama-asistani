# agents/data_analysis_agent.py (Complete English Version)

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

load_dotenv()

class AdvancedDataAnalysisAgent:
    """Advanced data analysis and prediction agent"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
    def _initialize_llm(self):
        """Initialize LLM"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found!")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=4000
        )
    
    def _detect_time_column(self):
        """Detect time column"""
        time_columns = []
        for col in self.df.columns:
            if self.df[col].dtype == 'datetime64[ns]':
                time_columns.append(col)
            elif 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(self.df[col])
                    time_columns.append(col)
                except:
                    pass
        return time_columns[0] if time_columns else None
    
    def _detect_numeric_columns(self):
        """Detect numeric columns"""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def _detect_categorical_columns(self):
        """Detect categorical columns"""
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def create_prediction_tool(self):
        """Create prediction tool"""
        def predict_values(query: str) -> str:
            try:
                time_col = self._detect_time_column()
                numeric_cols = self._detect_numeric_columns()
                
                if not time_col or not numeric_cols:
                    return "No time series data found for prediction."
                
                df_copy = self.df.copy()
                df_copy[time_col] = pd.to_datetime(df_copy[time_col])
                df_copy = df_copy.sort_values(time_col)
                
                predictions = {}
                
                for col in numeric_cols:
                    if col in df_copy.columns:
                        recent_data = df_copy.tail(min(12, int(len(df_copy) * 0.8)))
                        
                        if len(recent_data) >= 3:
                            x = np.arange(len(recent_data))
                            y = recent_data[col].values
                            
                            slope = np.polyfit(x, y, 1)[0]
                            intercept = np.polyfit(x, y, 1)[1]
                            
                            next_period = len(recent_data)
                            prediction = slope * next_period + intercept
                            
                            if slope > 0:
                                trend = "increasing"
                            elif slope < 0:
                                trend = "decreasing"
                            else:
                                trend = "stable"
                            
                            predictions[col] = {
                                'prediction': prediction,
                                'trend': trend,
                                'slope': slope,
                                'current_avg': recent_data[col].mean(),
                                'growth_rate': (slope / recent_data[col].mean() * 100) if recent_data[col].mean() != 0 else 0
                            }
                
                result = "📈 FORECAST ANALYSIS:\n\n"
                for col, pred in predictions.items():
                    result += f"**{col}:**\n"
                    result += f"- Next period prediction: {pred['prediction']:.2f}\n"
                    result += f"- Trend: {pred['trend']} ({pred['growth_rate']:.1f}% growth rate)\n"
                    result += f"- Current average: {pred['current_avg']:.2f}\n\n"
                
                return result
                
            except Exception as e:
                return f"Error calculating prediction: {str(e)}"
        
        return Tool(
            name="prediction_tool",
            description="Makes future period predictions from time series data",
            func=predict_values
        )
    
    def create_visualization_tool(self):
        """Create visualization tool - OPTIMIZED DIMENSIONS"""
        def create_charts(query: str) -> str:
            try:
                # Matplotlib settings - VERY SMALL CHART DIMENSIONS
                plt.style.use('default')
                plt.rcParams['figure.figsize'] = (6, 4)  # VERY SMALL SIZE
                plt.rcParams['font.size'] = 8
                plt.rcParams['axes.titlesize'] = 9
                plt.rcParams['axes.labelsize'] = 8
                plt.rcParams['xtick.labelsize'] = 7
                plt.rcParams['ytick.labelsize'] = 7
                
                query_lower = query.lower()
                
                if any(word in query_lower for word in ['pie', 'distribution']):
                    return self._create_pie_chart(query)
                elif any(word in query_lower for word in ['line', 'trend', 'time']):
                    return self._create_line_chart(query)
                elif any(word in query_lower for word in ['scatter', 'correlation']):
                    return self._create_scatter_plot(query)
                elif any(word in query_lower for word in ['prediction', 'forecast']):
                    return self._create_prediction_chart(query)
                elif any(word in query_lower for word in ['bar', 'column']):
                    return self._create_bar_chart(query)
                else:
                    return self._create_auto_chart(query)
                    
            except Exception as e:
                return f"Chart creation error: {str(e)}"
        
        return Tool(
            name="visualization_tool",
            description="Creates appropriate charts for data",
            func=create_charts
        )
    
    def _create_bar_chart(self, query):
        """Create bar chart - OPTIMIZED DIMENSIONS"""
        categorical_cols = self._detect_categorical_columns()
        numeric_cols = self._detect_numeric_columns()
        
        if not categorical_cols or not numeric_cols:
            return "No suitable data found for bar chart."
        
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        return f"""
```python
import matplotlib.pyplot as plt
import pandas as pd

# Create bar chart - OPTIMIZED SIZE
cat_col = '{cat_col}'
num_col = '{num_col}'

grouped_data = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)

plt.figure(figsize=(6, 4))  # VERY SMALL SIZE
bars = plt.bar(grouped_data.index, grouped_data.values, color='skyblue', edgecolor='navy', alpha=0.7)
plt.title(f'{cat_col} vs {num_col} Distribution', fontsize=9, fontweight='bold')
plt.xlabel(cat_col, fontsize=8)
plt.ylabel(num_col, fontsize=8)
plt.xticks(rotation=45, ha='right', fontsize=7)  # PREVENT TEXT OVERLAP
plt.yticks(fontsize=7)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{{height:.0f}}', ha='center', va='bottom', fontsize=7)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()  # PREVENT TEXT OVERLAP
```
"""
    
    def _create_line_chart(self, query):
        """Create line chart - OPTIMIZED DIMENSIONS"""
        time_col = self._detect_time_column()
        numeric_cols = self._detect_numeric_columns()
        
        if not time_col or not numeric_cols:
            return "No time series data found for trend chart."
        
        num_col = numeric_cols[0]
        
        return f"""
```python
import matplotlib.pyplot as plt
import pandas as pd

# Convert time column to datetime
df_copy = df.copy()
df_copy['{time_col}'] = pd.to_datetime(df_copy['{time_col}'])
df_copy = df_copy.sort_values('{time_col}')

plt.figure(figsize=(6, 4))  # VERY SMALL SIZE
plt.plot(df_copy['{time_col}'], df_copy['{num_col}'], marker='o', linewidth=2, markersize=3, color='blue', alpha=0.7)
plt.title('{num_col} Change Over Time', fontsize=9, fontweight='bold')
plt.xlabel('Date', fontsize=8)
plt.ylabel('{num_col}', fontsize=8)
plt.xticks(rotation=45, ha='right', fontsize=7)  # PREVENT TEXT OVERLAP
plt.yticks(fontsize=7)
plt.grid(True, alpha=0.3)
plt.tight_layout()  # PREVENT TEXT OVERLAP
```
"""
    
    def _create_pie_chart(self, query):
        """Create pie chart - OPTIMIZED DIMENSIONS"""
        categorical_cols = self._detect_categorical_columns()
        numeric_cols = self._detect_numeric_columns()
        
        if not categorical_cols or not numeric_cols:
            return "No suitable data found for pie chart."
        
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        
        return f"""
```python
import matplotlib.pyplot as plt
import pandas as pd

# Create pie chart - OPTIMIZED SIZE
cat_col = '{cat_col}'
num_col = '{num_col}'

grouped_data = df.groupby(cat_col)[num_col].sum()

plt.figure(figsize=(6, 4))  # VERY SMALL SIZE
colors = plt.cm.Set3(range(len(grouped_data)))
plt.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%', 
        startangle=90, colors=colors, textprops={{'fontsize': 7}})  # SMALL FONT
plt.title(f'{cat_col} vs {num_col} Distribution', fontsize=9, fontweight='bold', pad=10)
plt.tight_layout()  # PREVENT TEXT OVERLAP
```
"""
    
    def _create_prediction_chart(self, query):
        """Create prediction chart - OPTIMIZED DIMENSIONS"""
        time_col = self._detect_time_column()
        numeric_cols = self._detect_numeric_columns()
        
        if not time_col or not numeric_cols:
            return "No time series data found for prediction chart."
        
        num_col = numeric_cols[0]
        
        return f"""
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta

# Prepare data
df_copy = df.copy()
df_copy['{time_col}'] = pd.to_datetime(df_copy['{time_col}'])
df_copy = df_copy.sort_values('{time_col}')

plt.figure(figsize=(6, 4))  # VERY SMALL SIZE

# Current data
plt.plot(df_copy['{time_col}'], df_copy['{num_col}'], 
         marker='o', label='Actual Data', linewidth=2, color='blue', markersize=3)

# Simple prediction
recent_data = df_copy.tail(6)
if len(recent_data) >= 3:
    x = np.arange(len(recent_data))
    y = recent_data['{num_col}'].values
    slope, intercept = np.polyfit(x, y, 1)
    
    # Next 3 periods prediction
    future_periods = 3
    last_date = df_copy['{time_col}'].max()
    
    future_dates = []
    future_values = []
    
    for i in range(1, future_periods + 1):
        future_date = last_date + timedelta(days=30*i)
        future_value = slope * (len(recent_data) + i) + intercept
        future_dates.append(future_date)
        future_values.append(future_value)
    
    plt.plot(future_dates, future_values,
            marker='s', linestyle='--', color='red',
            label='Prediction', linewidth=2, markersize=3)

plt.title('{num_col} - Current Data and Future Predictions', fontsize=9, fontweight='bold')
plt.xlabel('Date', fontsize=8)
plt.ylabel('{num_col}', fontsize=8)
plt.legend(fontsize=7)
plt.xticks(rotation=45, ha='right', fontsize=7)  # PREVENT TEXT OVERLAP
plt.yticks(fontsize=7)
plt.grid(True, alpha=0.3)
plt.tight_layout()  # PREVENT TEXT OVERLAP
```
"""
    
    def _create_scatter_plot(self, query):
        """Create scatter plot - OPTIMIZED DIMENSIONS"""
        numeric_cols = self._detect_numeric_columns()
        
        if len(numeric_cols) < 2:
            return "At least 2 numeric columns required for scatter plot."
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        return f"""
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))  # VERY SMALL SIZE
plt.scatter(df['{x_col}'], df['{y_col}'], alpha=0.6, color='blue', s=20)
plt.title('{x_col} vs {y_col} Relationship', fontsize=9, fontweight='bold')
plt.xlabel('{x_col}', fontsize=8)
plt.ylabel('{y_col}', fontsize=8)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.grid(True, alpha=0.3)
plt.tight_layout()  # PREVENT TEXT OVERLAP
```
"""
    
    def _create_auto_chart(self, query):
        """Auto select chart"""
        categorical_cols = self._detect_categorical_columns()
        numeric_cols = self._detect_numeric_columns()
        time_col = self._detect_time_column()
        
        if time_col and numeric_cols:
            return self._create_line_chart(query)
        elif categorical_cols and numeric_cols:
            return self._create_bar_chart(query)
        else:
            return "Unable to determine suitable chart type."
    
    def create_agent(self):
        """Create enhanced agent - ENGLISH FOCUSED"""
        
        system_prompt = """
        You are an advanced data analysis expert and you MUST ALWAYS respond in ENGLISH. You have these capabilities:
        
        1. BASIC ANALYSIS: Data summarization, statistics, groupings
        2. FORECAST: Time series analysis, trend predictions, future period calculations
        3. GRAPHICS: Bar, line, pie, scatter charts
        
        VERY IMPORTANT - UNDERSTAND USER QUESTION AND ALWAYS RESPOND IN ENGLISH:
        
        1. QUESTION ANALYSIS:
           - Carefully determine which COLUMN the user wants
           - "by regions" = Use Region column
           - "by categories" = Use Category column  
           - "by customers" = Use Customer column
           - "by cities" = Use City column
        
        2. CHART TYPE ANALYSIS:
           - "pie chart" = MUST use plt.pie()
           - "bar chart" = use plt.bar()
           - "line chart" = use plt.plot()
           - "scatter" = use plt.scatter()
        
        3. COLUMN SELECTION RULES:
           - Use the exact column the user asks for
           - Never select wrong column!
           - If unsure, re-read the question carefully
        
        4. CHART DIMENSIONS (VERY IMPORTANT):
           - Use plt.figure(figsize=(6, 4)) - VERY SMALL SIZE for compact display!
           - Use plt.xticks(rotation=45, ha='right', fontsize=7) - PREVENT TEXT OVERLAP!
           - Use plt.tight_layout() - FOR PROPER LAYOUT!
           - All fontsize values very small: 7-9 range
           - Chart titles fontsize=9
           - NEVER use plt.show()!
        
        5. FOR FORECAST QUESTIONS:
           - Perform time series analysis
           - Calculate trend (increasing/decreasing/stable)
           - Predict future values
           - Use mathematical modeling
           - Use prediction_tool
           - Visualize results
           - Explain everything in English
        
        6. FOR GRAPHICS QUESTIONS:
           - Use visualization_tool
           - Use the exact CHART TYPE the user wants
           - Create Python code (use plt.tight_layout(), never plt.show())
           - Use the exact COLUMN the user wants
           - All chart titles and labels must be in English
        
        7. RESPONSE LANGUAGE: ALWAYS ENGLISH
           - Every explanation must be in English
           - Chart titles must be in English
           - All results must be in English
           - Never use any other language
        
        CRITICAL REQUIREMENTS: 
        - Analyze user's question WORD BY WORD
        - If you select wrong column, the entire analysis will be wrong
        - In chart codes NEVER use plt.show(), only use plt.tight_layout()!
        - Optimize chart dimensions (6x4) to fit page
        - MANDATORY: ALL RESPONSES IN ENGLISH ONLY
        
        You must always respond in English and explain all results in detail using English.
        """
        
        tools = [
            self.create_prediction_tool(),
            self.create_visualization_tool()
        ]
        
        agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            extra_tools=tools,
            max_iterations=3,
            agent_kwargs={
                'system_message': system_prompt
            }
        )
        
        return agent

def create_pandas_agent(df: pd.DataFrame):
    """Main agent creation function"""
    try:
        advanced_agent = AdvancedDataAnalysisAgent(df)
        return advanced_agent.create_agent()
    except Exception as e:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found!")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        return create_pandas_dataframe_agent(
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            allow_dangerous_code=True,
            max_iterations=3
        )