# app.py (Complete English Version)

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

# Agent import - FIXED DEPRECATION WARNING
from agents.data_analysis_agent import create_pandas_agent

# Streamlit configuration
st.set_page_config(
    page_title="AI Data Analysis Assistant",
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

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🤖 AI Data Analysis Assistant</h1>
    <p style="color: white; margin: 0; opacity: 0.9;">Your intelligent assistant for data analysis, prediction and visualization</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
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

# Simple analysis alternative
def create_simple_analysis(df, question, analysis_type):
    """Simple analysis when agent fails"""
    try:
        question_lower = question.lower()
        
        if analysis_type == "Forecast" or any(word in question_lower for word in ['predict', 'forecast', 'future', 'trend']):
            time_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    time_cols.append(col)
                elif df[col].dtype == 'datetime64[ns]':
                    time_cols.append(col)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if time_cols and numeric_cols:
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
                    
                    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                    
                    future_predictions = []
                    for i in range(1, 7):
                        pred = slope * (len(recent_data) + i) + np.polyfit(x, y, 1)[1]
                        future_predictions.append(pred)
                    
                    return f"""
**📈 TREND ANALYSIS AND PREDICTION:**

**Current Status:**
- Analyzed variable: {num_col}
- Data range: {df_copy[time_col].min().strftime('%Y-%m')} - {df_copy[time_col].max().strftime('%Y-%m')}
- Trend direction: **{trend.upper()}**
- Average value: {recent_data[num_col].mean():.2f}

**Next 6 Months Prediction:**
- Month 1: {future_predictions[0]:.2f}
- Month 2: {future_predictions[1]:.2f}
- Month 3: {future_predictions[2]:.2f}
- Month 4: {future_predictions[3]:.2f}
- Month 5: {future_predictions[4]:.2f}
- Month 6: {future_predictions[5]:.2f}

**Analysis Summary:**
- Recent period shows **{trend}** trend
- 6-month average forecast: {np.mean(future_predictions):.2f}
- Change rate: {((future_predictions[-1] - recent_data[num_col].iloc[-1]) / recent_data[num_col].iloc[-1] * 100):.1f}%
"""
            else:
                return "No suitable time series data found for prediction analysis."
        
        else:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                grouped = df.groupby(cat_col)[num_col].sum()
                top_category = grouped.idxmax()
                
                return f"""
**📊 QUICK ANALYSIS:**

**Question:** "{question}"

**Key Findings:**
- Highest value category: **{top_category}**
- This category's value: {grouped.max():.2f}
- Total categories: {len(grouped)}
- Overall average: {grouped.mean():.2f}

**Data Summary:**
- {len(df)} records analyzed
- {len(categorical_cols)} categorical, {len(numeric_cols)} numeric columns

💡 **You can request charts for more detailed analysis.**
"""
            else:
                return f"""
**ℹ️ Question:** "{question}"

**Dataset Information:**
- {len(df)} rows, {len(df.columns)} columns
- Available columns: {', '.join(df.columns.tolist())}

**💡 Suggestions:**
- "Show chart by categories"
- "Perform trend analysis"  
- "Find highest values"
"""
            
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Advanced code execution function
def execute_agent_code_advanced(code_string, df):
    """Execute agent code with optimized graphics"""
    
    # Matplotlib settings - VERY SMALL SIZES
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (6, 4)  # Much smaller size
    plt.rcParams['font.size'] = 8  # Smaller font
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    
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
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code_string, exec_globals)
        
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

def modify_code_based_on_request(code_string, user_question):
    """Modify code based on user request"""
    
    if any(word in user_question.lower() for word in ['pie', 'distribution']):
        if 'plt.bar(' in code_string or 'ax.bar(' in code_string:
            lines = code_string.split('\n')
            new_lines = []
            
            for line in lines:
                if 'plt.bar(' in line or 'ax.bar(' in line:
                    if 'grouped_data' in code_string:
                        new_lines.append("plt.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%', startangle=90)")
                    else:
                        new_lines.append("# Bar code replaced with pie chart")
                elif 'xlabel' in line or 'ylabel' in line:
                    new_lines.append("# " + line)
                else:
                    new_lines.append(line)
            
            return '\n'.join(new_lines)
    
    elif any(word in user_question.lower() for word in ['line', 'trend']):
        if 'plt.bar(' in code_string:
            code_string = code_string.replace('plt.bar(', 'plt.plot(')
            code_string = code_string.replace(', color=', ', marker="o", linewidth=2, color=')
    
    return code_string

def extract_code_from_response(response_text):
    """Extract code from agent response"""
    code_blocks = []
    
    python_blocks = re.findall(r'```python\n(.*?)```', response_text, re.DOTALL)
    code_blocks.extend(python_blocks)
    
    if not code_blocks:
        general_blocks = re.findall(r'```\n(.*?)```', response_text, re.DOTALL)
        for block in general_blocks:
            if any(keyword in block for keyword in ['plt.', 'sns.', 'px.', 'df[', 'import']):
                code_blocks.append(block)
    
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

def clean_response_from_code(response_text):
    """Clean code blocks from AI response"""
    
    cleaned_text = re.sub(r'```python\n.*?```', '', response_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'```.*?```', '', cleaned_text, flags=re.DOTALL)
    
    lines = cleaned_text.split('\n')
    clean_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if not any(keyword in line for keyword in [
            'import ', 'plt.', 'df[', 'df.', 'pd.', 'np.',
            '=', 'pandas', 'matplotlib', 'seaborn',
            'plt.', 'sns.', 'px.', 'go.', 'fig,', 'ax',
            '.groupby(', '.plot(', '.show()', '.savefig(',
            'pd.DataFrame', 'np.random', 'plt.figure'
        ]):
            if line_stripped and not line_stripped.startswith('#'):
                clean_lines.append(line)
    
    cleaned_text = '\n'.join(clean_lines)
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

# Sidebar - Data loading - CSV/EXCEL ONLY
with st.sidebar:
    st.header("📁 1. Data Source")
    
    uploaded_file = st.file_uploader(
        "Select CSV/Excel File",
        type=['csv', 'xlsx', 'xls'],
        help="You can upload CSV or Excel files"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading file..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            st.markdown('<div class="success-message">✅ File loaded successfully!</div>',
                       unsafe_allow_html=True)
            
            st.subheader("📊 Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            
            with st.expander("🔍 Column Details"):
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    st.write(f"**{col}:** {dtype} ({null_count} missing)")
                    
        except Exception as e:
            st.error(f"❌ File reading error: {e}")
    
    # Sample dataset - SINGLE BUTTON
    st.subheader("📋 Sample Dataset")
    
    if st.button("🚀 Load Sales Data"):
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Cosmetics']
        regions = ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Antalya', 'Adana']
        customers = ['John', 'Sarah', 'Mike', 'Emily', 'David', 'Lisa', 'Alex', 'Emma']  # ENGLISH NAMES
        
        data = []
        for date in dates:
            for _ in range(np.random.randint(1, 5)):
                data.append({
                    'Date': date,
                    'Category': np.random.choice(categories),
                    'Region': np.random.choice(regions),
                    'Customer': np.random.choice(customers),  # ADDED ENGLISH CUSTOMER COLUMN
                    'Sales_Amount': np.random.randint(100, 10000),
                    'Quantity': np.random.randint(1, 50),
                    'Profit': np.random.randint(10, 2000),
                    'Customer_Type': np.random.choice(['Individual', 'Corporate'])
                })
        
        st.session_state.df = pd.DataFrame(data)
        st.success("✅ Sample dataset loaded!")
        st.rerun()

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Data preview
    st.header("👀 2. Data Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_rows = len(df)
    missing_data = df.isnull().sum().sum()
    completeness = ((total_rows * len(df.columns) - missing_data) / (total_rows * len(df.columns))) * 100
    duplicate_count = df.duplicated().sum()
    
    with col1:
        st.metric("📊 Total Rows", f"{total_rows:,}")
    
    with col2:
        st.metric("📋 Columns", f"{len(df.columns)}")
    
    with col3:
        st.metric("✅ Data Integrity", f"{completeness:.1f}%")
    
    with col4:
        st.metric("🔍 Missing Values", f"{missing_data}")
    
    with st.expander("🔧 Detailed Data Info"):
        st.write(f"**Duplicate Records:** {duplicate_count}")
        st.write(f"**Numeric Columns:** {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"**Categorical Columns:** {len(df.select_dtypes(include=['object', 'category']).columns)}")
        if df.select_dtypes(include=['datetime']).columns.tolist():
            st.write(f"**Date Columns:** {', '.join(df.select_dtypes(include=['datetime']).columns.tolist())}")

    # Analysis section
    st.header("🎯 3. Smart Data Analysis")
    
    # Analysis categories - "Forecast" and "Graphics"
    analysis_type = st.selectbox(
        "🔍 Select Analysis Type:",
        [
            "Forecast",  # CHANGED from Prediction
            "Graphics",  # CHANGED from Visualization
            "Statistical Analysis",
            "Trend Analysis"
        ]
    )
    
    # Example questions by category
    if analysis_type == "Forecast":  # CHANGED
        example_questions = [
            "Predict next month's performance for best-selling product",
            "Forecast 2025 sales",
            "Perform trend analysis for next 6 months"
        ]
    elif analysis_type == "Graphics":  # CHANGED
        example_questions = [
            "Show sales distribution by categories with bar chart",
            "Show sales trend over time with line chart",
            "Show performance by regions with pie chart"
        ]
    elif analysis_type == "Statistical Analysis":
        example_questions = [
            "Find best-selling category",
            "Calculate average sales amount",
            "Analyze sales by regions"
        ]
    elif analysis_type == "Trend Analysis":
        example_questions = [
            "Calculate year-over-year growth rate",
            "Find fastest growing categories",
            "Analyze monthly sales trends"
        ]
    
    # Example questions display
    with st.expander(f"💡 {analysis_type} Example Questions"):
        for i, question in enumerate(example_questions, 1):
            if st.button(f"{i}. {question}", key=f"example_{i}"):
                st.session_state.user_question = question
    
    # Question input
    user_question = st.text_area(
        "🤔 Write your question here:",
        value=st.session_state.get('user_question', ''),
        height=120,
        placeholder=f"For example: {example_questions[0]}",
        help="Detailed questions provide better results"
    )
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analyze",
            type="primary",
            use_container_width=True,
            disabled=not user_question.strip()
        )
    
    # Analysis process
    if analyze_button and user_question.strip():
        st.session_state.analysis_history.append({
            'question': user_question,
            'timestamp': datetime.now(),
            'type': analysis_type
        })
        
        with st.spinner("🤖 AI is analyzing and preparing results..."):
            try:
                pandas_agent_executor = create_pandas_agent(df)
                
                # ENGLISH prompt
                enhanced_question = f"""
                Analysis Type: {analysis_type}
                User Question: "{user_question}"
                
                IMPORTANT: ANALYZE THE QUESTION CAREFULLY AND RESPOND IN ENGLISH!
                
                1. QUESTION ANALYSIS:
                   Which column/category does the user want to analyze?
                   - "by regions" → Use Region/City column
                   - "by categories" → Use Category column
                   - "by customers" → Use Customer column
                   - "by products" → Use Product column
                
                2. CHART TYPE ANALYSIS:
                   What chart type does the user want?
                   - "pie chart" → MUST use plt.pie()
                   - "bar chart" → use plt.bar()
                   - "line chart" → use plt.plot()
                
                3. RESPONSE LANGUAGE: ENGLISH ONLY
                
                4. CHART DIMENSIONS (VERY IMPORTANT):
                   - Use plt.figure(figsize=(6, 4)) - VERY SMALL SIZE!
                   - Use plt.xticks(rotation=45, ha='right', fontsize=7) - PREVENT TEXT OVERLAP!
                   - Use plt.tight_layout() (NEVER use plt.show()!)
                   - All font sizes should be very small: fontsize=7-9
                   - Chart title fontsize=9
                
                5. IF PREDICTION/FORECASTING QUERY:
                   - Perform time series analysis
                   - Calculate trend (increasing/decreasing/stable)
                   - Predict future values
                   - Use mathematical modeling
                   - Visualize results
                   - Explain analysis in English
                
                6. IF VISUALIZATION QUERY:
                   - Use the column the user wants
                   - Use the chart type the user wants
                   - CORRECT COLUMN SELECTION IS CRITICAL!
                   - Use plt.tight_layout()
                   - English titles and labels
                
                AVAILABLE DATA COLUMNS:
                {', '.join(df.columns.tolist())}
                
                CRITICAL: 
                - Use the column the user requests
                - Don't select wrong column!
                - Select correct chart type
                - All explanations in English
                - Optimize chart dimensions (6x4)
                """
                
                try:
                    response = pandas_agent_executor.invoke(enhanced_question)
                    st.session_state.agent_response = response
                    
                except Exception as agent_error:
                    if analysis_type == "Forecast":
                        simple_response = create_simple_analysis(df, user_question, analysis_type)
                    elif any(word in user_question.lower() for word in ['chart', 'visual', 'show', 'plot', 'pie', 'bar']):
                        simple_response = "Chart analysis ready..."
                    else:
                        simple_response = create_simple_analysis(df, user_question, "General Analysis")
                    
                    st.session_state.agent_response = {"output": simple_response}
                
            except Exception as e:
                st.error(f"❌ Analysis error: {e}")

# Show results
if st.session_state.agent_response is not None:
    st.header("📊 Analysis Results")
    
    response = st.session_state.agent_response
    
    if isinstance(response, dict):
        agent_output = response.get('output', 'No result found.')
    else:
        agent_output = str(response)
    
    clean_response = clean_response_from_code(agent_output)
    
    if len(clean_response.strip()) < 50:
        lines = agent_output.split('\n')
        clean_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if not in_code_block:
                if not any(keyword in line for keyword in ['import ', 'plt.', 'df[', 'df.', '=', 'pandas', 'matplotlib']):
                    if line.strip():
                        clean_lines.append(line)
        
        clean_response = '\n'.join(clean_lines).strip()
    
    st.subheader("🤖 AI Assistant's Response")
    if clean_response:
        st.write(clean_response)
    else:
        st.write("Analysis completed. Chart shown below.")
    
    # Direct chart system
    if any(word in user_question.lower() for word in ['chart', 'visual', 'show', 'plot', 'pie', 'bar', 'distribution']):
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_cat_col = None
        target_num_col = None
        
        query_lower = user_question.lower()
        
        # Column selection
        if 'region' in query_lower or 'city' in query_lower:
            for col in categorical_cols:
                if any(word in col.lower() for word in ['region', 'city']):
                    target_cat_col = col
                    break
        elif 'categor' in query_lower:
            for col in categorical_cols:
                if 'categor' in col.lower():
                    target_cat_col = col
                    break
        elif 'customer' in query_lower:
            for col in categorical_cols:
                if 'customer' in col.lower():
                    target_cat_col = col
                    break
        elif 'product' in query_lower:
            for col in categorical_cols:
                if 'product' in col.lower():
                    target_cat_col = col
                    break
        
        if 'sales' in query_lower or 'amount' in query_lower:
            for col in numeric_cols:
                if any(word in col.lower() for word in ['sales', 'amount', 'revenue']):
                    target_num_col = col
                    break
        elif 'quantity' in query_lower:
            for col in numeric_cols:
                if any(word in col.lower() for word in ['quantity', 'qty']):
                    target_num_col = col
                    break
        elif 'profit' in query_lower:
            for col in numeric_cols:
                if 'profit' in col.lower():
                    target_num_col = col
                    break
        
        if not target_cat_col and categorical_cols:
            target_cat_col = categorical_cols[0]
        if not target_num_col and numeric_cols:
            target_num_col = numeric_cols[0]
        
        # Create chart - OPTIMIZED DIMENSIONS
        if target_cat_col and target_num_col:
            st.subheader("📈 Graphics")  # CHANGED
            
            try:
                plt.figure(figsize=(6, 4))  # VERY SMALL SIZE
                plt.rcParams['font.size'] = 8
                
                data_grouped = df.groupby(target_cat_col)[target_num_col].sum()
                
                if any(word in query_lower for word in ['pie', 'distribution']):
                    # PIE CHART - OPTIMIZED
                    colors = plt.cm.Set3(range(len(data_grouped)))
                    plt.pie(data_grouped.values, labels=data_grouped.index, autopct='%1.1f%%', 
                           startangle=90, colors=colors, textprops={'fontsize': 7})
                    plt.title(f'{target_cat_col} vs {target_num_col} Distribution', fontsize=9, pad=10)
                    
                elif any(word in query_lower for word in ['line', 'trend']):
                    # LINE CHART - OPTIMIZED
                    plt.plot(data_grouped.index, data_grouped.values, marker='o', linewidth=2, markersize=4, color='blue')
                    plt.title(f'{target_cat_col} vs {target_num_col} Trend', fontsize=9)
                    plt.xlabel(target_cat_col, fontsize=8)
                    plt.ylabel(target_num_col, fontsize=8)
                    plt.xticks(rotation=45, ha='right', fontsize=7)
                    plt.yticks(fontsize=7)
                    plt.grid(True, alpha=0.3)
                    
                else:
                    # BAR CHART - OPTIMIZED
                    colors = plt.cm.viridis(range(len(data_grouped)))
                    plt.bar(data_grouped.index, data_grouped.values, color=colors, alpha=0.8)
                    plt.title(f'{target_cat_col} vs {target_num_col} Distribution', fontsize=9)
                    plt.xlabel(target_cat_col, fontsize=8)
                    plt.ylabel(target_num_col, fontsize=8)
                    plt.xticks(rotation=45, ha='right', fontsize=7)
                    plt.yticks(fontsize=7)
                
                plt.tight_layout()  # PREVENT TEXT OVERLAP
                st.pyplot(plt, use_container_width=True)
                
                # Analysis summary
                st.write(f"**📊 Analysis Summary:**")
                st.write(f"- **Selected category:** {target_cat_col}")
                st.write(f"- **Analyzed value:** {target_num_col}")
                st.write(f"- **Highest value:** {data_grouped.idxmax()} ({data_grouped.max():,.2f})")
                st.write(f"- **Total value:** {data_grouped.sum():,.2f}")
                st.write(f"- **Number of categories:** {len(data_grouped)}")
                
            except Exception as e:
                st.error(f"Chart creation error: {e}")
    
    # Code extraction and execution
    code_blocks = extract_code_from_response(agent_output)
    
    if code_blocks and not any(word in user_question.lower() for word in ['chart', 'visual', 'show', 'plot', 'pie', 'bar']):
        st.subheader("📈 Additional Graphics")  # CHANGED
        
        for i, code_block in enumerate(code_blocks):
            modified_code = modify_code_based_on_request(code_block, user_question)
            result = execute_agent_code_advanced(modified_code, df)
            
            if result['success']:
                for fig in result['matplotlib_figs']:
                    st.pyplot(fig, use_container_width=True)
                break
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Results"):
                st.session_state.agent_response = None
                st.session_state.generated_plots = []
                st.rerun()
        
        with col2:
            if st.button("📥 Download Results"):
                st.download_button(
                    label="📄 Download Analysis Report",
                    data=clean_response,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )

# Sidebar - Analysis history
if st.session_state.analysis_history:
    with st.sidebar:
        st.header("📝 Analysis History")
        
        for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):  # Last 5 analyses
            with st.expander(f"{item['type']} - {item['timestamp'].strftime('%H:%M')}"):
                st.write(f"**Question:** {item['question'][:100]}...")
                if st.button(f"🔄 Re-run", key=f"rerun_{i}"):
                    st.session_state.user_question = item['question']
                    st.rerun()

elif st.session_state.df is None:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome!</h2>
        <p style="font-size: 1.2rem; color: #666;">
            To get started, upload a data file from the left panel
            or try the sample dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features introduction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Smart Analysis
        - Ask questions in natural language
        - Automatic data discovery
        - Statistical analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Prediction Models
        - Time series analysis
        - Trend forecasting
        - Future period projections
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Visualization
        - Automatic chart generation
        - Interactive charts
        - Customizable views
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>💡 Usage Tips:</strong></p>
    <p>• For predictions use keywords like "future", "forecast", "2025"</p>
    <p>• For charts use "show", "plot", "visualize"</p>
    <p>• Detailed questions provide better results</p>
</div>
""", unsafe_allow_html=True)