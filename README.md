🤖 AI Data Analysis Assistant
An intelligent Streamlit application that provides advanced data analysis, forecasting, and visualization capabilities using AI-powered agents.
🌟 Features
📊 Smart Data Analysis

Natural language query processing
Automatic data discovery and column detection
Statistical analysis and data summarization
Real-time data insights

📈 Advanced Forecasting

Time series analysis and trend detection
Future period predictions using mathematical modeling
Growth rate calculations
Trend visualization (increasing/decreasing/stable)

📉 Interactive Visualizations

Automatic chart type selection
Multiple chart types: Bar, Line, Pie, Scatter plots
Optimized chart dimensions for web display
Interactive Plotly integration

🔧 Data Processing

CSV and Excel file support
Sample dataset generation
Data quality assessment
Missing value analysis

🏗️ Project Structure
CHATBOT-RAPORLAMA-ASISTANI/
├── .devcontainer/          # Development container configuration
├── .vscode/               # VS Code settings
├── agents/                # AI agent modules
│   ├── __pycache__/      # Python cache files
│   └── data_analysis_agent.py  # Main analysis agent
├── features/              # Feature modules
│   ├── __pycache__/      # Python cache files
│   └── gemini_query.py   # Gemini AI integration
├── static/                # Static files
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── app.py                # Main Streamlit application
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
🚀 Installation
Prerequisites

Python 3.8 or higher
Google API Key for Gemini AI

Setup Steps

Clone the repository

bashgit clone <repository-url>
cd CHATBOT-RAPORLAMA-ASISTANI

Create virtual environment

bashpython -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Install dependencies

bashpip install -r requirements.txt

Environment Configuration
Create a .env file in the root directory:

envGOOGLE_API_KEY=your_google_api_key_here

Run the application

bashstreamlit run app.py
📦 Dependencies
txtstreamlit>=1.28.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
numpy>=1.24.0
langchain>=0.0.350
langchain-experimental>=0.0.50
langchain-google-genai>=1.0.0
python-dotenv>=1.0.0
openpyxl>=3.1.0
🔧 Configuration
Google API Setup

Visit Google AI Studio
Create a new API key
Add it to your .env file as GOOGLE_API_KEY

Chart Optimization
The application uses optimized chart dimensions:

Figure size: 6x4 inches
Font sizes: 7-9 pt for compact display
Automatic layout adjustment to prevent text overlap

🎯 Usage
1. Data Upload

Supported formats: CSV, Excel (.xlsx, .xls)
Use the sidebar file uploader
Or try the sample sales dataset

2. Analysis Types
Forecast Analysis
Ask questions like:

"Predict next month's performance"
"Forecast 2025 sales"
"Perform trend analysis for next 6 months"

Graphics/Visualization
Request charts like:

"Show sales distribution by categories with bar chart"
"Show sales trend over time with line chart"
"Show performance by regions with pie chart"

Statistical Analysis
Query data insights:

"Find best-selling category"
"Calculate average sales amount"
"Analyze sales by regions"

3. Natural Language Queries
The AI agent understands natural language and automatically:

Detects relevant columns based on your question
Selects appropriate chart types
Performs statistical calculations
Generates predictions

🧠 AI Agent Architecture
Core Components
AdvancedDataAnalysisAgent

Prediction Tool: Time series forecasting with trend analysis
Visualization Tool: Automatic chart generation
Column Detection: Smart identification of time, numeric, and categorical columns

Features

Multi-language Support: Responses in English
Error Handling: Graceful fallback to simple analysis
Memory: Conversation history tracking
Code Execution: Safe Python code execution for charts

Agent Capabilities

Data Understanding: Automatic column type detection
Query Processing: Natural language to SQL-like operations
Prediction Modeling: Linear trend analysis and forecasting
Visualization: Context-aware chart selection
Response Generation: Structured analysis reports

📊 Sample Outputs
Forecast Analysis Example
📈 FORECAST ANALYSIS:

**Sales_Amount:**
- Next period prediction: 15,430.25
- Trend: increasing (12.5% growth rate)
- Current average: 13,245.80

**Quantity:**
- Next period prediction: 87.50
- Trend: stable (2.1% growth rate)
- Current average: 85.20
Statistical Summary Example
📊 Analysis Summary:
- Selected category: Category
- Analyzed value: Sales_Amount
- Highest value: Electronics (125,430.50)
- Total value: 1,250,340.25
- Number of categories: 6
🔍 Advanced Features
Data Quality Assessment

Missing value detection
Duplicate record identification
Data completeness percentage
Column type analysis

Interactive Elements

Real-time chart updates
Analysis history tracking
Result downloading
Advanced options panel

Performance Optimization

Efficient data processing
Optimized chart rendering
Memory management
Error recovery mechanisms

🐛 Troubleshooting
Common Issues
1. Google API Key Error
Error: GOOGLE_API_KEY not found!
Solution: Ensure your .env file contains the correct API key.
2. Chart Display Issues

Charts too large: Application automatically uses optimized 6x4 dimensions
Text overlap: Uses plt.tight_layout() and rotated labels

3. Data Loading Errors

Check file format (CSV/Excel only)
Ensure proper encoding (UTF-8)
Verify file permissions

Performance Tips

Use sample datasets for testing
Limit data size for better performance
Clear results between analyses for memory optimization

🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request


Streamlit for the amazing web framework
LangChain for AI agent capabilities
Google Gemini for powerful language model
Plotly & Matplotlib for visualization tools