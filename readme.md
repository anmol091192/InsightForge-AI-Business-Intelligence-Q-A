# 🔍 InsightForge: AI Business Intelligence Q&A

InsightForge is an intelligent business intelligence application that allows users to upload their data (CSV or PDF) and ask natural language questions to get insights powered by AI. The application combines retrieval-augmented generation (RAG) with AI agents to provide accurate, context-aware answers about your business data.

## ✨ Features

- **Multi-format Data Support**: Upload CSV files for business analytics or PDF documents for document Q&A
- **AI-Powered Analysis**: Choose between Google Gemini and OpenAI GPT models
- **Intelligent Agent System**: Combines document retrieval with external tools (web search) for comprehensive answers
- **Automatic Evaluation**: Built-in evaluation system for CSV-based queries using reference answers
- **Interactive Visualizations**: Automatic generation of charts and graphs for sales data analysis
- **Chat History**: Persistent conversation history for better context
- **Real-time Processing**: Fast document processing with FAISS vector indexing

## 🏗️ Architecture

```
User Query → Document Retrieval → AI Agent → Response
                ↓                    ↓
           FAISS Index        External Tools
                              (Web Search)
```

The application uses:
- **LangChain** for AI orchestration and document processing
- **FAISS** for vector similarity search
- **Streamlit** for the web interface
- **OpenAI/Google Gemini** for language models
- **Matplotlib** for data visualizations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Google Gemini API key
- Serper API key (for web search functionality)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/InsightForge.git
   cd InsightForge
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

### API Keys Setup

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Google Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Serper API Key**: Get from [Serper.dev](https://serper.dev/) (for web search functionality)

### Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Usage

### For CSV Data Analysis

1. **Upload your CSV file** with the following required columns:
   - `Date`: Transaction date
   - `Sales`: Sales amount
   - `Product`: Product name
   - `Region`: Sales region
   - `Customer_Age`: Customer age
   - `Customer_Gender`: Customer gender

2. **Choose your preferred LLM** (Gemini or OpenAI)

3. **Ask questions** such as:
   - "What is the sales trend?"
   - "Which product performed best?"
   - "Which region had the highest sales?"
   - "What is the average sales by age group?"

### For PDF Document Q&A

1. **Upload your PDF document**
2. **Choose your preferred LLM**
3. **Ask questions** about the document content

### Example Questions for CSV Data

- **Sales Analysis**: "What are the monthly sales trends?"
- **Product Performance**: "Which products are performing best this quarter?"
- **Regional Analysis**: "Compare sales performance across different regions"
- **Customer Demographics**: "What's the sales breakdown by customer age groups?"
- **Statistical Insights**: "What are the median and standard deviation of sales?"

## 🔧 Configuration

The application uses a `Config` class for easy customization:

```python
class Config:
    CHUNK_SIZE = 1000          # Document chunk size for processing
    CHUNK_OVERLAP = 100        # Overlap between chunks
    TEMPERATURE = 0.0          # LLM temperature (0 = deterministic)
    MAX_RETRIES = 3           # Maximum retry attempts for agent
    AGE_BINS = [0,18,25,35,45,55,65,100]  # Age group boundaries
    AGE_LABELS = ['<18','18-25','26-35','36-45','46-55','56-65','65+']
```

## 📁 Project Structure

```
InsightForge/
├── main.py                 # Main Streamlit application
├── tools/
│   ├── pdf_tool.py        # PDF processing tool
│   └── search_tool.py     # Web search tool
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔍 Features in Detail

### Intelligent Agent System
- Retrieves relevant documents using FAISS vector search
- Decides whether to use document context or external tools
- Handles parsing errors gracefully with retry logic

### Automatic Evaluation
- For CSV data, compares AI responses with reference answers
- Provides evaluation scores (CORRECT/INCORRECT)
- Helps validate response accuracy

### Data Visualizations
- Automatic generation of sales trend charts
- Product and regional performance comparisons
- Customer demographic analysis
- Statistical summary with key metrics

### Chat History
- Persistent conversation history
- Easy reference to previous questions and answers
- Better context for follow-up questions

## 🛠️ Dependencies

Key dependencies include:
- `streamlit` - Web application framework
- `langchain` - AI orchestration and document processing
- `openai` - OpenAI API integration
- `google-generativeai` - Google Gemini integration
- `faiss-cpu` - Vector similarity search
- `pandas` - Data manipulation
- `matplotlib` - Data visualization
- `python-dotenv` - Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all API keys are correctly set in the `.env` file
2. **Memory Issues**: For large files, consider reducing `CHUNK_SIZE` in the config
3. **FAISS Index Errors**: Delete existing `faiss_index_*` folders to rebuild indices
4. **CSV Format Issues**: Ensure your CSV has all required columns with correct names

### Error Messages

- **"Missing API keys"**: Add required API keys to `.env` file
- **"CSV is missing required columns"**: Check CSV format and column names
- **"Cannot hash argument"**: This is handled automatically with underscore prefixes

## 📧 Support

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Made with ❤️ using Streamlit, LangChain, and AI**