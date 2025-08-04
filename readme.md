# 🔍 InsightForge: AI Business Intelligence Q&A

InsightForge is an intelligent business intelligence application that allows users to upload their data (CSV or PDF) and ask natural language questions to get insights powered by AI. The application combines retrieval-augmented generation (RAG) with AI agents to provide accurate, context-aware answers about your business data.

## ✨ Features

- **Multi-format Data Support**: Upload CSV files for business analytics or PDF documents for document Q&A
- **Flexible AI Provider Selection**: Choose between Google Gemini and OpenAI GPT models - only one API key required
- **User-Provided API Keys**: Secure approach where users provide their own API keys (stored only in session memory)
- **Provider-Specific Embeddings**: Automatic selection of appropriate embeddings (OpenAI or Google) based on your chosen provider
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
- **FAISS** for vector similarity search with provider-specific embeddings
- **Streamlit** for the web interface
- **OpenAI/Google Gemini** for language models and embeddings
- **Matplotlib** for data visualizations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- **Either** OpenAI API key **OR** Google Gemini API key (not both required)
- Serper API key (for web search functionality - can be obtained for free)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/anmol091192/InsightForge.git
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
   
   Create a `.env` file in the root directory with only the Serper API key:
   ```env
   SERPER_API_KEY=your_serper_api_key_here
   ```
   
   **Note**: OpenAI and Gemini API keys are provided through the UI for security.

### API Keys Setup

1. **Serper API Key (Required)**: Get from [Serper.dev](https://serper.dev/) - free tier available
2. **Choose ONE of the following**:
   - **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
   - **Google Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 🔐 Security & API Key Management

### Why User-Provided API Keys?
- **Cost Control**: You only pay for your own usage
- **Security**: Your keys are never stored on our servers
- **Scalability**: No rate limits from shared usage
- **Privacy**: Your data and usage remain private

### Security Features
- API keys stored only in browser session memory
- Keys automatically deleted when you close the browser
- No server-side storage or logging of API keys
- Secure password-type input fields

## 📊 Usage

### Getting Started

1. **Open the application** in your browser
2. **Choose your AI provider** (OpenAI or Google Gemini) in the sidebar
3. **Enter your API key** securely in the sidebar
4. **Upload your data file** (CSV or PDF)
5. **Start asking questions** about your data

### For CSV Data Analysis

**Required CSV columns:**
- `Date`: Transaction date
- `Sales`: Sales amount
- `Product`: Product name
- `Region`: Sales region
- `Customer_Age`: Customer age
- `Customer_Gender`: Customer gender

**Example questions:**
- "What is the sales trend?"
- "Which product performed best?"
- "Which region had the highest sales?"
- "What is the average sales by age group?"

### For PDF Document Q&A

1. **Upload your PDF document**
2. **Ask questions** about the document content
3. **Get contextual answers** based on document content

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
├── .env                   # Environment variables (only Serper API key)
├── .gitignore            # Git ignore file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔍 Features in Detail

### User-Provided API Key System
- Choose between OpenAI or Google Gemini (only one required)
- Secure password-type input in sidebar
- Session-only storage with automatic cleanup
- Clear cost information and usage warnings

### Provider-Specific Embeddings
- **OpenAI users**: Uses OpenAI embeddings with FAISS
- **Gemini users**: Uses Google Generative AI embeddings with FAISS
- Separate FAISS indices for different embedding types
- Automatic provider detection and appropriate embedding selection

### Intelligent Agent System
- Retrieves relevant documents using FAISS vector search
- Decides whether to use document context or external tools
- Handles parsing errors gracefully with retry logic
- Uses Serper API for external web search when needed

### Automatic Evaluation
- For CSV data, compares AI responses with reference answers
- Provides evaluation scores (CORRECT/INCORRECT)
- Helps validate response accuracy
- Only available for CSV data with predefined reference questions

### Data Visualizations
- Automatic generation of sales trend charts
- Product and regional performance comparisons
- Customer demographic analysis
- Statistical summary with key metrics
- Interactive charts using Matplotlib

### Chat History
- Persistent conversation history within session
- Easy reference to previous questions and answers
- Better context for follow-up questions
- Clean, organized display format

## 🛠️ Dependencies

Key dependencies include:
- `streamlit>=1.28.0` - Web application framework
- `langchain>=0.1.0` - AI orchestration and document processing
- `openai>=1.0.0` - OpenAI API integration
- `google-generativeai>=0.3.0` - Google Gemini integration
- `langchain-google-genai>=0.0.8` - LangChain Google integration
- `faiss-cpu>=1.7.0` - Vector similarity search
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.8.0` - Data visualization
- `python-dotenv>=1.0.0` - Environment variable management

## 💰 Cost Information

### OpenAI Pricing
- **GPT-3.5 Turbo**: ~$0.002 per 1K tokens
- **Embeddings**: ~$0.0001 per 1K tokens
- **Typical query cost**: ~$0.01-0.05

### Google Gemini Pricing
- **Gemini 1.5 Flash**: Often has generous free tier
- **Embeddings**: Included with Gemini API
- **Cost-effective option** for most users

### Serper API
- **Free tier**: 2,500 searches per month
- **Additional searches**: $5 per 1,000 searches

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

1. **"SERPER_API_KEY not found"**: Add the Serper API key to your `.env` file
2. **"No valid API key found for embeddings"**: Ensure you've provided a valid API key for your chosen provider
3. **Memory Issues**: For large files, consider reducing `CHUNK_SIZE` in the config
4. **FAISS Index Errors**: Delete existing `faiss_index_*` folders to rebuild indices
5. **CSV Format Issues**: Ensure your CSV has all required columns with correct names

### Error Messages

- **"Please provide your [Provider] API key"**: Enter your API key in the sidebar
- **"CSV is missing required columns"**: Check CSV format and column names  
- **"API key validation failed"**: Check your API key format and validity
- **"Cannot hash argument"**: This is handled automatically with underscore prefixes

### Provider-Specific Issues

- **OpenAI**: Ensure your API key starts with `sk-`
- **Gemini**: Ensure your API key is from Google AI Studio
- **Mixed providers**: Each provider uses its own embedding type and FAISS index

### Screenshots

<img width="654" height="478" alt="Screenshot 2025-07-25 at 1 36 27 PM" src="https://github.com/user-attachments/assets/d80e8839-7141-4e8a-8173-5e72b6806c99" />
<img width="676" height="516" alt="Screenshot 2025-07-25 at 1 36 35 PM" src="https://github.com/user-attachments/assets/0b11f4ab-550b-4e5e-9ee8-4a22c2e7c330" />
<img width="689" height="571" alt="Screenshot 2025-07-25 at 1 36 41 PM" src="https://github.com/user-attachments/assets/9778cfd7-1c71-482d-a8a5-17b99566342a" />
<img width="671" height="410" alt="Screenshot 2025-07-25 at 1 36 46 PM" src="https://github.com/user-attachments/assets/6b6a3aa6-62f4-407a-a5cf-5c76c0c4ac08" />
<img width="667" height="856" alt="Screenshot 2025-07-25 at 1 36 52 PM" src="https://github.com/user-attachments/assets/fede1a17-b1ce-4cd1-bdf3-21651190932f" />
<img width="630" height="516" alt="Screenshot 2025-07-25 at 1 37 45 PM" src="https://github.com/user-attachments/assets/5dc6202d-d458-456a-9dc7-5c862efe5942" />
<img width="703" height="502" alt="Screenshot 2025-07-25 at 1 43 41 PM" src="https://github.com/user-attachments/assets/0871cd6a-1273-404f-93a5-e1b64db19e9d" />

---

**Made with ❤️ using Streamlit, LangChain, and AI**
