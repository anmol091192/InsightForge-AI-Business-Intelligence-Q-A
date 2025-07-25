import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationSummaryMemory
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
import matplotlib.pyplot as plt
import pandas as pd

# Custom tools for data loading and search
from tools.pdf_tool import load_pdf_data
from tools.search_tool import search_the_internet

# Configuration Constants
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    TEMPERATURE = 0.0
    MAX_RETRIES = 3
    AGE_BINS = [0,18,25,35,45,55,65,100]
    AGE_LABELS = ['<18','18-25','26-35','36-45','46-55','56-65','65+']

# Load environment variables for API keys
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

def validate_api_keys():
    """Validate that required API keys are available"""
    missing_keys = []
    if not gemini_api_key:
        missing_keys.append("GEMINI_API_KEY")
    if not openai_api_key:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Please add your API keys to the .env file")
        st.stop()

def validate_csv_data(df):
    """Validate CSV data has required columns"""
    required_columns = ['Date', 'Sales', 'Product', 'Region', 'Customer_Age', 'Customer_Gender']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"CSV is missing required columns: {missing_columns}")
        st.info(f"Required columns: {required_columns}")
        return False
    
    # Check for empty data
    if df.empty:
        st.error("CSV file is empty")
        return False
    
    return True

def process_csv_data(df):
    """Process CSV data and create document summaries"""
    summaries = []
    
    # Total sales by product
    product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
    summaries.append(Document(page_content=f"Total sales by product:\n{product_sales.to_string()}"))
    
    # Total sales by region
    region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    summaries.append(Document(page_content=f"Total sales by region:\n{region_sales.to_string()}"))
    
    # Monthly sales trend
    monthly_sales = df.resample('ME', on='Date')['Sales'].sum()
    summaries.append(Document(page_content=f"Monthly sales trend:\n{monthly_sales.to_string()}"))
    
    # Age group analysis
    df['Age_Group'] = pd.cut(df['Customer_Age'], bins=Config.AGE_BINS, labels=Config.AGE_LABELS, right=False)
    age_sales = df.groupby('Age_Group', observed=False)['Sales'].mean()
    summaries.append(Document(page_content=f"Average sales by age group:\n{age_sales.to_string()}"))
    
    # Gender analysis
    gender_sales = df.groupby('Customer_Gender')['Sales'].mean()
    summaries.append(Document(page_content=f"Average sales by gender:\n{gender_sales.to_string()}"))
    
    # Statistical measures
    sales_median = df['Sales'].median()
    sales_std = df['Sales'].std()
    summaries.append(Document(page_content=f"Sales median: {sales_median}\nSales standard deviation: {sales_std}"))
    
    return summaries, {
        'product_sales': product_sales,
        'region_sales': region_sales,
        'monthly_sales': monthly_sales,
        'age_sales': age_sales,
        'gender_sales': gender_sales,
        'sales_median': sales_median,
        'sales_std': sales_std
    }

def create_reference_answers(metrics):
    """Create reference answers for evaluation"""
    return {
        "What is the sales trend?": f"Monthly sales trend:\n{metrics['monthly_sales'].to_string()}",
        "Which product performed best?": f"Total sales by product:\n{metrics['product_sales'].to_string()}",
        "Which region had the highest sales?": f"Total sales by region:\n{metrics['region_sales'].to_string()}",
        "What is the average sales by age group?": f"Average sales by age group:\n{metrics['age_sales'].to_string()}",
        "What is the average sales by gender?": f"Average sales by gender:\n{metrics['gender_sales'].to_string()}",
        "What is the median and standard deviation of sales?": f"Sales median: {metrics['sales_median']}\nSales standard deviation: {metrics['sales_std']}",
    }

@st.cache_resource
def create_or_load_faiss_index(_all_docs, data_source):  # Add underscore prefix
    """Create or load FAISS index with caching"""
    openai_embed = OpenAIEmbeddings()
    faiss_index_path = f"faiss_index_{data_source.lower()}"
    
    try:
        if os.path.exists(faiss_index_path):
            faiss_index = FAISS.load_local(
                faiss_index_path,
                openai_embed,
                allow_dangerous_deserialization=True
            )
            st.success("✅ Loaded existing index")
        else:
            faiss_index = FAISS.from_documents(_all_docs, openai_embed)  # Use _all_docs
            faiss_index.save_local(faiss_index_path)
            st.success("✅ Created new index")
        
        return faiss_index
    except Exception as e:
        st.error(f"Error with FAISS index: {str(e)}")
        return None

def safe_agent_invoke(agent, prompt, max_retries=Config.MAX_RETRIES):
    """Safely invoke agent with retry logic"""
    for attempt in range(max_retries):
        try:
            result = agent.invoke(prompt)
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            else:
                return str(result)
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Agent failed after {max_retries} attempts: {str(e)}")
                return "I apologize, but I encountered an error processing your request."
            st.warning(f"Attempt {attempt + 1} failed, retrying...")
    
    return "Unable to process request after multiple attempts."

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None
    if "embeddings_created" not in st.session_state:
        st.session_state.embeddings_created = False

def create_visualizations(df):
    """Create and display visualizations"""
    if df is None:
        return
    
    st.markdown("---")
    st.header("📊 Sales Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly sales trend
        monthly_sales = df.resample('ME', on='Date')['Sales'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(monthly_sales['Date'], monthly_sales['Sales'], marker='o', linewidth=2, color='#1f77b4')
        ax.set_title("Sales Trends Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Product performance
        product_sales = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        product_sales.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_title("Product Performance Comparison", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Product")
        ax2.set_ylabel("Total Sales")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Regional sales
        region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        region_sales.plot(kind='bar', ax=ax3, color='orange')
        ax3.set_title("Regional Sales Analysis", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Region")
        ax3.set_ylabel("Total Sales")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    
    with col4:
        # Age group analysis
        df['Age_Group'] = pd.cut(df['Customer_Age'], bins=Config.AGE_BINS, labels=Config.AGE_LABELS, right=False)
        age_sales = df.groupby('Age_Group', observed=False)['Sales'].mean()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        age_sales.plot(kind='bar', ax=ax4, color='green')
        ax4.set_title("Average Sales by Age Group", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Age Group")
        ax4.set_ylabel("Average Sales")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    # Gender analysis (full width)
    gender_sales = df.groupby('Customer_Gender')['Sales'].mean()
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    gender_sales.plot(kind='bar', ax=ax5, color='purple')
    ax5.set_title("Average Sales by Gender", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Gender")
    ax5.set_ylabel("Average Sales")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("Median Sales", f"${df['Sales'].median():,.2f}")
    
    with col_stats2:
        st.metric("Average Sales", f"${df['Sales'].mean():,.2f}")
    
    with col_stats3:
        st.metric("Standard Deviation", f"${df['Sales'].std():,.2f}")
    
    with col_stats4:
        st.metric("Total Records", f"{len(df):,}")

def display_chat_history():
    """Display chat history in a better format"""
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Chat History")
        
        for i, (role, msg) in enumerate(st.session_state.chat_history):
            if role == "User":
                with st.container():
                    st.markdown(f"**🙋 You:** {msg}")
            elif role == "AI":
                with st.container():
                    st.markdown(f"**🤖 InsightForge:** {msg}")
                    if i < len(st.session_state.chat_history) - 1:  # Not the last message
                        st.markdown("---")

# Main Application
def main():
    # Validate API keys
    validate_api_keys()
    
    # Initialize session state
    initialize_session_state()
    
    # Define Streamlit UI title
    st.title("🔍 InsightForge: AI Business Intelligence Q&A")
    st.markdown("Upload your data and ask intelligent questions!")
    
    # User selects data source and LLM
    col1, col2 = st.columns(2)
    with col1:
        data_source = st.selectbox("📊 Choose your data source:", ["CSV", "PDF"])
    with col2:
        llm_choice = st.selectbox("🤖 Choose your LLM:", ["Gemini", "OpenAI"])
    
    uploaded_file = None
    df = None
    reference_answers = {}
    
    # CSV data upload and preprocessing
    if data_source == "CSV":
        uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['Date'])
                
                # Validate CSV data
                if not validate_csv_data(df):
                    st.stop()
                
                st.success(f"✅ Successfully loaded CSV with {len(df)} records")
                
                # Process CSV data
                with st.spinner("Processing CSV data..."):
                    summaries, metrics = process_csv_data(df)
                    all_docs = summaries
                    reference_answers = create_reference_answers(metrics)
                
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                st.stop()
    
    # PDF data upload and preprocessing
    elif data_source == "PDF":
        uploaded_file = st.file_uploader("📁 Upload your PDF file", type=["pdf"])
        if uploaded_file is not None:
            try:
                temp_pdf_path = f"temp_{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing PDF..."):
                    pdf_text = load_pdf_data(temp_pdf_path)
                    doc_for_embedding = Document(page_content=pdf_text)
                    
                    # Split PDF into chunks for embedding
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=Config.CHUNK_SIZE, 
                        chunk_overlap=Config.CHUNK_OVERLAP
                    )
                    chunked_docs = splitter.split_documents([doc_for_embedding])
                    all_docs = chunked_docs
                
                # Clean up temp file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                
                st.success(f"✅ Successfully processed PDF with {len(all_docs)} chunks")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.stop()
    
    # Main logic runs after file upload
    if uploaded_file is not None:
        # Choose LLM based on user selection
        try:
            if llm_choice == "Gemini":
                chat = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=Config.TEMPERATURE,
                    google_api_key=gemini_api_key,
                    convert_system_message_to_human=True
                )
            else:  # OpenAI
                chat = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=Config.TEMPERATURE,
                    openai_api_key=openai_api_key
                )
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.stop()
        
        # Create embeddings and FAISS index for retrieval
        with st.spinner("Creating embeddings..."):
            faiss_index = create_or_load_faiss_index(all_docs, data_source)
            if faiss_index is None:
                st.stop()
            retriever = faiss_index.as_retriever()
        
        # Initialize chains and agent
        try:
            # Conversation memory for context
            memory = ConversationSummaryMemory(
                llm=chat,
                memory_key="chat_history",
                input_key="query"
            )
            
            # RetrievalQA chain for answering questions
            qa_chain = RetrievalQA.from_chain_type(
                llm=chat,
                chain_type="stuff",
                retriever=retriever,
                memory=memory,
                return_source_documents=False,
                output_key="result"
            )
            
            # QA evaluation chain for automatic grading (only for CSV)
            if data_source == "CSV":
                qa_eval_chain = QAEvalChain.from_llm(chat)
            
            # Initialize agent with tools
            tools = [search_the_internet]
            agent = initialize_agent(
                tools=tools,
                llm=chat,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True 
            )
            
        except Exception as e:
            st.error(f"Error initializing chains/agent: {str(e)}")
            st.stop()
        
        # User input for query
        st.markdown("---")
        st.subheader("❓ Ask a Question")
        user_query = st.text_input("What would you like to know about your data?", 
                                  placeholder="e.g., What is the sales trend?")
        
        # Handle query submission
        if st.button("🚀 Submit", type="primary") and user_query.strip():
            try:
                with st.spinner("🤔 Processing your question..."):
                    # Step 1: Retrieve relevant documents
                    retrieved_docs = retriever.get_relevant_documents(user_query)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # Step 2: Build agent prompt with context
                    agent_prompt = (
                        f"You are a business intelligence analyst. "
                        f"Here is relevant context from the data:\n{context}\n\n"
                        f"User question: {user_query}\n"
                        "If you need external information, use the available tools. "
                        "Otherwise, answer using the provided context."
                    )
                    
                    # Step 3: Get answer from agent
                    answer = safe_agent_invoke(agent, agent_prompt)
                    
                    # Store in chat history
                    st.session_state.chat_history.append(("User", user_query))
                    st.session_state.chat_history.append(("AI", answer))
                    
                    # Display answer
                    st.success("✅ Answer generated!")
                    st.markdown("### 🤖 Agent Answer:")
                    st.markdown(answer)
                    
                    # Only evaluate if reference answers exist (i.e., CSV mode)
                    if data_source == "CSV" and reference_answers and user_query in reference_answers:
                        try:
                            st.markdown("### 📊 Model Evaluation:")
                            examples = [{
                                "query": user_query,
                                "answer": reference_answers[user_query]
                            }]
                            predictions = [{
                                "result": answer
                            }]
                            eval_result = qa_eval_chain.evaluate(examples, predictions)
                            if isinstance(eval_result, list) and len(eval_result) > 0:
                                eval_result = eval_result[0]
                            
                            result_text = eval_result.get('results', 'N/A')
                            if 'CORRECT' in result_text:
                                st.success(f"✅ **Evaluation Result:** {result_text}")
                            else:
                                st.warning(f"⚠️ **Evaluation Result:** {result_text}")
                                
                        except Exception as e:
                            st.error(f"Evaluation error: {e}")
                    elif data_source == "PDF":
                        st.info("ℹ️ Reference-based evaluation is only available for CSV data.")
                    elif data_source == "CSV":
                        st.info("ℹ️ No reference answer available for this question.")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        # Display chat history
        display_chat_history()
        
        # Create visualizations (only for CSV)
        if data_source == "CSV" and df is not None:
            create_visualizations(df)

if __name__ == "__main__":
    main()