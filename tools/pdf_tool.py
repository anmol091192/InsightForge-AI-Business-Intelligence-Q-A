from langchain.document_loaders import PyPDFLoader
from langchain.tools import tool

@tool
def load_pdf_data(pdf_path='pdf/your_file.pdf'):
    """Loads and combines text from a PDF file given its path."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    # Combine all page contents into a single string
    pdf_text = "\n".join([page.page_content for page in pages])
    return pdf_text