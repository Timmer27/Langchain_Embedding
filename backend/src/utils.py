from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_persisted_chroma_db() -> Chroma:
    # Load the Chroma vector store from the persisted directory
    return Chroma(
        persist_directory="vector_store",
        embedding_function=OpenAIEmbeddings(),
    )

def train_pdf_chroma_db(dataPath) -> Chroma:
    data = []
    for file in os.listdir(dataPath):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(dataPath, file)
            loader = PyPDFLoader(pdf_path)
            data.extend(loader.load())
    # 데이터 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 0
    )
    documents = text_splitter.split_documents(data)

    db = Chroma.from_documents(persist_directory="vector_store", documents=documents, embedding=OpenAIEmbeddings())
    return db