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

def train_pdf_chroma_db(files_with_path: list) -> Chroma:
    data = []
    for file in files_with_path:
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file)
            data.extend(loader.load())
    # 데이터 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 0
    )
    documents = text_splitter.split_documents(data)

    db = Chroma.from_documents(persist_directory="vector_store", documents=documents, embedding=OpenAIEmbeddings())
    return db

def check_documents_in_chroma_db(db: Chroma, dataPath: str) -> bool:
    # Check if all PDF files in the directory are already processed
    existing_filenames = set([doc['source'] for doc in db.get()['metadatas']])
    file_lists = []

    for file in os.listdir(dataPath):
        pdf_path = os.path.join(dataPath, file)
        if file.endswith('.pdf') and pdf_path not in existing_filenames:
            file_lists.append(pdf_path)
        else:
            continue
    # print('file_lists', file_lists)
    return file_lists

def initialize_chroma_db(dataPath: str):
    db = load_persisted_chroma_db()
    files_for_train = check_documents_in_chroma_db(db, dataPath)
    # print('files_for_trainfiles_for_train', files_for_train)
    if os.path.exists("vector_store") and os.path.isdir("vector_store"):
        print("Loading existing Chroma DB...")
        if len(files_for_train) == 0:
            print("All documents are already processed.")
            return db
        else:
            print("Some documents are not processed yet. Re-training Chroma DB...")
            return train_pdf_chroma_db(files_for_train)
    else:
        print("Training new Chroma DB from documents...")
        return train_pdf_chroma_db(files_for_train)