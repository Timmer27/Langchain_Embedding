from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def load_persisted_chroma_db() -> Chroma:
    # Load the Chroma vector store from the persisted directory
    return Chroma(
        persist_directory="vector_store",
        embedding_function=OpenAIEmbeddings(),
    )