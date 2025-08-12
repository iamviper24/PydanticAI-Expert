import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

SOURCE_DIR = "crawled_content"
CHROMA_PERSIST_DIR = "chroma_db_pydanticai"
CHROMA_COLLECTION_NAME = "pydantic_ai_rag"
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    print("\nStage 2: Embedding and Storing in ChromaDB")
    
    if not os.path.exists(SOURCE_DIR) or not os.listdir(SOURCE_DIR):
        print(f"FATAL: Source directory '{SOURCE_DIR}' is empty or does not exist.")
        print("Please run `python 1_crawl.py` first to generate the content files.")
        return

    #Check if directory already exisits, if so delete it
    if os.path.exists(CHROMA_PERSIST_DIR):
        print(f"Found existing database at '{CHROMA_PERSIST_DIR}'. Deleting it.")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    print(f"Loading documents from '{SOURCE_DIR}'...")
    
    #Document Loading
    loader = DirectoryLoader(SOURCE_DIR, glob="**/*.txt", show_progress=True) #Load documents, with .txt as a suffix
    documents = loader.load()
    
    if not documents:
        print("FATAL: No documents were loaded. Check the glob pattern and the content of the 'crawled_content' directory.")
        return
        
    print(f"Loaded {len(documents)} documents.")

    #Chunking
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(splits)} chunks.")

    #Creating Emeddings
    print(f"Creating embeddings using '{HF_EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    
    print(f"Persisting {len(splits)} chunks to ChromaDB at '{CHROMA_PERSIST_DIR}'...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )
    
    print("\n--- Embedding Complete! ---")
    print("You can now run the Streamlit app: streamlit run app.py")


if __name__ == "__main__":
    main()