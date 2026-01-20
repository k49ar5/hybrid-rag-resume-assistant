import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


PDF_FILE= "test_cv.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "cv_test"

logging.basicConfig(
    level= logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
    )
logger = logging.getLogger(__name__)
try:
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
except Exception as e:
    logger.error(f"Filed to open a file: {e}")

logger.info(f"File has {len(docs)} sites")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap= 50
)

chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
try:

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    logger.info("Qdrant has been created")
except Exception as e:
        logger.error(f"Qdrant hasn't been created: {e}")