import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

# LangChain Integrations
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("api")

# Environment Variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cv_test")
COLAB_URL = os.getenv("COLAB_URL")
EMBEDDING_MODEL = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")


# --- Data Models (DTOs) ---
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User query for the RAG system")


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    processing_time: float


# --- Application State ---
# Holds the initialized instances of our AI models
app_state: Dict[str, Any] = {}


# --- Lifespan Manager (Modern Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the application.
    Initializes connections to Qdrant and the remote LLM on startup.
    """
    logger.info("Initializing Hybrid RAG services...")

    # 1. Initialize Vector Database Connection
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        app_state["vector_store"] = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL
        )
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        logger.critical(f"Failed to connect to Vector Store: {e}")
        # We don't exit here to allow health checks to pass, but logic will fail gracefully

    # 2. Initialize LLM Client
    if not COLAB_URL:
        logger.warning("COLAB_URL is not set in environment variables. LLM features will be disabled.")
    else:
        try:
            app_state["llm"] = ChatOllama(
                base_url=COLAB_URL,
                model=LLM_MODEL,
                temperature=0.1
            )
            logger.info(f"LLM Client initialized pointing to: {COLAB_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")

    yield  # Application runs here

    # Cleanup
    logger.info("Shutting down services.")
    app_state.clear()


# --- API Definition ---
app = FastAPI(
    title="Hybrid RAG Assistant API",
    description="Backend service for semantic resume analysis using Qdrant and Llama 3.",
    version="1.0.0",
    lifespan=lifespan
)


# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    status = {
        "status": "healthy",
        "qdrant": "connected" if "vector_store" in app_state else "disconnected",
        "llm": "connected" if "llm" in app_state else "disconnected"
    }
    if status["qdrant"] == "disconnected":
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Processes the user query using RAG pipeline.
    """
    start_time = time.time()

    # Dependency Injection manually from state
    vector_store = app_state.get("vector_store")
    llm = app_state.get("llm")

    # Guard clauses
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Database unavailable")
    if not llm:
        raise HTTPException(status_code=503, detail="LLM Service unavailable (Check Ngrok URL)")

    query = request.question.strip()
    logger.info(f"Processing query: {query}")

    # 1. Retrieval
    try:
        docs = vector_store.similarity_search(query, k=3)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Internal Vector Search Error")

    if not docs:
        return ChatResponse(
            answer="I couldn't find any relevant information in the documents provided.",
            sources=[],
            processing_time=round(time.time() - start_time, 2)
        )

    # 2. Augmentation & Generation
    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_template("""
    You are a professional technical assistant. 
    Answer the question based strictly on the following context.

    Context:
    {context}

    Question: 
    {question}

    If the answer is not in the context, concisely state that the information is missing.
    """)

    try:
        chain = prompt_template | llm
        response = chain.invoke({"context": context_text, "question": query})

        duration = round(time.time() - start_time, 2)
        logger.info(f"Request completed in {duration}s")

        return ChatResponse(
            answer=response.content,
            sources=[doc.page_content[:100] + "..." for doc in docs],
            processing_time=duration
        )

    except Exception as e:
        logger.error(f"LLM Inference error: {e}")
        raise HTTPException(status_code=502, detail="Upstream LLM Error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)