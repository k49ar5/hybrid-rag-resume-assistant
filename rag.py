import sys
import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "cv_test"
COLAB_URL = "https://unslumbering-roseanne-interchondral.ngrok-free.dev"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3"

# --- LOGGING SETUP ---
# Configuring standard logging format used in production apps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_vector_store() -> Optional[QdrantVectorStore]:
    """
    Initializes and returns the Qdrant vector store connection.
    Returns None if connection fails.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL
        )
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return None


def main():
    logger.info("Starting RAG Application...")

    # 1. Initialize Database Connection
    vector_store = get_vector_store()
    if not vector_store:
        logger.critical("Database connection failed. Exiting application.")
        sys.exit(1)

    # 2. Initialize LLM Client
    # We initialize the client once to avoid overhead in the loop
    try:
        llm = ChatOllama(
            base_url=COLAB_URL,
            model=LLM_MODEL,
            temperature=0.1
        )
    except Exception as e:
        logger.critical(f"Failed to initialize LLM client: {e}")
        sys.exit(1)

    logger.info("System ready. Waiting for user input.")
    print("\n--- RAG CLI Interface (Type 'exit' to quit) ---\n")

    # 3. Main Event Loop
    while True:
        try:
            # Get user input
            query = input("User: ").strip()

            # Handle exit commands
            if query.lower() in ["exit", "quit", "q"]:
                logger.info("Shutdown requested by user.")
                break

            if not query:
                continue

            # A. Retrieval
            docs = vector_store.similarity_search(query, k=3)

            if not docs:
                print("System: No relevant information found in the knowledge base.")
                continue

            # B. Context Assembly
            context_text = "\n\n".join([doc.page_content for doc in docs])

            # C. Prompt Construction
            template = ChatPromptTemplate.from_template("""
            You are a technical assistant. Answer the question based strictly on the context provided.

            Context:
            {context}

            Question: 
            {question}

            If the answer is not in the context, reply with: "I don't have enough information."
            """)

            final_prompt = template.invoke({
                "context": context_text,
                "question": query
            })

            # D. Generation
            # Using print for the answer to separate it from system logs
            print("System: Processing...")
            response = llm.invoke(final_prompt)
            print(f"Response: {response.content}\n")

        except KeyboardInterrupt:
            print("\n")
            logger.info("Force shutdown detected.")
            break
        except Exception as e:
            logger.error(f"Runtime error: {e}")


if __name__ == "__main__":
    main()