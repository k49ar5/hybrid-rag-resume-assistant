# Hybrid RAG Resume Assistant

This project implements a Retrieval-Augmented Generation (RAG) pipeline that enables semantic search and conversation with a resume (CV).

The goal was to create a cost-effective architecture that runs the application logic locally while offloading heavy LLM inference to the cloud.

## Architecture Overview

The system uses a **Hybrid Cloud Architecture**:

1. **Local Environment:** - **FastAPI:** Handles HTTP requests and application logic.
   - **Qdrant (Docker):** Stores vector embeddings of the PDF document locally for privacy and speed.
   - **HuggingFace:** Generates embeddings using `all-MiniLM-L6-v2`.

2. **Cloud Compute (Google Colab):**
   - **Ollama:** Serves the Llama 3 model on a free T4 GPU.
   - **Ngrok:** Provides a secure tunnel to expose the model to the local backend.

## Tech Stack

- **Language:** Python 3.10
- **Frameworks:** FastAPI, LangChain, Pydantic
- **Database:** Qdrant (Vector DB)
- **LLM:** Llama 3
- **DevOps:** Docker, Docker Compose
- **Tools:** Ngrok, Google Colab

## Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher installed.
- Docker Desktop installed and running.
- An active Google Colab instance running Ollama and Ngrok.

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd hybrid-rag-resume-assistant
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
