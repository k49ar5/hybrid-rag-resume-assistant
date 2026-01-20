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
- An Ngrok account (for the authentication token).

### 2. Cloud Setup (The "Brain")
Since this is a hybrid architecture, the LLM runs in the cloud.

1.  Upload the file `llm_server_colab.ipynb` from this repository to **Google Colab**.
2.  In the notebook, find the cell containing:
    ```python
    !ngrok config add-authtoken <YOUR_TOKEN>
    ```
    and replace `<YOUR_TOKEN>` with your personal Ngrok Authtoken.
3.  **Run all cells** (Runtime -> Run all).
4.  Copy the public Ngrok URL (e.g., `https://xxxx-xx-xx.ngrok-free.dev`) generated in the last step.

### 3. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/k49ar5/hybrid-rag-resume-assistant
cd hybrid-rag-resume-assistant
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt

```
### 4. Configuration

Create a .env file in the root directory with the following variables:

QDRANT_URL=http://localhost:6333
COLLECTION_NAME=cv_test
# Update this URL based on your current Ngrok session
COLAB_URL=[https://your-ngrok-url.ngrok-free.dev](https://your-ngrok-url.ngrok-free.dev)

### 5. Running the Application

run.bat

Manual Start (Linux/Mac):
docker-compose up -d
python api.py

Usage
Once the application is running, access the interactive API documentation (Swagger UI) at: http://localhost:8000/docs

Example Request (POST /chat)
JSON
{
  "question": "Does the candidate have experience with Python and Docker?"
}
Project Structure
api.py: Main entry point for the FastAPI application.

ingest.py: Script to process PDF files and load embeddings into Qdrant.

docker-compose.yml: Configuration for the Qdrant database container.

run.bat: Helper script for Windows deployment.

requirements.txt: List of project dependencies.
