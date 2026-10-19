# GenAI Synthetic Data Generator & Talk to Your Data

## Tech Stack
- **UI**: Streamlit
- **LLM**: Vertex AI (`gemini-2.5-flash` via `google-genai` SDK)
- **Database**: PostgreSQL 16 (via Docker Compose)
- **Observability**: Langfuse

## How to Run Locally
1. Start PostgreSQL:
   ```bash
   docker compose up -d
Install dependencies:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Configure environment variables (.env):
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://us.cloud.langfuse.com"
Authenticate with GCP and start Streamlit:
gcloud auth application-default login
python3 -m streamlit run app.py
