FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (keeps image smaller than the default CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py ./
COPY data/ ./data/
COPY mint_repo/ ./mint_repo/

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
