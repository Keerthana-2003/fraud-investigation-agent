FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY src ./src
COPY data/policies ./data/policies
COPY app.py ./
RUN pip install -e .

# Generate synthetic data and train the risk model at build time so the container starts fast.
RUN python -m fraud_agent.bootstrap

# Hugging Face Spaces (Docker SDK) expects port 7860; locally docker compose maps 8501.
ENV PORT=8501
EXPOSE 8501 7860
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
