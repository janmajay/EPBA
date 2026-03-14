FROM python:3.11-slim

WORKDIR /app

# Copy and install shared lib
COPY shared/setup.py /app/shared/setup.py
COPY shared/src /app/shared/src
RUN pip install -e /app/shared

# Install service deps
RUN pip install langgraph langchain-openai fastapi uvicorn httpx

# Copy service code
COPY services/orchestrator/src /app/services/orchestrator/src
COPY config /app/config

# Set python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "services.orchestrator.src.app:app", "--host", "0.0.0.0", "--port", "8000"]
