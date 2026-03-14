FROM python:3.11-slim

WORKDIR /app

# Copy and install shared lib
COPY shared/setup.py /app/shared/setup.py
COPY shared/src /app/shared/src
RUN pip install -e /app/shared

# Install service deps
RUN pip install langchain langchain-chroma langchain-openai langchain-community fastapi uvicorn pypdf

# Copy service code
COPY services/vector_agent/src /app/services/vector_agent/src
COPY config /app/config

# Set python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8002

# Run app
CMD ["uvicorn", "services.vector_agent.src.app:app", "--host", "0.0.0.0", "--port", "8002"]
