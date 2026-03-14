FROM python:3.11-slim

WORKDIR /app

# Copy and install shared lib
COPY shared/setup.py /app/shared/setup.py
COPY shared/src /app/shared/src
RUN pip install -e /app/shared

# Install service deps
RUN pip install langchain langchain-openai langchain-community fastapi uvicorn

# Copy service code
COPY services/summarization_agent/src /app/services/summarization_agent/src
COPY config /app/config

# Set python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8003

# Run app
CMD ["uvicorn", "services.summarization_agent.src.app:app", "--host", "0.0.0.0", "--port", "8003"]
