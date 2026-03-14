FROM python:3.11-slim

WORKDIR /app

# Install shared library
COPY shared/ /app/shared/
RUN pip install -e /app/shared/

# Install service dependencies
RUN pip install fastapi uvicorn httpx pyyaml python-dotenv structlog

# Copy service code
COPY services/agent_registry/ /app/services/agent_registry/

# Copy config
COPY config/ /app/config/

EXPOSE 8004

CMD ["uvicorn", "services.agent_registry.src.app:app", "--host", "0.0.0.0", "--port", "8004"]
