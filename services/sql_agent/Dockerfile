FROM python:3.11-slim

WORKDIR /app

# Copy root requirements first (cache optimization)
# In monorepo, usually we copy specific service requirements, but we are moving to shared lib.
# Let's install the shared lib which has deps.
COPY shared/setup.py /app/shared/setup.py
COPY shared/src /app/shared/src
RUN pip install -e /app/shared

# Install service deps (LangChain etc) - ideally should be in pyproject.toml or setup.py of service
# but for now we inline or use old requirements logic if needed. 
# Since we removed requirements.txt, let's install common ML deps manually or via shared if generic.
# Actually we need langchain packages specifically for this service.
RUN pip install langchain langchain-community langchain-openai fastapi uvicorn

# Copy service code
COPY services/sql_agent/src /app/services/sql_agent/src
COPY config /app/config

# Set python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run app (module path changes due to moving to src)
CMD ["uvicorn", "services.sql_agent.src.app:app", "--host", "0.0.0.0", "--port", "8001"]
