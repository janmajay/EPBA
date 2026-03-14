FROM python:3.11-slim

WORKDIR /app

# Copy and install shared lib
COPY shared/setup.py /app/shared/setup.py
COPY shared/src /app/shared/src
RUN pip install -e /app/shared

RUN pip install streamlit requests

COPY services/frontend/src /app/services/frontend/src
COPY config /app/config

# Set python path
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "services/frontend/src/app.py", "--server.address=0.0.0.0"]
