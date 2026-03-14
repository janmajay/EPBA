import structlog
import logging
import os
import time
from contextlib import contextmanager

def configure_logger(service_name: str, log_dir: str = "logs"):
    # Ensure log directory exists
    service_log_dir = os.path.join(log_dir, service_name)
    os.makedirs(service_log_dir, exist_ok=True)
    
    log_file = os.path.join(service_log_dir, f"{service_name}.json")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.WriteLoggerFactory(
            file=open(log_file, "a") # Simple append mode for demo
        )
    )
    
    return structlog.get_logger(service_name=service_name)

# Quick helper context manager for timing
@contextmanager
def log_execution_time(logger, operation_name: str):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info("execution_completed", operation=operation_name, duration_seconds=duration)
