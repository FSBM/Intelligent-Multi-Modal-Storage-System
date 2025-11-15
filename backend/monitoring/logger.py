"""
Monitoring & Logging
Tracks ingestion, schema decisions, errors, and storage ops
"""
import logging
import sys
from pythonjsonlogger import jsonlogger
from datetime import datetime

<<<<<<< HEAD
# creates logger
=======

>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
def setup_logger(name: str = "smart_storage") -> logging.Logger:
    """Setup structured JSON logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Also add simple formatter for development
    simple_handler = logging.StreamHandler(sys.stderr)
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_handler.setFormatter(simple_formatter)
    logger.addHandler(simple_handler)
    
    return logger

