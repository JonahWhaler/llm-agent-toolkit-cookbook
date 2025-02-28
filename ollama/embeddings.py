"""
Ollama Encoding Example

This module demonstrate how to use the Ollama embed API for transforming text to embedidings.
"""

import logging
from llm_agent_toolkit.encoder.local import OllamaEncoder


logging.basicConfig(
    filename="./logs/example-ollama.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def transform(encoder: OllamaEncoder, text: str) -> list[float]:
    """
    Transform input text to embeddings.

    Args:
        encoder (OllamaEncoder): An instance for embeddings interactions.
        text (str): Text chunk to be transformed.

    Returns:
        list[float]: Embeddings.
    """
    try:
        embeddings = encoder.encode(text=text)
        return embeddings
    except RuntimeError as re:
        logger.error(str(re), exc_info=True)
    except Exception as e:
        logger.error(str(e), exc_info=True)
    return []


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    CONNECTION_STRING = "http://localhost:11434"
    MODEL_NAME = "bge-m3:latest"
    PROMPT = "Summarize this https://pypi.org/project/ollama/ page for me."

    encoder = OllamaEncoder(connection_string=CONNECTION_STRING, model_name=MODEL_NAME)
    embeddings = transform(encoder, PROMPT)
    if embeddings:
        print(embeddings)
