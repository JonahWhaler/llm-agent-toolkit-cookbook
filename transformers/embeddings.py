"""
Ollama Encoding Example

This module demonstrate how to use the Ollama embed API for transforming text to embedidings.
"""

import logging
from llm_agent_toolkit.encoder import TransformerEncoder


logging.basicConfig(
    filename="./logs/example-transformers.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def transform(encoder: TransformerEncoder, text: str) -> list[float]:
    """
    Transform input text to embeddings.

    Args:
        encoder (TransformerEncoder): An instance for embeddings interactions.
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
    MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
    PROMPT = "Summarize this https://pypi.org/project/transformers/ page for me."

    encoder = TransformerEncoder(model_name=MODEL_NAME)
    embeddings = transform(encoder, PROMPT)

    if embeddings is not None:
        print(embeddings)
