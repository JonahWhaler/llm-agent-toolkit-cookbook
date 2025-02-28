"""
OpenAI Encoding Example

This module demonstrate how to use the OpenAI embed API for transforming text to embedidings.

Attention: Add `OPENAI_API_KEY` to .env file.
"""

import logging
from dotenv import load_dotenv
from llm_agent_toolkit.encoder.remote import OpenAIEncoder


logging.basicConfig(
    filename="./logs/example-openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def transform(encoder: OpenAIEncoder, text: str) -> list[float]:
    """
    Transform input text to embeddings.

    Args:
        encoder (OpenAIEncoder): An instance for embeddings interactions.
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
    load_dotenv()
    logger = logging.getLogger(__name__)
    MODEL_NAME = "text-embedding-3-small"
    PROMPT = "Summarize this https://pypi.org/project/openai/ page for me."

    encoder = OpenAIEncoder(model_name=MODEL_NAME, dimension=512)
    embeddings = transform(encoder, PROMPT)
    if embeddings:
        print(embeddings)
