"""
ChromaMemory Example

This module demonstrate how to use the ChromaMemory which is a wrapper class on top of ChromaDB. It has encoding and chunking capabilities.

For the sake of simplicity:
* `OllamaEncoder` is used as the encoding strategy.
* `FixedCharacterChunker` is used as the chunking strategy.
"""

import logging
import chromadb
from llm_agent_toolkit import VectorMemory
from llm_agent_toolkit.memory import ChromaMemory
from llm_agent_toolkit.encoder import OllamaEncoder
from llm_agent_toolkit.chunkers import FixedCharacterChunker


logging.basicConfig(
    filename="./logs/example-memory.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    CONNECTION_STRING = "http://localhost:11434"
    ENCODER_NAME = "bge-m3:latest"

    # Initialization
    encoder = OllamaEncoder(
        connection_string=CONNECTION_STRING, model_name=ENCODER_NAME
    )

    vdb = chromadb.Client(
        settings=chromadb.Settings(  # type: ignore
            is_persistent=True, persist_directory="./output/vect"
        )
    )

    chunker = FixedCharacterChunker(config={"chunk_size": 200, "stride_rate": 1.0})
    vector_memory = ChromaMemory(
        vdb=vdb, encoder=encoder, chunker=chunker, namespace="test", overwrite=True
    )

    FILEPATH = "./assets/sample.md"

    # Add document to the store
    content = load(FILEPATH)
    vector_memory.add(content)

    # Query from the store
    Q = "How to install?"
    query_response = vector_memory.query(query_string=Q)

    # Output resutls
    result = query_response["result"]
    documents = result["documents"]
    for document_chunk in documents:
        logger.info(">> %s", document_chunk)
