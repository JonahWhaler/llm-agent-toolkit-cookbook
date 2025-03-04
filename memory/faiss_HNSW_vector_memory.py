import logging
from llm_agent_toolkit import VectorMemory
from llm_agent_toolkit.memory import FaissHNSWDB, FaissMemory
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
    encoder = OllamaEncoder(
        connection_string=CONNECTION_STRING, model_name=ENCODER_NAME
    )

    vdb = FaissHNSWDB(
        namespace="test", dimension=encoder.dimension, db_folder="./output/vect"
    )

    chunker = FixedCharacterChunker(config={"chunk_size": 200, "stride_rate": 1.0})
    vector_memory = FaissMemory(
        vdb=vdb, encoder=encoder, chunker=chunker, overwrite=True
    )

    FILEPATH = "./assets/sample.md"
    content = load(FILEPATH)
    # @llm-agent-toolkit[all]==0.0.30.2, metadata is required!!!
    # @llm-agent-toolkit[all]>=0.0.30.4, metadata is optional!!!
    vector_memory.add(document_string=content)  # , metadata={"filename": "sample.md"}

    Q = "How to install?"

    query_response = vector_memory.query(query_string=Q)

    result = query_response["result"]
    documents = result["documents"]

    for document_chunk in documents:
        logger.info(">> %s", document_chunk)
