"""
ShortTermMemory Example

This module demonstrate how to use the ShortTermMemory to maintain a fixed size container
which also remove oldest element when the container is full.

This is very useful when we want to ensure not more than `N` conversations are passed as the context to the LLM.
"""

import logging
from llm_agent_toolkit import ShortTermMemory


logging.basicConfig(
    filename="./logs/example-memory.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    MAX_ENTRY = 3
    N_INTERACTIONS = 10
    stm = ShortTermMemory(max_entry=MAX_ENTRY)

    for i in range(N_INTERACTIONS):
        stm.push({"role": "user", "content": str(i)})
        data_points = stm.to_list()
        logger.info("\n[%d] has %d data points", i, len(data_points))
        for dp in data_points:
            logger.info(dp)
