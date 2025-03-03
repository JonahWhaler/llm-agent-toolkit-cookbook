"""
Ollama Image Interpretation Example

This module demonstrates how to use the Ollama image-to-text API for image interpretation.
It shows how to send an image file along with a prompt to the API and receive a textual description
or analysis of the image.
"""

import logging
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.local import Image_to_Text

logging.basicConfig(
    filename="./logs/example-ollama.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def interpret(llm: Image_to_Text, prompt: str, filepath: str) -> str | None:
    """
    Interpret an image using the Ollama image-to-text API.

    This function sends a prompt along with the image specified by the filepath to the API.
    It processes the API response to extract and log the textual interpretation of the image.

    Args:
        llm (Image_to_Text): An instance configured for image-to-text interactions.
        prompt (str): The prompt guiding the interpretation (e.g., "Describe this image.").
        filepath (str): Path to the image file to be interpreted.

    Returns:
        str | None: The final interpreted text prefixed with "Assistant:", or None if an error occurs.
    """
    try:
        responses, token_usage = llm.interpret(
            query=prompt, context=None, filepath=filepath
        )
        output_strings = ["\nAssistant:"]
        for response in responses:
            output_strings.append(response["content"])

        output_string = "\n".join(output_strings)
        logger.info(output_string)
        return "Assistant:" + responses[-1]["content"]
    except RuntimeError as re:
        logger.error(str(re), exc_info=True)
    except Exception as e:
        logger.error(str(e), exc_info=True)
    return None


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    CONNECTION_STRING = "http://localhost:11434"
    CONFIG = ChatCompletionConfig(
        name="llava:7b",
        return_n=1,
        max_iteration=1,
        max_tokens=2048,
        max_output_tokens=1024,
        temperature=0.7,
    )
    llm = Image_to_Text(
        connection_string=CONNECTION_STRING,
        system_prompt="You are a helpful assistant.",
        config=CONFIG,
    )

    PROMPT = "Describe this image."
    FILEPATH = "./assets/author.jpeg"

    answer = interpret(llm, PROMPT, FILEPATH)
    if answer:
        print(answer)
