"""
OpenAI Chat Completion Example

This module demonstrates how to use the OpenAI text-to-text API for generating chat
completions in both single-round and multi-round (interactive) modes.

Attention: Add `OPENAI_API_KEY` to .env file.
"""

import logging
from dotenv import load_dotenv
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.open_ai import OpenAICore, Text_to_Text

logging.basicConfig(
    filename="./logs/example-openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate(llm: Text_to_Text, prompt: str) -> str | None:
    """
    Generate a single chat completion response.

    This function takes an instance of a text-to-text model and a user prompt,
    sends the prompt to the model, and returns the generated response as a string.
    In case of runtime errors or exceptions, the error is logged and the function
    returns None.

    Args:
        llm (Text_to_Text): An instance configured for text-to-text interactions.
        prompt (str): The input query or message from the user.

    Returns:
        str | None: The concatenated response from the model, or None if an error occurs.
    """
    try:
        responses = llm.run(query=prompt, context=None)
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


def multiround(llm: Text_to_Text, prompt: str | None = None) -> list[dict] | None:
    """
    Engage in a multi-round chat conversation with the assistant.

    This function starts an interactive session where the user can have a continuous
    conversation with the AI assistant. The conversation context is maintained across
    exchanges, allowing the assistant to reference previous messages. The session
    terminates when the user inputs "/exit".

    Args:
        llm (Text_to_Text): An instance of a text-to-text model for interactive conversation.
        prompt (str | None): Optional initial prompt; if None, the user will be prompted for input.

    Returns:
        list[dict] | None: A list of conversation turns (each represented as a dictionary)
                           capturing both user and assistant messages. Returns None if an error occurs.
    """
    conversations: list[dict[str, str | dict]] = []

    if prompt is None:
        prompt = input("User: ")

    try:
        while prompt != "/exit":
            responses = llm.run(
                query=prompt, context=conversations if len(conversations) > 0 else None  # type: ignore
            )
            output_strings = ["\nAssistant:"]
            for response in responses:
                output_strings.append(response["content"])

            output_string = "\n".join(output_strings)
            logger.info(output_string)
            conversations.append({"role": "user", "content": prompt})
            conversations.extend(responses)  # type: ignore
            print("AI:\n" + responses[-1]["content"] + "\n")
            prompt = input("User: ")

    except Exception as e:
        logger.error(str(e), exc_info=True)
    return conversations


if __name__ == "__main__":
    load_dotenv()
    logger = logging.getLogger(__name__)
    CONFIG = ChatCompletionConfig(
        name="gpt-4o-mini",
        return_n=1,
        max_iteration=1,
        max_tokens=2048,
        max_output_tokens=1024,
        temperature=0.7,
    )
    # Load configuration for the OpenAi API from CSV
    OpenAICore.load_csv("./config/openai.csv")
    llm = Text_to_Text(
        system_prompt="You are a helpful assistant.",
        config=CONFIG,
        tools=None,
    )
    # Single-round conversation example
    PROMPT = "Give me a novel insight on the color of wind."
    answer = generate(llm, PROMPT)
    if answer:
        print(answer)

    # Multi-round conversation example
    conversations = multiround(llm, None)
