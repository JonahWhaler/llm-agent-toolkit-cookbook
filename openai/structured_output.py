"""
OpenAI Chat Completion Example with Structured Output

This module demonstrates how to use the OpenAI text-to-text API for generating chat completions that output results which is compliance with the provided pydantic model.

Attention: Add `OPENAI_API_KEY` to .env file.
"""

import json
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from llm_agent_toolkit.core.open_ai import OpenAICore, OAI_StructuredOutput_Core

logging.basicConfig(
    filename="./logs/example-openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ResponseModel(BaseModel):
    interpretation: str
    thought_process: list[str]
    response: str


def generate(llm: OAI_StructuredOutput_Core, prompt: str) -> str | None:
    """
    Generate a single chat completion structured response.

    This function sends a prompt to a text-to-text model and expects the LLM to return a JSON-formatted
    response (compliance with the provided pydantic model).
    In case of errors, the error is logged and the function returns None.

    Args:
        llm (OAI_StructuredOutput_Core): An instance configured for text-to-text interactions with structured output.
        prompt (str): The user prompt or query sent to the language model.

    Returns:
        str | None: The `json.loads` ready model's output, or None if an error occurs.
    """
    try:
        responses, usage = llm.run(
            query=prompt, context=None, mode=ResponseMode.SO, format=ResponseModel
        )

        output_strings = ["\nAssistant:"]
        for response in responses:
            output_strings.append(response["content"])

        output_string = "\n".join(output_strings)
        logger.info(output_string)

        final_step_content = responses[-1]["content"]
        # Check whether `final_step_content` is a valid JSON string
        try:
            content_dict = json.loads(final_step_content)

            for k, v in content_dict.items():
                logger.info("%s: %s", k, v)

            return final_step_content
        except json.JSONDecodeError as jde:
            logger.error(str(jde), exc_info=True)
    except RuntimeError as re:
        logger.error(str(re), exc_info=True)
    except Exception as e:
        logger.error(str(e), exc_info=True)
    return None


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
    # Load configuration for the OpenAI API from CSV
    OpenAICore.load_csv("./config/openai.csv")
    llm = OAI_StructuredOutput_Core(
        system_prompt="You are a helpful assistant.", config=CONFIG
    )
    # Single-round conversation example
    PROMPT = "Give me a novel insight on the color of wind."
    answer = generate(llm, PROMPT)
    if answer:
        print(answer)
