"""
DeepSeek Chat Completion Example with JSON Output

This module demonstrates how to use the DeepSeek text-to-text API for generating chat completions that output results in JSON format.

Attention: Add `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` to .env file.
"""

import json
import logging
from dotenv import load_dotenv
from llm_agent_toolkit import ChatCompletionConfig, ResponseMode
from llm_agent_toolkit.core.deep_seek import Text_to_Text_SO

logging.basicConfig(
    filename="./logs/example-deepseek.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate(llm: Text_to_Text_SO, prompt: str) -> str | None:
    """
    Generate a single chat completion response in JSON format.

    This function sends a prompt to a text-to-text model and expects the LLM to return a JSON-formatted
    response. In case of errors, the error is logged and the function returns None.

    Args:
        llm (Text_to_Text_SO): An instance configured for text-to-text interactions with JSON output.
        prompt (str): The user prompt or query sent to the language model.

    Returns:
        str | None: The `json.loads` ready model's output, or None if an error occurs.
    """
    try:
        responses = llm.run(query=prompt, context=None, mode=ResponseMode.JSON)

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


SYSTEM_PROMPT = """You are a helpful assistant.

Output Format (JSON):
---
{
    \"interpretation\": {{LLM's interpretation of the input}},
    \"thought_process\": [
        {{Idea|Monologue}}
    ],
    \"response\": {{LLM's response}}
}
---
"""


if __name__ == "__main__":
    load_dotenv()
    logger = logging.getLogger(__name__)
    CONFIG = ChatCompletionConfig(
        name="deepseek-chat",
        return_n=1,
        max_iteration=1,
        max_tokens=2048,
        max_output_tokens=1024,
        temperature=0.7,
    )
    llm = Text_to_Text_SO(system_prompt=SYSTEM_PROMPT, config=CONFIG)
    # Single-round conversation example
    PROMPT = "Give me a novel insight on the color of wind."
    answer = generate(llm, PROMPT)
    if answer:
        print(answer)
