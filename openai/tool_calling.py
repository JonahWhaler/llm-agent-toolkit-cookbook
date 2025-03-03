"""
OpenAI Chat Completion with Web Summarization Example

This module demonstrates how to integrate a web fetching tool with the OpenAI text-to-text API.
It includes a utility function to fetch and clean web page content and uses the language model
to generate responses (e.g., summarizations) based on the fetched content.

Additional Dependencies:
    - requests==2.32.3
    - beautifulsoup4==4.12.2

Attention: Add `OPENAI_API_KEY` to .env file.
"""

import logging
from dotenv import load_dotenv
from llm_agent_toolkit import ChatCompletionConfig
from llm_agent_toolkit.core.open_ai import OpenAICore, Text_to_Text
from llm_agent_toolkit.tool import LazyTool

logging.basicConfig(
    filename="./logs/example-openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def fetch_web_page(url: str) -> str:
    """
    Retrieve and clean the content of a web page.

    This function performs an HTTP GET request to fetch the HTML content from the specified URL.
    It then uses BeautifulSoup to parse the HTML and extracts cleaned text from the <body> tag.
    If the body is not found, it returns the first 4096 characters of the raw page text. In the event
    of an error (e.g., network issues), it returns "Not Available".

    Args:
        url (str): URL of the targeted web page.

    Returns:
        str: Cleaned text content from the page, or a fallback string if fetching fails.
    """
    import re
    import random

    import requests
    from bs4 import BeautifulSoup

    logger.info(f"Fetching {url}")

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Firefox/120.0",
    ]
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    try:
        page = requests.get(url=url, headers=headers, timeout=2, stream=False)
        page_text = page.text
        soup = BeautifulSoup(page_text, "html.parser")
        body = soup.find("body")
        if body:
            cleaned_text = re.sub(r"\s+", " ", body.text)
            cleaned_text = re.sub(r"\n{3,}", "\n", cleaned_text)
            return cleaned_text
        return page_text[:4096]
    except Exception as _:
        return "Not Available"


def generate(llm: Text_to_Text, prompt: str) -> str | None:
    """
    Generate a response using the text-to-text language model.

    This function sends a prompt to the language model and returns the model's response.
    It handles runtime errors by logging them and returns None if any exception occurs.

    Args:
        llm (Text_to_Text): An instance of the text-to-text language model.
        prompt (str): The input query or instruction for the language model.

    Returns:
        str | None: The generated response from the language model, or None if an error occurs.
    """
    try:
        responses, usage = llm.run(query=prompt, context=None)

        output_strings = ["\nAssistant:"]
        for response in responses:
            output_strings.append(response["content"])
        output_string = "\n\n".join(output_strings)
        logger.info(output_string)
        # Return the final step
        return responses[-1]["content"]
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
        max_iteration=3,
        max_tokens=2048,
        max_output_tokens=1024,
        temperature=0.7,
    )
    # Load configuration for the OpenAi API from CSV
    OpenAICore.load_csv("./config/openai.csv")

    # Create a LazyTool wrapping the web page fetching function.
    web_tool = LazyTool(fetch_web_page, is_coroutine_function=False)
    llm = Text_to_Text(
        system_prompt="You are a helpful assistant.",
        config=CONFIG,
        tools=[web_tool],
    )
    # Single-round example
    PROMPT = "Summarize this https://pypi.org/project/openai/ page for me."
    answer = generate(llm, PROMPT)
    if answer:
        print(answer)
