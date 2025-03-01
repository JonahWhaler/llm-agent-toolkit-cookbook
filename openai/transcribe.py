"""
OpenAI Whisper Transcribe Example

This module demonstrates how to use `whisper-1` for transcript generation.

Attention: Add `OPENAI_API_KEY` to .env file.
"""

import os
import logging
from dotenv import load_dotenv
from llm_agent_toolkit.transcriber import AudioParameter, TranscriptionConfig
from llm_agent_toolkit.transcriber.open_ai import OpenAITranscriber


logging.basicConfig(
    filename="./logs/example-openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def transcribe(
    transcriber: OpenAITranscriber, prompt: str, filepath: str, tmp_directory: str
) -> str | None:
    """
    Generate text transcript from audio file.

    Args:
        transcriber (OpenAITranscriber): An instance configured for trascription.
        prompt (str): The prompt guiding the transcription.
        filepath (str): Path to the audio file.
        tmp_directory (str): The folder to store sliced audio chunks.

    Returns:
        str | None: The generated transcript, or None if an error occurs.
    """
    try:
        responses = transcriber.transcribe(
            prompt=prompt, filepath=filepath, tmp_directory=tmp_directory
        )
        output_strings = [response["content"] for response in responses]
        output_string = "\n".join(output_strings)
        return "Whisper: " + output_string
    except RuntimeError as re:
        logger.error(str(re), exc_info=True)
    except Exception as e:
        logger.error(str(e), exc_info=True)


def write(filepath: str, content: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    load_dotenv()
    logger = logging.getLogger(__name__)
    OUTPUT_DIRECTORY = "./output"  # Generated audio chunked will be stored here
    AUDIO_PARAMETER = AudioParameter(
        max_size_mb=10
    )  # Configure the audio chunking method
    PROMPT = "Give me a novel insight on the color of wind."
    FILEPATH = "./assets/short.mp3"
    MODEL_NAME = "whisper-1"

    logger.info("Test 1: Text")
    RESPONSE_FORMAT_TEXT = "text"
    whisper_1 = OpenAITranscriber(
        config=TranscriptionConfig(
            name=MODEL_NAME,
            return_n=1,
            max_iteration=1,
            response_format=RESPONSE_FORMAT_TEXT,
            temperature=0.0,
        ),
        audio_parameter=AUDIO_PARAMETER,
    )
    transcript_1 = transcribe(
        whisper_1, prompt=PROMPT, filepath=FILEPATH, tmp_directory=OUTPUT_DIRECTORY
    )
    if transcript_1:
        _filename = f"{MODEL_NAME}-{RESPONSE_FORMAT_TEXT}.md"
        _export_path = os.path.join(OUTPUT_DIRECTORY, _filename)
        write(_export_path, transcript_1)

    logger.info("Test 2: JSON")
    RESPONSE_FORMAT_JSON = "json"
    whisper_2 = OpenAITranscriber(
        config=TranscriptionConfig(
            name=MODEL_NAME,
            return_n=1,
            max_iteration=1,
            response_format=RESPONSE_FORMAT_JSON,
            temperature=0.0,
        ),
        audio_parameter=AUDIO_PARAMETER,
    )
    transcript_2 = transcribe(
        whisper_2, prompt=PROMPT, filepath=FILEPATH, tmp_directory=OUTPUT_DIRECTORY
    )
    if transcript_2:
        _filename = f"{MODEL_NAME}-{RESPONSE_FORMAT_JSON}.md"
        _export_path = os.path.join(OUTPUT_DIRECTORY, _filename)
        write(_export_path, transcript_2)
