import os

from dotenv import load_dotenv


def test_load_openai_key():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key is not None and openai_api_key != ""
