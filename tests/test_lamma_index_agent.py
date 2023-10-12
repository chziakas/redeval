import os

from dotenv import load_dotenv

from epiphany.agents.rag.lamma_index_agent import RagAgent


def test_get_responses():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Initialization
    agent = RagAgent(
        "data/examples/rag",
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo-16k",
    )
    questions_list = ["What is the weather in SF?"]

    # Method under test
    responses = agent.get_responses(questions_list)

    # Assertions
    assert isinstance(responses, list), "Expected a list of responses"
    assert len(responses) == len(questions_list), "Expected one response for each question"
