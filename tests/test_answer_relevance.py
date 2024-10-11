import os

from dotenv import load_dotenv

from redeval.evaluators.answer_relevance import AnswerRelevance

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def test_context_relevance_bad():
    # Create an instance of AnswerRelevance
    answer_relevance = AnswerRelevance("gpt-3.5-turbo", openai_api_key)

    # Running the evaluate method
    query = "Who invented the linux os?"
    response = "bjarne stroustrup invented C++"
    metric_result, explanation = answer_relevance.evaluate(query, response)

    # Assertions
    assert metric_result == 1


def test_context_relevance_good():

    # Create an instance of AnswerRelevance class
    context_relevance = AnswerRelevance("gpt-3.5-turbo", openai_api_key)

    # Running the evaluate method
    query = "Who invented C++?"
    response = "bjarne stroustrup invented C++"
    metric_result, explanation = context_relevance.evaluate(query, response)

    # Assertions
    assert metric_result == 0
