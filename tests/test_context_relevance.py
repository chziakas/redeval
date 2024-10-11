import os

from dotenv import load_dotenv

from redeval.evaluators.context_relevance import ContextRelevance

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def test_context_relevance_bad():
    # Create an instance of ContextRelevance class
    context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)

    # Running the evaluate method
    query = "Who invented the linux os?"
    context = "bjarne stroustrup invented C++"
    metric_result, explanation = context_relevance.evaluate(query, context)

    # Assertions
    assert metric_result == 1


def test_context_relevance_good():

    # Create an instance of ContextRelevance class
    context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)

    # Running the evaluate method
    query = "Who invented C++?"
    context = "bjarne stroustrup invented C++"
    metric_result, explanation = context_relevance.evaluate(query, context)

    # Assertions
    assert metric_result == 0
