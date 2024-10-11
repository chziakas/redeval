import os

from dotenv import load_dotenv

from redeval.evaluators.faithfulness import Faithfulness

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def test_context_relevance_bad():
    # Create an instance of Faithfulness class
    faithfulness = Faithfulness("gpt-3.5-turbo", openai_api_key)

    # Running the evaluate method
    context = "bjarne stroustrup invented C++"
    answer = "bjarne stroustrup invented Linux"
    metric_result, explanation = faithfulness.evaluate(context, answer)

    # Assertions
    assert metric_result == 1


def test_context_relevance_good():

    # Create an instance of Faithfulness class
    faithfulness = Faithfulness("gpt-3.5-turbo", openai_api_key)

    # Running the evaluate method
    context = "bjarne stroustrup invented C++"
    answer = "bjarne stroustrup invented C++"
    metric_result, explanation = faithfulness.evaluate(context, answer)

    # Assertions
    assert metric_result == 0
