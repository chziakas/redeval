from dotenv import load_dotenv
from llama_index import ServiceContext
from llama_index.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms import OpenAI

load_dotenv()


def evaluate_responses(questions_list, response_vectors):
    # We will use GPT-4 for evaluating the responses
    gpt4 = OpenAI(temperature=0, model="gpt-4")

    # Define service context for GPT-4 for evaluation
    service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

    # Define Faithfulness and Relevancy Evaluators which are based on GPT-4
    faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
    relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)

    failed_questions_faith = []
    failed_questions_relevance = []

    for question, response_vector in zip(questions_list, response_vectors):
        faithfulness_result = faithfulness_gpt4.evaluate_response(response=response_vector)

        relevancy_result = relevancy_gpt4.evaluate_response(
            query=question, response=response_vector
        )

        if not faithfulness_result.passing:
            failed_questions_faith.append(question)

        if not relevancy_result.passing:
            failed_questions_relevance.append(question)

    return failed_questions_relevance, failed_questions_faith


# Example usage
questions_list = ["Question1", "Question2"]  # Replace with your list of questions
response_vectors = []  # Replace with your list of response vectors

failed_questions_relevance, failed_questions_faith = evaluate_responses(
    questions_list, response_vectors
)
