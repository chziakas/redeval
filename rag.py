# Import necessary libraries and modules
import json
import os

from dotenv import load_dotenv

from epiphany.agents.rag.lamma_index_agent import RagAgent
from epiphany.evaluators.rag.answer_relevance import AnswerRelevance
from epiphany.evaluators.rag.context_relevance import ContextRelevance
from epiphany.evaluators.rag.faithfulness import Faithfulness
from epiphany.generators.questions.lamma_index_generator import QuestionGenerator

# Load environment variables from a .env file (if it exists)
load_dotenv()

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the QuestionGenerator with the OpenAI API key
generator = QuestionGenerator(openai_api_key=openai_api_key)

# Generate questions using the generator
# This includes fetching 2 questions from a directory and appending one manually specified question
questions = generator.generate_questions(
    num_questions=2,
    directory_path="data/examples/rag",
    additional_questions=["Tell me a bad joke."],
)

# Initialize the RAG (Retrieval-augmented generation) Agent with a specific model
agent = RagAgent("data/examples/rag", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")

# Fetch contexts for the questions using the RAG Agent
contexts = agent.get_contexts(questions)

# Fetch responses for the questions using the RAG Agent
responses = agent.get_responses(questions)

# Create a list to store each set of data as a dictionary
data_list = []

# Iterate over each question, context, and response
for question, context, response in zip(questions, contexts, responses):
    print(f"({question},{context}, {response})")

    # Evaluate context relevance for the question-context pair
    context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)
    context_relevance_failure, context_relevance_explanation = context_relevance.evaluate(
        question, context
    )
    print(f"context_relevance_failure:{context_relevance_failure}")
    print(f"context_relevance_explanation:{context_relevance_explanation}")

    # Evaluate answer relevance for the question-response pair
    answer_relevance = AnswerRelevance("gpt-3.5-turbo", openai_api_key)
    answer_relevance_failure, answer_relevance_explanation = answer_relevance.evaluate(
        question, response
    )
    print(f"answer_relevance_failure:{answer_relevance_failure}")
    print(f"answer_relevance_explanation:{answer_relevance_explanation}")

    # Evaluate faithfulness of the response to the context
    faithfulness = Faithfulness("gpt-3.5-turbo", openai_api_key)
    faithfulness_failure, faithfulness_explanation = faithfulness.evaluate(context, response)
    print(f"faithfulness_failure:{faithfulness_failure}")
    print(f"faithfulness_explanation:{faithfulness_explanation}")

    # Create a dictionary for the current set of data
    data_dict = {
        "question": question,
        "context": context,
        "response": response,
        "context_relevance_failure": context_relevance_failure,
        "context_relevance_explanation": context_relevance_explanation,
        "answer_relevance_failure": answer_relevance_failure,
        "answer_relevance_explanation": answer_relevance_explanation,
        "faithfulness_failure": faithfulness_failure,
        "faithfulness_explanation": faithfulness_explanation,
    }

    # Append the dictionary to the list
    data_list.append(data_dict)


# write it to a JSON file
with open("data/examples/rag/results.json", "w") as file:
    json.dump(data_list, file, indent=4)
