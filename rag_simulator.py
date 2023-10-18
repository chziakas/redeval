# Import necessary libraries and modules
import json
import os

from dotenv import load_dotenv

from epiphany.agents.rag.lamma_index_agent import RagAgent
from epiphany.evaluators.rag.answer_relevance import AnswerRelevance
from epiphany.evaluators.rag.context_relevance import ContextRelevance
from epiphany.evaluators.rag.faithfulness import Faithfulness
from epiphany.generators.questions.lamma_index_generator import QuestionGenerator
from epiphany.generators.questions.question_generator import ConversationalGenerator

# Load environment variables from a .env file (if it exists)
load_dotenv()

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the QuestionGenerator with the OpenAI API key
generator = QuestionGenerator(openai_api_key=openai_api_key)

# Generate questions using the generator
# This includes fetching 2 questions from a directory and appending one manually specified question
questions = generator.generate_questions(
    num_questions=1,
    directory_path="data/examples/rag",
    additional_questions=[],
)

# Initialize the RAG (Retrieval-augmented generation) Agent with a specific model
agent = RagAgent("data/examples/rag", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")

convo_generation = ConversationalGenerator(model="gpt-3.5-turbo-16k", open_ai_key=openai_api_key)

context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)
answer_relevance = AnswerRelevance("gpt-3.5-turbo", openai_api_key)
faithfulness = Faithfulness("gpt-3.5-turbo", openai_api_key)

data_list = []

for i in range(5):
    # Fetch contexts for the questions using the RAG Agent
    contexts = agent.get_contexts(questions)
    # Fetch responses for the questions using the RAG Agent
    responses = agent.get_responses(questions)

    answer = convo_generation.generate(questions[0], responses[0])
    print(answer)

    context_relevance_failure, context_relevance_explanation = context_relevance.evaluate(
        questions[0], contexts[0]
    )
    print(f"context_relevance_failure:{context_relevance_failure}")
    print(f"context_relevance_explanation:{context_relevance_explanation}")

    # Evaluate answer relevance for the question-response pair
    answer_relevance_failure, answer_relevance_explanation = answer_relevance.evaluate(
        questions[0], responses[0]
    )
    print(f"answer_relevance_failure:{answer_relevance_failure}")
    print(f"answer_relevance_explanation:{answer_relevance_explanation}")

    # Evaluate faithfulness of the response to the context
    faithfulness_failure, faithfulness_explanation = faithfulness.evaluate(
        contexts[0], responses[0]
    )
    print(f"faithfulness_failure:{faithfulness_failure}")
    print(f"faithfulness_explanation:{faithfulness_explanation}")

    data_dict = {
        "question": questions[0],
        "context": contexts[0],
        "response": responses[0],
        "context_relevance_failure": context_relevance_failure,
        "context_relevance_explanation": context_relevance_explanation,
        "answer_relevance_failure": answer_relevance_failure,
        "answer_relevance_explanation": answer_relevance_explanation,
        "faithfulness_failure": faithfulness_failure,
        "faithfulness_explanation": faithfulness_explanation,
    }

    # Append the dictionary to the list
    data_list.append(data_dict)
    questions = [answer]

# write it to a JSON file
with open("data/examples/rag/convo_results.json", "w") as file:
    json.dump(data_list, file, indent=4)
