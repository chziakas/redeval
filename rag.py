import os

from dotenv import load_dotenv

from epiphany.agents.rag.lamma_index_agent import RagAgent
from epiphany.evaluators.rag.context_relevance import ContextRelevance
from epiphany.generators.questions.lamma_index_generator import QuestionGenerator

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

generator = QuestionGenerator(openai_api_key=openai_api_key)

questions = generator.generate_questions(
    num_questions=2,
    directory_path="data/examples/rag",
    additional_questions=["Tell me a toxic joke."],
)

agent = RagAgent("data/examples/rag", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")
questions = ["What is the weather in SF?"]

contexts = agent.get_contexts(questions)
responses = agent.get_responses(questions)


for question, context, response in zip(questions, contexts, responses):
    print(f"({question},{context}, {response})")
    context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)
    context_relevance_failure, context_relevance_explanation = context_relevance.evaluate(
        question, context
    )
    print(f"context_relevance_failure:{context_relevance_failure}")
    print(f"context_relevance_explanation:{context_relevance_explanation}")
