# Import necessary libraries and modules
import json
from datetime import datetime

from redeval.agents.rag_agent import RagAgent
from redeval.evaluators.answer_relevance import AnswerRelevance
from redeval.evaluators.context_relevance import ContextRelevance
from redeval.evaluators.faithfulness import Faithfulness
from redeval.generators.questions.conversational_generator import ConversationalGenerator
from redeval.generators.questions.regular_question_generator import RegularQuestionGenerator


class PerformanceSimulator:
    def __init__(
        self,
        openai_api_key,
        n_turns=5,
        num_questions=1,
        data_path="data/examples/company",
        eval_path=None,
        additional_questions=[],
    ):
        # Initialize various agents and generators with provided parameters
        self.initial_generator = RegularQuestionGenerator(
            model_name="gpt-3.5-turbo-16k", temperature=1.0, openai_api_key=openai_api_key
        )
        self.agent = RagAgent(
            "data/examples/company", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k"
        )
        self.conversation_generator = ConversationalGenerator(
            model="gpt-3.5-turbo-16k", open_ai_key=openai_api_key
        )
        self.context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)
        self.answer_relevance = AnswerRelevance("gpt-3.5-turbo", openai_api_key)
        self.faithfulness = Faithfulness("gpt-3.5-turbo", openai_api_key)

        self.n_turns = n_turns
        self.questions = self.initial_generator.generate_questions(
            num_questions=num_questions,
            directory_path=data_path,
            additional_questions=additional_questions,
        )
        self.initial_question = self.questions[0]
        self.data_list = []
        self.eval_path = eval_path
        if eval_path is None:
            self.eval_path = f"data/examples/{str(self.__class__.__name__)}_simulation.json"

    def simulate(self):
        # Simulate a conversation for n_turns and collect evaluation data
        question = self.initial_question
        for _ in range(self.n_turns):
            response, context = self.agent.get_response(question)
            print(response)
            next_question = self.conversation_generator.generate(question, response)

            data_dict = self.evaluate_performance(question, response, context)
            self.data_list.append(data_dict)

            question = next_question

        # Save the simulation results to a JSON file
        with open(self.eval_path, "w") as file:
            json.dump(self.data_list, file, indent=4)

    def evaluate_performance(self, question, response, context):
        # Evaluate the response on various criteria like faithfulness, relevance, etc.
        faithfulness_failure, faithfulness_explanation = self.faithfulness.evaluate(
            context, response
        )
        answer_relevance_failure, answer_relevance_explanation = self.answer_relevance.evaluate(
            question, response
        )
        context_relevance_failure, context_relevance_explanation = self.context_relevance.evaluate(
            question, context
        )
        current_time = datetime.now().timestamp()

        metric_faithfulness = "Faithfulness"
        metric_answer_relevance = "Answer Relevance"
        metric_context_relevance = "Context Relevance"

        data_dict = {
            "question": question,
            "context": context,
            "response": response,
            "evaluations": [
                {
                    "metric": metric_faithfulness,
                    "eval_explanation": faithfulness_explanation,
                    "is_failure": faithfulness_failure,
                },
                {
                    "metric": metric_answer_relevance,
                    "eval_explanation": answer_relevance_explanation,
                    "is_failure": answer_relevance_failure,
                },
                {
                    "metric": metric_context_relevance,
                    "eval_explanation": context_relevance_explanation,
                    "is_failure": context_relevance_failure,
                },
            ],
            "date_created": int(current_time),
        }

        return data_dict
