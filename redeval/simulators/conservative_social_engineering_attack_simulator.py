# Import necessary libraries and modules
import json
from datetime import datetime

from redeval.agents.conservative_human_agent import CosnervativeHumanAgent
from redeval.evaluators.answer_relevance import AnswerRelevance
from redeval.evaluators.context_relevance import ContextRelevance
from redeval.evaluators.faithfulness import Faithfulness
from redeval.generators.questions.conversational_generator import ConversationalGenerator
from redeval.generators.questions.social_engineering_attack_question_generator import (
    SocialEngineeringAttackQuestionGenerator,
)


class ConservativeSocialEngineeringAttackSimulator:
    def __init__(
        self,
        openai_api_key,
        n_turns=3,
        num_questions=1,
        context_file="data/examples/company/airbnb.txt",
        data_path="data/examples/company",
        eval_path="data/examples/conservative_social_attack_simulation.json",
        additional_questions=[],
    ):
        # Initialize various agents and generators with provided parameters
        with open(context_file, "r") as file:
            context = file.read()
        self.initial_generator = SocialEngineeringAttackQuestionGenerator(
            model="gpt-3.5-turbo-16k", open_ai_key=openai_api_key
        )
        self.agent = CosnervativeHumanAgent(
            "data/examples/company", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k"
        )
        self.conversation_generator = ConversationalGenerator(
            model="gpt-3.5-turbo-16k", open_ai_key=openai_api_key
        )
        self.context_relevance = ContextRelevance("gpt-3.5-turbo", openai_api_key)
        self.answer_relevance = AnswerRelevance("gpt-3.5-turbo", openai_api_key)
        self.faithfulness = Faithfulness("gpt-3.5-turbo", openai_api_key)

        self.n_turns = n_turns
        self.initial_question = self.initial_generator.generate(
            receipient="Christos",
            writer="Bob",
            role="Data Scientist",
            company="Athina",
            context=context,
        )
        self.data_list = []
        self.eval_path = eval_path

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
        # Evaluate the response on various criteria
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
            "date_created": int(current_time),
        }

        return data_dict
