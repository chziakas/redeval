# Import necessary libraries and modules
import json
from datetime import datetime

from redeval.agents.rag_agent import RagAgent
from redeval.evaluators.toxicity import Toxicity
from redeval.generators.questions.toxic_conversational_generator import ToxicConversationalGenerator
from redeval.generators.questions.toxic_question_generator import ToxicQuestionGenerator


class ToxicitySimulator:
    def __init__(
        self,
        openai_api_key,
        n_turns=3,
        context_file="data/examples/company/airbnb.txt",
        data_path="data/examples/company",
        eval_path=None,
    ):
        # Initialize various agents and generators with provided parameters
        self.initial_generator = ToxicQuestionGenerator(
            model="gpt-3.5-turbo-16k", open_ai_key=openai_api_key
        )
        self.agent = RagAgent(
            "data/examples/company", openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k"
        )
        self.conversation_generator = ToxicConversationalGenerator(
            model="gpt-3.5-turbo-16k", open_ai_key=openai_api_key
        )
        self.toxicity = Toxicity("gpt-3.5-turbo", openai_api_key)

        self.n_turns = n_turns
        with open(context_file, "r") as file:
            self.context = file.read()
        self.initial_question = self.initial_generator.generate(context=self.context)
        self.data_list = []
        self.eval_path = eval_path
        if eval_path is None:
            self.eval_path = f"data/examples/{str(self.__class__.__name__)}_simulation.json"

    def simulate(self):
        # Simulate a conversation for n_turns and collect evaluation data
        question = self.initial_question
        print(question)
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
        toxicity_failure, toxicity_explanation = self.toxicity.evaluate(question, response)
        current_time = datetime.now().timestamp()
        metric = "Toxicity"
        data_dict = {
            "question": question,
            "context": context,
            "response": response,
            "evaluations": [
                {
                    "metric": metric,
                    "eval_explanation": toxicity_explanation,
                    "is_failure": toxicity_failure,
                }
            ],
            "date_created": int(current_time),
        }
        return data_dict
