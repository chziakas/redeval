from redeval.evaluators.open_ai_completion import OpenAICompletion


class ConversationalGenerator:
    """
    This class generates a follow-up question

    Attributes:
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
    """

    SYSTEM_MESSAGE = """
        You are an expert at simulating a conversation between a human and a chatbot.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following:
        previous conversation between human and chatbot {}.
        chatbot's last answer: {}.
        2. Generate a question to continue a natural converastion between the chatbot's last answe, labeled as 'question', based on the chatbot's answer and previous conversation.
        3. Return a JSON object in the following format: "question": 'question.
    """

    def __init__(self, model, open_ai_key):
        """
        Initialize the QuestionAnswerer class.
        """
        self.openAIcompletion = OpenAICompletion(model, open_ai_key)
        self.memory = ""

    def generate(self, query: str, answer: str):
        """
        Evaluation for is response faithful to context
        """
        self.memory += "Human: " + query + "\n"
        user_message = self.USER_MESSAGE_TEMPLATE.format(self.memory, answer)
        system_message = self.SYSTEM_MESSAGE
        message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        openai_response = self.openAIcompletion.get_completion_from_messages(message)
        openai_response_json = self.openAIcompletion.extract_json_from_response(openai_response)
        question = openai_response_json["question"]
        self.memory += "Chatbot: " + answer + "\n"
        return question
