from redeval.llms.mistral_7b_completion import Mistral7bCompletion
from redeval.llms.open_ai_completion import OpenAICompletion

# Uncomment the following line if Llama70bCompletion is ever required and imported.



class Chatbot:
    """
    This class provides functionality to generate responses to user queries.
    Depending on the model specified during initialization, it interacts with
    different backends such as OpenAI, Mistral7b, or Llama70b (if supported in the future).

    Attributes:
        agent (Union[OpenAICompletion, Mistral7bCompletion, Llama70bCompletion]):
        The model instance used for generating responses.
    """

    SYSTEM_MESSAGE = "You are a chatbot."

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step:
        1. Consider the following:
        - question: {}
        2. Respond to the question.
    """

    def __init__(self, model: str, open_ai_key: str):
        """
        Initialize the Chatbot class with the specified model.

        Args:
            model (str): Model name to determine which backend to use for responses.
            open_ai_key (str): API key for OpenAI (only required if using OpenAI as backend).
        """
        if model == "llama2_70b":
            self.agent = Mistral7bCompletion()
            # Uncomment the following line if Llama70bCompletion is ever supported.
            # self.agent = Llama270bCompletion()
        elif model == "mistral_7b":
            self.agent = Mistral7bCompletion()
        else:
            self.agent = OpenAICompletion(model, open_ai_key)

    def generate(self, query: str) -> str:
        """
        Generate a response to the given user query.

        Args:
            query (str): The user's question or input.

        Returns:
            str: The generated response.
        """
        # Formulate message differently based on the type of agent (model backend).
        if isinstance(self.agent, OpenAICompletion):
            user_message = self.USER_MESSAGE_TEMPLATE.format(query)
            system_message = self.SYSTEM_MESSAGE
            message = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        else:
            message = query

        response = self.agent.get_completion_from_messages(message)

        return response
