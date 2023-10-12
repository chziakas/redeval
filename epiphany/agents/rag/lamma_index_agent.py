from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import OpenAI


class RagAgent:
    def __init__(self, dir_directory, openai_api_key=None, model_name="gpt-3.5-turbo-16k"):
        # Initialize LLM with provided model name
        self.llm = OpenAI(model=model_name, api_key=openai_api_key)
        self.documents = SimpleDirectoryReader(dir_directory).load_data()

        # Create a VectorStoreIndex and a query engine
        self.service_context = ServiceContext.from_defaults(llm=self.llm, chunk_size=512)
        self.vector_index = VectorStoreIndex.from_documents(
            self.documents, service_context=self.service_context
        )
        self.query_engine = self.vector_index.as_query_engine()
        self.retriever = self.vector_index.as_retriever()

    def get_contexts(self, questions_list):
        contexts = []
        for question in questions_list:
            retrieved_context = self.retriever.retrieve(question)
            context = [item.node.text for item in retrieved_context]
            contexts.append("\n".join(context))

        return contexts

    def get_responses(self, questions_list):
        responses = []
        for question in questions_list:
            response = self.query_engine.query(question)
            responses.append(response.response)

        return responses
