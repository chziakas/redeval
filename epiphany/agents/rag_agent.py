from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate


class RagAgent:

    PROMPT = PromptTemplate(
        """
        You are a helpful chatbot. Given the context information and not prior knowledge, answer the received message.
        Message: {message}
        Context: {context_str}
        Answer:
        """
    )

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

    def get_response(self, question):
        retrieved_context = self.retriever.retrieve(question)
        context_str = "\n\n".join([r.get_content() for r in retrieved_context])
        fmt_qa_prompt = self.PROMPT.format(context_str=context_str, message=question)
        response = self.llm.complete(fmt_qa_prompt)
        return str(response), context_str
