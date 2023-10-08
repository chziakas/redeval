from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext

load_dotenv()

def rag_agent(dir_directory, questions_list):
    # Initialize LLM
    llm = OpenAI(model="gpt-3.5-turbo-16k")
    documents = SimpleDirectoryReader(dir_directory).load_data()

    # Create a VectorStoreIndex and a query engine
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = vector_index.as_query_engine()

    responses = []
    for question in questions_list:
        response_vector = query_engine.query(question)
        responses.append(response_vector)
        
    return responses

# Set up the directory and questions
dir_directory = 'data/examples/rag'
questions_list = ['What is the weather in SF?']

# Get responses for the questions
responses = rag_agent(dir_directory, questions_list)
print(responses)
