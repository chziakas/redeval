from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader
from llama_index.evaluation import DatasetGenerator
from llama_index.llms import OpenAI

load_dotenv()

# Initialization of GPT-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)


def generate_questions(num_questions, directory_path):
    reader = SimpleDirectoryReader(directory_path)
    documents = reader.load_data()
    data_generator = DatasetGenerator.from_documents(documents)

    # Generate questions
    eval_questions = data_generator.generate_questions_from_nodes(num=num_questions)

    # You can add any 'bad' or additional questions here
    bad_questions = ["What is the weather in SF?"]

    # Combine the lists
    questions_list = eval_questions + bad_questions

    return questions_list


# Example use:
questions = generate_questions(num_questions=2, directory_path="data/examples/rag")
print(questions)
