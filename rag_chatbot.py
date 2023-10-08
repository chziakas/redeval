import json
from dotenv import load_dotenv
from llama_index.llms import OpenAI
import os
from llama_index.prompts import PromptTemplate
from llama_index import (
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)

load_dotenv()

# We will use GPT-4 for evaluating the responses
gpt4 = OpenAI(temperature=0, model="gpt-4")

# Define service context for GPT-4 for evaluation
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

# Define Faithfulness and Relevancy Evaluators which are based on GPT-4
faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)


def generate_response_text(retrieved_nodes, query_str, qa_prompt, llm):
    context_str = "\n\n".join([r.get_content() for r in retrieved_nodes])
    fmt_qa_prompt = qa_prompt.format(context_str=context_str, query_str=query_str)
    response = llm.complete(fmt_qa_prompt)
    return str(response), fmt_qa_prompt

def ask_directory_questions(dir_directory, questions):
    llm = OpenAI(model="gpt-3.5-turbo-16k")
    documents = SimpleDirectoryReader(dir_directory).load_data()

    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    
    query_engine = vector_index.as_query_engine()

    qa_prompt = PromptTemplate(
        """\
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer: \
    """
    )

    failed_questions_relevance = []
    failed_questions_faith =[]

    for question in questions:
        response_vector = query_engine.query(question)
        #print(response_vector)
        faithfulness_result = faithfulness_gpt4.evaluate_response(
            response=response_vector
        )
        
        relevancy_result = relevancy_gpt4.evaluate_response(
            query=question, response=response_vector
        )
        print(f'Question:{question} \n')
        print(f'Passed Faithfullness:{faithfulness_result.passing} \n')
        print(f'Passed Question-Context Relevance:{relevancy_result.passing} \n')
        if(faithfulness_result.passing == False):
            failed_questions_faith.append(question)
        if(relevancy_result.passing == False):
            failed_questions_relevance.append(question)

    return(failed_questions_relevance, failed_questions_faith)


file_path = os.path.abspath("data/examples/rag")

# Use file_path in your code
reader = SimpleDirectoryReader(file_path)



documents = reader.load_data()
data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes(num = 2)
bad_questions = ['What is the weather in SF?']
dir_directory = 'data/examples/rag' 

print(f'Generated Questions:{eval_questions} \n')
print(f'Ad-hoc Questions:{bad_questions} \n')
questions_list = eval_questions + bad_questions
print(questions_list)
failed_questions_relevance, failed_questions_faith = ask_directory_questions(dir_directory, questions_list)


from epiphany.question_mutator import generate_mutations_for_question
for question in failed_questions_faith:
    mutated_questions =[]
    mutations = generate_mutations_for_question(question, 3)
    for i, mutation in enumerate(mutations):
        print(f"Mutation {i}: {mutation}")
        mutated_questions.append(mutation)

print(f'Generated Mutated Questions:{mutated_questions} \n')
failed_questions_relevance, failed_questions_faith = ask_directory_questions(dir_directory, mutated_questions)
print(failed_questions_relevance)
print(failed_questions_faith)