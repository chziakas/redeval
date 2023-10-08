#from .mutator import generate_mutations_for_question
from .question_generation import generate_questions
from .rag_agent import rag_agent
from .evaluation import evaluate_responses

questions = generate_questions(num_questions = 2, directory_path = 'data/examples/rag')
print(questions)


# Set up the directory and questions
dir_directory = 'data/examples/rag'


# Get responses for the questions
responses = rag_agent(dir_directory, questions)

failed_questions_relevance, failed_questions_faith = evaluate_responses(questions, responses)







from epiphany.question_mutator import generate_mutations_for_question
for question in failed_questions_faith:
    mutated_questions =[]
    mutations = generate_mutations_for_question(question, 3)
    for i, mutation in enumerate(mutations):
        print(f"Mutation {i}: {mutation}")
        mutated_questions.append(mutation)

failed_questions_relevance, failed_questions_faith = ask_directory_questions(dir_directory, mutated_questions)
print(failed_questions_relevance)
print(failed_questions_faith)