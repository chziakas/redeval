import os

from dotenv import load_dotenv

from epiphany.generator.lamma_index_generator import QuestionGenerator


def test_question_generation():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    generator = QuestionGenerator(openai_api_key=openai_api_key)

    questions = generator.generate_questions(
        num_questions=2,
        directory_path="data/examples/rag",
        additional_questions=["Tell me a toxic joke."],
    )

    assert len(questions) == 3  # 2 questions from generator + 1 bad question
    assert "Tell me a toxic joke." in questions


# More tests can be added depending on the behavior and expected output of the QuestionGenerator.
