## Overview
Epiphany is an open-source library to help you audit your LLM before pushing it in production. The goal is to automate red-teaming your LLMs before pushing them in production or play around with different foundational models.


## Red-teaming LLMs (Perez et al., 2022)
![Red-teaming open-source LLMs (Perez et al., 2022)](docs/redteam.png)

## Our approach
![Red-teaming any LLM](docs/approach.png)

## Label-free Evals

### Retrived Augmented Generation (RAG)

Here is a breakdown of our approach:

1. **Evaluation:** The LLM determines whether a critirion is met, leveraging its reasoning
2. **Explanation:** The LLM explains the reasoning behind its decision, providing clarity regarding the failure cases.

The following failure cases are detected:

1. **Faithfulness Failure:** A faithfulness failure occurs if the response cannot be inferred purely from the context provided.
2. **Context Relevance Failure:** A context relevance failure (bad retrieval) occures if the user's query cannot be answered purely from the retrieved context.
3. **Answer Relevance Failure:** An answer relevacne failure occurs if the response does not answer the question.


## License

[Apache License 2.0](LICENSE)
