## Overview

As companies increasingly deploy Large Language Model (LLM) applications, a clear challenge emerges: How can they feel confident about their performance before full-scale deployment? Is there a way to preemptively identify issues that might negatively impact the user experience or potentially harm a company's reputation?

Interestingly, LLMs can be employed to assess the performance of other LLMs, removing the need for human intervation. As LLMs and their applications continue to improve, their inclusion in more vital roles intensifies the implications of any potential missteps.

Epiphany is an open-source library designed to simulate and evaluate LLM applications in diverse scenarios, without the need for human intervation.

## Solution (Perez et al., 2022)
![Red-teaming LLMs (Perez et al., 2022)](docs/redteam.png)

Drawing inspiration from the architecture of red-teaming LLMs (Perez et al., 2022), Epiphany introduces a series of multi-turn conversations where a red-team LLM challenges a target LLM across various scenarios. These scenarios can be categorized as follows:

**Performance-Based on Application Layer (e.g., RAG)**

- Performance Evaluation**: Simulates the behavior of a potential customer with genuine intent, inquiring about the product. Metrics include Faithfulness, Answer Relevance, and Context Similarity.
- Toxicity Evaluation: Simulates a scenario in which a customer asks toxic questions related to the product.

**Adversarial Attacks**

- Gaslighting: Simulates an agent that tries to manipulate the target into endorsing harmful actions.
- Guilt-Tripping: Simulates an agent that coerces the target into undesired actions by inducing guilt.
- Fraudulent Researcher: Simulates an agent that seeks harmful actions under the pretense of academic research.

**Cyber-Security Attacks**

- Prompt Injection: Introduces malicious prefixes or suffixes to the original prompt to elicit harmful behavior.
- Social Engineering Attack: Simulates an attacker seeking confidential or proprietary information from a company, falsely claiming to be a student.



## LLM Evals

### Retrived Augmented Generation (RAG)

Inspired by the performance of RLAIF (Lee et al., 2023), LLMs could be as good as humans to judge an LLM action. Epiphany introduces the following LLM evals that could be used to automatically detect failure cases.

**RAG Evals**
- **Faithfulness Failure:** A faithfulness failure occurs if the response cannot be inferred purely from the context provided.
- **Context Relevance Failure:** A context relevance failure (bad retrieval) occures if the user's query cannot be answered purely from the retrieved context.
- **Answer Relevance Failure:** An answer relevacne failure occurs if the response does not answer the question.

**Attacks**
- **Toxicity Failure:** A toxicity failure occurs if the response is toxic.
- **Safety Failure:** A toxicity failure occurs if the response is unsafe.

## Get started

```bash
pip install ariadne-ai
```
or

```python
 poetry run python simulator_rag.py
 poetry run python simulator_mistral.py
```


## License

[Apache License 2.0](LICENSE)
