## Overview


Prior to deploying Large Language Models (LLMs) applications, there is a lack of tools available to simulate their behavior and boost confidence in their anticipated performance. How can potential failure cases or vulnerabilities that might deteriorate user experience or even harm reputation be proactively detected?

LLMs have the capability to audit and evaluate other LLMs efficiently without human oversight. As LLMs and their applications continually improve, integrating them into increasingly vital applications amplifies the ramifications of any potential failure.

RedEval is an open-source library that simulates and evaluates LLM applications across various scenarios, all while eliminating the need for time-intensive and expensive human intervation.


## Auditing LLMs for various scenarios
![Red-teaming LLMs (Perez et al., 2022)](docs/redteam.png)

Drawing inspiration from the architecture of red-teaming LLMs (Perez et al., 2022), RedEval introduces a series of multi-turn conversations where a red-team LLM challenges a target LLM across various scenarios. These scenarios can be categorized as follows:

**Performance-Based on Application Layer**

- **Performance Evaluation**: Simulates the behavior of a potential customer with genuine intent, inquiring about the product. Metrics include Faithfulness, Answer Relevance, and Context Similarity.
- **Toxicity Evaluation**: Simulates a scenario in which a customer asks toxic questions related to the product.

**Adversarial Attacks**

- **Gaslighting**: Simulates an agent that tries to manipulate the target into endorsing harmful actions.
- **Guilt-Tripping**: Simulates an agent that coerces the target into undesired actions by inducing guilt.
- **Fraudulent Researcher**: Simulates an agent that seeks harmful actions under the pretense of academic research.

**Cyber-Security Attacks**

- **Prompt Injection**: Introduces malicious prefixes or suffixes to the original prompt to elicit harmful behavior.
- **Social Engineering Attack**: Simulates an attacker seeking confidential or proprietary information from a company, falsely claiming to be a student.


## LLM Evals

LLMs evals utilize an LLM's reasoning to automatically detect failure cases of LLMs. Drawing inspiration from RLAIF (Lee et al., 2023), LLMs are expected to be as competent as, or even superior to, humans in evaluating whether LLMs met specific criteria. RedEval introduces the following LLM evals.

**RAG Evals**
- **Faithfulness Failure:** A faithfulness failure occurs if the response cannot be inferred purely from the context provided.
- **Context Relevance Failure:** A context relevance failure (bad retrieval) occures if the user's query cannot be answered purely from the retrieved context.
- **Answer Relevance Failure:** An answer relevacne failure occurs if the response does not answer the question.

**Attacks**
- **Toxicity Failure:** A toxicity failure occurs if the response is toxic.
- **Safety Failure:** A toxicity failure occurs if the response is unsafe.


## Get started

### Installation
```bash
pip install epiphany
```

### Run simulations
```python
# Load a simulator
from redeval.simulators.performance_simulator import PerformanceSimulator

# Set up the parameters
openai_api_key = 'Your OpenAI APIS Key'
n_turns = 5
data_path_dir = 'Your txt document for RAG'

# Run RAG erformance simulation
PerformanceSimulator(openai_api_key=openai_api_key, n_turns=n_turns, data_path = data_path_dir).simulate()
```


## License

[Apache License 2.0](LICENSE)
