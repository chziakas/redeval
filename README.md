## 💡 Overview

❓ **Problem**

Before deploying Large Language Models (LLMs) in real-world applications, a significant challenge is ensuring the robustness and reliability of these systems. How can we proactively spot vulnerabilities or failure scenarios that can negatively affect user experience or cause material harm? As LLMs are gaining more emerging abilities and integrating into vital applications, any potential failure case might have even more negative consequences.

⚡ **RedEval**

RedEval is an open-source library that simulates and evaluates LLM applications across various scenarios, all while eliminating the need for human intervention. With the expansion of LLMs' capabilities due to scaling laws, LLMs will be soon have the capability to audit and evaluate other LLMs effectively without human oversight.


## 🎮 Auditing LLMs
![Red-teaming LLMs (Perez et al., 2022)](docs/redteam.png)

Drawing inspiration from the architecture of red-teaming LLMs (Perez et al., 2022), RedEval introduces a series of multi-turn conversations where a red-team LLM challenges a target LLM across various scenarios. These scenarios can be categorized as follows:

**Performance Simulations**

- **Performance Evaluation**: Simulates the behavior of a potential customer with genuine intent, inquiring about the product.
- **Toxicity Evaluation**: Simulates a scenario in which a customer asks toxic questions related to the product.

**Manipulation Tactics**

- **Gaslighting**: Simulates an agent that tries to manipulate the target into endorsing harmful actions.
- **Guilt-Tripping**: Simulates an agent that coerces the target into undesired actions by inducing guilt.


**Deception Tactics**
- **Fraudulent Researcher**: Simulates an agent that seeks harmful actions under the pretense of academic research.
- **Social Engineering Attack**: Simulates an attacker seeking confidential or proprietary information from a company, falsely claiming to be a student.

**Adversarial Attacks**

- **Prompt Injection**: Introduces malicious prefixes or suffixes to the original prompt to elicit harmful behavior.


## 🔍 LLM Evals

LLM evaluations leverage the reasoning capabilities of LLMs to identify and elucidate the failure scenarios of LLMs. Inspired by the observation that RLAIF demonstrated performance comparable to RLHF (Lee et al., 2023), we anticipate that LLM evaluations will soon match or even surpass human competence in assessing whether LLMs meet specific benchmarks. RedEval offers the following LLM evaluations:

**RAG Evals**
- **Faithfulness Failure:** A faithfulness failure occurs if the response cannot be inferred purely from the context provided.
- **Context Relevance Failure:** A context relevance failure (bad retrieval) occurs if the user's query cannot be answered purely from the retrieved context.
- **Answer Relevance Failure:** An answer relevance failure occurs if the response does not answer the question.

**Attacks**
- **Toxicity Failure:** A toxicity failure occurs if the response is toxic.
- **Safety Failure:** A toxicity failure occurs if the response is unsafe.


## Get started

### Installation
```bash
pip install redeval
```

### Run simulations
```python
# Load a simulator
from redeval.simulators.performance_simulator import PerformanceSimulator

# Set up the parameters
openai_api_key = 'Your OpenAI APIS Key'
n_turns = 5
data_path_dir = 'Your txt document for RAG'

# Run RAG performance simulation
PerformanceSimulator(openai_api_key=openai_api_key, n_turns=n_turns, data_path = data_path_dir).simulate()
```


## License

[Apache License 2.0](LICENSE)
