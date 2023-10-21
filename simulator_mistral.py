# Import necessary libraries and modules
import os

from dotenv import load_dotenv

from redeval.simulators.gaslighting_simulator import GaslightingSimulator
from redeval.simulators.guilting_simulator import GuiltingSimulator
from redeval.simulators.prompt_injection_attack_simulator import PromptInjectionAttackSimulator
from redeval.simulators.research_purpose_simulator import ResearchPurposeSimulator

# Fetch OpenAI API key from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

n_turns = 5

PromptInjectionAttackSimulator(
    openai_api_key=openai_api_key, application="chat", model="mistral_7b", n_turns=5
).simulate()
ResearchPurposeSimulator(
    openai_api_key=openai_api_key, application="chat", model="mistral_7b", n_turns=n_turns
).simulate()
GaslightingSimulator(
    openai_api_key=openai_api_key, application="chat", model="mistral_7b", n_turns=n_turns
).simulate()
GuiltingSimulator(
    openai_api_key=openai_api_key, application="chat", model="mistral_7b", n_turns=n_turns
).simulate()
