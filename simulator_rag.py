# Import necessary libraries and modules
import os

from dotenv import load_dotenv

from redeval.simulators.gaslighting_simulator import GaslightingSimulator
from redeval.simulators.guilting_simulator import GuiltingSimulator
from redeval.simulators.performance_simulator import PerformanceSimulator

from redeval.simulators.prompt_injection_attack_simulator import PromptInjectionAttackSimulator
from redeval.simulators.research_purpose_simulator import ResearchPurposeSimulator
from redeval.simulators.social_engineering_attack_simulator import SocialEngineeringAttackSimulator
from redeval.simulators.toxicity_simulator import ToxicitySimulator

# Load environment variables from a .env file (if it exists)


load_dotenv()

# Fetch OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
n_turns = 2

# Let's run the rag use-cases
# Performance evaluation
simulator = PerformanceSimulator(openai_api_key=openai_api_key, n_turns=n_turns).simulate()
simulator = ToxicitySimulator(openai_api_key=openai_api_key, n_turns=n_turns).simulate()

# Convince an agent to do something for you
ResearchPurposeSimulator(
    openai_api_key=openai_api_key, application="rag", n_turns=n_turns
).simulate()
GaslightingSimulator(openai_api_key=openai_api_key, application="rag", n_turns=n_turns).simulate()
GuiltingSimulator(openai_api_key=openai_api_key, application="rag", n_turns=n_turns).simulate()

# Attacks
PromptInjectionAttackSimulator(openai_api_key=openai_api_key, n_turns=5).simulate()
receipient = "Bob"
writer = "Alice"
receipient_role = "Data Scientist"
receipient_company = "Airbnb"
SocialEngineeringAttackSimulator(
    openai_api_key=openai_api_key,
    receipient=receipient,
    writer=writer,
    receipient_role=receipient_role,
    receipient_company=receipient_company,
    n_turns=n_turns,
).simulate()
