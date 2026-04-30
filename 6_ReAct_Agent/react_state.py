import operator  # Imports operator module (used for combining lists)
from typing import Annotated, TypedDict, Union  # Imports typing helpers

from langchain_core.agents import AgentAction, AgentFinish  # Imports agent result types

class AgentState(TypedDict):  # Defines a structured dictionary for agent state
    input: str  # Stores the user input string
    
    agent_outcome: Union[AgentAction, AgentFinish, None]  
    # Stores the agent's result (either an action, final answer, or None)
    
    intermediate_steps: Annotated[
        list[tuple[AgentAction, str]], operator.add
    ]  
    # Stores a list of (action, result) steps and combines them using operator.add