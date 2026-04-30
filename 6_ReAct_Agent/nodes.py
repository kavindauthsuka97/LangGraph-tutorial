from dotenv import load_dotenv  # Imports function to load environment variables from .env file

from agent_reason_runnable import react_agent_runnable, tools  
# Imports the ReAct agent runnable and available tools

from react_state import AgentState  
# Imports the AgentState type definition

load_dotenv()  # Loads environment variables (e.g., API keys) from .env file


def reason_node(state: AgentState):  # Defines the reasoning step of the agent
    agent_outcome = react_agent_runnable.invoke(state)  
    # Runs the agent with current state to decide next action or final answer
    
    return {"agent_outcome": agent_outcome}  
    # Returns the agent's decision (action or finish)


def act_node(state: AgentState):  # Defines the action execution step
    agent_action = state["agent_outcome"]  
    # Gets the action decided by the agent
    
    # Extract tool name and input from AgentAction
    tool_name = agent_action.tool  # Name of the tool to call
    tool_input = agent_action.tool_input  # Input arguments for the tool
    
    # Find the matching tool function
    tool_function = None  # Initialize tool function as None
    for tool in tools:  # Loop through available tools
        if tool.name == tool_name:  # Match tool by name
            tool_function = tool  # Assign matching tool
            break  # Stop loop once found
    
    # Execute the tool with the input
    if tool_function:  # If a matching tool is found
        if isinstance(tool_input, dict):  # If input is a dictionary
            output = tool_function.invoke(**tool_input)  
            # Call tool with keyword arguments
        else:
            output = tool_function.invoke(tool_input)  
            # Call tool with single argument
    else:
        output = f"Tool '{tool_name}' not found"  
        # Handle case where tool is missing
    
    return {"intermediate_steps": [(agent_action, str(output))]}  
    # Returns updated intermediate steps with action and its result