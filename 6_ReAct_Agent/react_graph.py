from dotenv import load_dotenv  # Imports function to load environment variables from .env file

load_dotenv()  # Loads environment variables like API keys from .env file

from langchain_core.agents import AgentFinish, AgentAction  # Imports agent result/action classes
from langgraph.graph import END, StateGraph  # Imports LangGraph END marker and graph builder

from nodes import reason_node, act_node  # Imports graph node functions
from react_state import AgentState  # Imports the graph state type


REASON_NODE = "reason_node"  # Name for the reasoning node
ACT_NODE = "act_node"  # Name for the action node


def should_continue(state: AgentState) -> str:  # Decides where the graph should go next
    if isinstance(state["agent_outcome"], AgentFinish):  # Checks if the agent has finished
        return END  # Ends the graph execution
    else:
        return ACT_NODE  # Continues to the action node


graph = StateGraph(AgentState)  # Creates a graph using AgentState as state schema

graph.add_node(REASON_NODE, reason_node)  # Adds the reasoning node to the graph
graph.set_entry_point(REASON_NODE)  # Sets reasoning node as the first node
graph.add_node(ACT_NODE, act_node)  # Adds the action node to the graph


graph.add_conditional_edges(  # Adds conditional routing after reasoning
    REASON_NODE,  # Starts conditional check from reasoning node
    should_continue,  # Uses function to decide next node
)

graph.add_edge(ACT_NODE, REASON_NODE)  # Sends control back to reasoning after action

app = graph.compile()  # Compiles the graph into a runnable app

result = app.invoke(  # Runs the graph with initial input state
    {
        "input": "How many days ago was the latest SpaceX launch?",  # User question
        "agent_outcome": None,  # No agent result at the start
        "intermediate_steps": []  # No previous tool steps at the start
    }
)

print(result["agent_outcome"].return_values["output"], "final result")  
# Prints the final answer from the agent