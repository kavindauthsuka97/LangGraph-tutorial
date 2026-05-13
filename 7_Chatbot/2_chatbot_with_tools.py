# Import TypedDict to define state structure
# Annotated adds extra functionality to types
from typing import TypedDict, Annotated

# Import LangGraph tools
# add_messages manages message history
# StateGraph creates workflow graph
# END marks graph ending
from langgraph.graph import add_messages, StateGraph, END

# Import Groq LLM wrapper
from langchain_groq import ChatGroq

# Import message types for chat
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
from dotenv import load_dotenv

# Import Tavily web search tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Import ToolNode to execute tools in graph
from langgraph.prebuilt import ToolNode


# Load environment variables (API keys)
load_dotenv()


# Define chatbot state structure
class BasicChatBot(TypedDict):

    # Store chat messages
    # add_messages automatically updates message history
    messages: Annotated[list, add_messages]


# Create Tavily search tool
# max_results=2 means return only 2 search results
search_tool = TavilySearchResults(max_results=2)

# Put tools inside a list
tools = [search_tool]


# Create Groq LLM model
llm = ChatGroq(model="llama-3.1-8b-instant")


# Bind tools to the LLM
# This allows the model to call tools when needed
llm_with_tools = llm.bind_tools(tools=tools)


# Chatbot node function
def chatbot(state: BasicChatBot):

    # Send conversation messages to LLM
    # LLM decides whether to answer normally or use a tool
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }


# Function to decide routing
def tools_router(state: BasicChatBot):

    # Get the latest message
    last_message = state["messages"][-1]

    # Check if LLM requested a tool call
    if (
        hasattr(last_message, "tool_calls") and
        len(last_message.tool_calls) > 0
    ):

        # Go to tool node
        return "tool_node"

    else:

        # End graph if no tool needed
        return END


# Create tool execution node
tool_node = ToolNode(tools=tools)


# Create graph
graph = StateGraph(BasicChatBot)


# Add chatbot node
graph.add_node("chatbot", chatbot)

# Add tool node
graph.add_node("tool_node", tool_node)

# Set chatbot as starting node
graph.set_entry_point("chatbot")


# Add conditional routing
# Decide whether to go to tool_node or END
graph.add_conditional_edges("chatbot", tools_router)


# After tool execution, go back to chatbot
graph.add_edge("tool_node", "chatbot")


# Compile graph into runnable app
app = graph.compile()


# Start chat loop
while True:

    # Take user input
    user_input = input("User: ")

    # Exit if user types exit or end
    if user_input in ["exit", "end"]:
        break

    else:

        # Send user message to graph
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        # Print chatbot response
        print(result)