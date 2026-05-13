# Import TypedDict for defining state structure
# Annotated is used to attach extra behavior to a type
from typing import TypedDict, Annotated

# Import tools from LangGraph
# add_messages helps manage chat history
# StateGraph creates a graph workflow
# END marks the end of the graph
from langgraph.graph import add_messages, StateGraph, END

# Import Groq LLM model wrapper
from langchain_groq import ChatGroq

# Import message types for AI and Human conversations
from langchain_core.messages import AIMessage, HumanMessage

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables (like API keys)
load_dotenv()

# Create the language model instance
# Using llama-3.1-8b-instant model from Groq
llm = ChatGroq(model="llama-3.1-8b-instant")


# Define the structure of the graph state
class BasicChatState(TypedDict):

    # Store messages list
    # add_messages helps append new messages automatically
    messages: Annotated[list, add_messages]


# Create chatbot function (graph node)
def chatbot(state: BasicChatState):

    # Send messages to LLM and return response
    return {
        "messages": [llm.invoke(state["messages"])]
    }


# Create a state graph using BasicChatState
graph = StateGraph(BasicChatState)

# Add chatbot node to graph
graph.add_node("chatbot", chatbot)

# Set chatbot as the starting point
graph.set_entry_point("chatbot")

# After chatbot runs, end the graph
graph.add_edge("chatbot", END)

# Compile graph into runnable app
app = graph.compile()


# Infinite chat loop
while True:

    # Get user input from terminal
    user_input = input("User: ")

    # Stop chatbot if user types exit or end
    if user_input in ["exit", "end"]:
        break

    else:

        # Invoke graph with user's message
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        # Print chatbot response
        print(result)