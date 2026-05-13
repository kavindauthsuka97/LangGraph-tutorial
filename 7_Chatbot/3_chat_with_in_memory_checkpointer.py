# Import TypedDict for state structure
# Annotated adds extra behavior to the type
from typing import TypedDict, Annotated

# Import LangGraph tools
# add_messages manages chat history
# StateGraph creates graph workflow
# END marks workflow ending
from langgraph.graph import add_messages, StateGraph, END

# Import Groq LLM wrapper
from langchain_groq import ChatGroq

# Import chat message types
from langchain_core.messages import AIMessage, HumanMessage

# Import dotenv to load environment variables
from dotenv import load_dotenv

# Import MemorySaver for storing chat memory
from langgraph.checkpoint.memory import MemorySaver


# Load environment variables (.env file)
load_dotenv()


# Create memory object
# This stores conversation history
memory = MemorySaver()


# Create LLM model
llm = ChatGroq(model="llama-3.1-8b-instant")


# Define chatbot state structure
class BasicChatState(TypedDict):

    # Store messages list
    # add_messages automatically appends messages
    messages: Annotated[list, add_messages]


# Chatbot node function
def chatbot(state: BasicChatState):

    # Send conversation history to LLM
    # Return AI response
    return {
       "messages": [llm.invoke(state["messages"])]
    }


# Create graph with state type
graph = StateGraph(BasicChatState)


# Add chatbot node to graph
graph.add_node("chatbot", chatbot)


# End graph after chatbot runs
graph.add_edge("chatbot", END)


# Set chatbot as entry point
graph.set_entry_point("chatbot")


# Compile graph with memory checkpoint
# MemorySaver stores previous conversations
app = graph.compile(checkpointer=memory)


# Configuration settings
config = {
    "configurable": {

        # Unique thread ID for conversation memory
        # Same ID = same memory
        "thread_id": 1
    }
}


# Start chat loop
while True:

    # Take user input
    user_input = input("User: ")

    # Stop chatbot if user types exit or end
    if user_input in ["exit", "end"]:
        break

    else:

        # Invoke graph with user message
        # config keeps conversation memory
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        # Print only latest AI response
        print("AI: " + result["messages"][-1].content)