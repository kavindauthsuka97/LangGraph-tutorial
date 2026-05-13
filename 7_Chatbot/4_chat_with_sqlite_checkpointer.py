# Import TypedDict for defining state structure
# Annotated adds extra functionality to types
from typing import TypedDict, Annotated

# Import LangGraph tools
# add_messages manages message history
# StateGraph creates graph workflow
# END marks graph ending
from langgraph.graph import add_messages, StateGraph, END

# Import Groq LLM wrapper
from langchain_groq import ChatGroq

# Import HumanMessage for user input
from langchain_core.messages import HumanMessage

# Import dotenv to load environment variables
from dotenv import load_dotenv

# Import SQLite-based memory saver
from langgraph.checkpoint.sqlite import SqliteSaver

# Import sqlite3 to create SQLite database connection
import sqlite3


# Load environment variables (.env file)
load_dotenv()


# Create SQLite database connection
# "checkpoint.sqlite" file stores memory data
# check_same_thread=False allows multi-thread access
sqlite_conn = sqlite3.connect(
    "checkpoint.sqlite",
    check_same_thread=False
)


# Create memory object using SQLite
# Chat memory will be saved permanently in database
memory = SqliteSaver(sqlite_conn)


# Create Groq LLM model
llm = ChatGroq(model="llama-3.1-8b-instant")


# Define chatbot state structure
class BasicChatState(TypedDict):

    # Store conversation messages
    # add_messages automatically updates message list
    messages: Annotated[list, add_messages]


# Chatbot node function
def chatbot(state: BasicChatState):

    # Send messages to LLM
    # Return AI response
    return {
       "messages": [llm.invoke(state["messages"])]
    }


# Create graph
graph = StateGraph(BasicChatState)


# Add chatbot node
graph.add_node("chatbot", chatbot)


# End graph after chatbot runs
graph.add_edge("chatbot", END)


# Set chatbot as entry point
graph.set_entry_point("chatbot")


# Compile graph with SQLite memory
# Saves chat history in database
app = graph.compile(checkpointer=memory)


# Configuration settings
config = {
    "configurable": {

        # Unique thread ID for conversation memory
        # Same thread_id = same conversation history
        "thread_id": 1
    }
}


# Start chatbot loop
while True:

    # Get user input
    user_input = input("User: ")

    # Stop chatbot if user types exit or end
    if user_input in ["exit", "end"]:
        break

    else:

        # Send user message to graph
        # config keeps memory linked to thread_id
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        # Print latest AI response only
        print("AI: " + result["messages"][-1].content)