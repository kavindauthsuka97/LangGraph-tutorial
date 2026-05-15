# Load environment variables from the .env file
from dotenv import load_dotenv

# Load all keys written inside .env
load_dotenv()

# Import TypedDict to define the structure of state
from typing import TypedDict, Annotated

# Import HumanMessage for user messages
from langchain_core.messages import HumanMessage

# Import LangGraph tools
from langgraph.graph import add_messages, StateGraph, END

# Import Groq chat model
from langchain_groq import ChatGroq


# Define the state structure for the graph
class State(TypedDict):

    # Store all conversation messages
    messages: Annotated[list, add_messages]


# Create Groq LLM object
# GROQ_API_KEY will be loaded automatically from .env
llm = ChatGroq(model="llama-3.1-8b-instant")


# Node name constants
GENERATE_POST = "generate_post"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"


# Generate LinkedIn post using LLM
def generate_post(state: State):

    # Send all messages to LLM and return AI response
    return {
        "messages": [llm.invoke(state["messages"])]
    }


# Ask user whether to approve or improve the post
def get_review_decision(state: State):

    # Get latest AI-generated post
    post_content = state["messages"][-1].content

    # Display current post
    print("\n📢 Current LinkedIn Post:\n")
    print(post_content)
    print("\n")

    # Ask user for approval
    decision = input("Post to LinkedIn? (yes/no): ")

    # If approved, go to POST node
    if decision.lower() == "yes":
        return POST

    # Otherwise, go to feedback node
    return COLLECT_FEEDBACK


# Final posting step
def post(state: State):

    # Get final approved post
    final_post = state["messages"][-1].content

    # Display final post
    print("\n📢 Final LinkedIn Post:\n")
    print(final_post)

    # Simulate successful posting
    print("\n✅ Post has been approved and is now live on LinkedIn!")


# Collect feedback from user
def collect_feedback(state: State):

    # Ask user how to improve the post
    feedback = input("How can I improve this post? ")

    # Add feedback as a new human message
    return {
        "messages": [HumanMessage(content=feedback)]
    }


# Create a LangGraph workflow
graph = StateGraph(State)


# Add post generation node
graph.add_node(GENERATE_POST, generate_post)

# Add feedback collection node
graph.add_node(COLLECT_FEEDBACK, collect_feedback)

# Add final posting node
graph.add_node(POST, post)


# Set first node to run
graph.set_entry_point(GENERATE_POST)


# After generating post, ask user whether to post or revise
graph.add_conditional_edges(
    GENERATE_POST,
    get_review_decision
)


# If user gives feedback, generate post again
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)


# After posting, end the workflow
graph.add_edge(POST, END)


# Compile the graph into a runnable app
app = graph.compile()


# Start the workflow with initial user request
response = app.invoke({
    "messages": [
        HumanMessage(
            content="Write me a LinkedIn post on AI Agents taking over content creation"
        )
    ]
})


# Print final response state
print(response)