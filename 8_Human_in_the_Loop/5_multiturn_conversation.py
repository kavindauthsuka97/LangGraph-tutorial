# Import StateGraph, START node, and add_messages helper
from langgraph.graph import StateGraph, START, add_messages

# Import Command and interrupt for human-in-the-loop workflow
from langgraph.types import Command, interrupt

# Import typing helpers
from typing import TypedDict, Annotated, List

# Import memory checkpointer
from langgraph.checkpoint.memory import MemorySaver

# Import Groq LLM
from langchain_groq import ChatGroq

# Import message classes
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

# Import dotenv loader
from dotenv import load_dotenv

# Import Path to locate .env file
from pathlib import Path

# Import os for environment variables
import os

# Import uuid for unique thread IDs
import uuid


# -------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------

# Get path to .env file (project root folder)
env_path = Path(__file__).resolve().parents[1] / ".env"

# Load .env file
load_dotenv(env_path)

# Check whether GROQ_API_KEY exists
if not os.getenv("GROQ_API_KEY"):

    # Raise error if API key is missing
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )


# -------------------------------
# CREATE LLM
# -------------------------------

# Create Groq model instance
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


# -------------------------------
# DEFINE GRAPH STATE
# -------------------------------

# Define graph state structure
class State(TypedDict):

    # Topic for LinkedIn post
    linkedin_topic: str

    # Store generated posts
    generated_post: Annotated[
        List[AIMessage],
        add_messages
    ]

    # Store human feedback list
    human_feedback: List[str]


# -------------------------------
# MODEL NODE
# -------------------------------

# Function to generate LinkedIn content
def model(state: State):

    # Print log message
    print("[model] Generating content")

    # Get LinkedIn topic from state
    linkedin_topic = state["linkedin_topic"]

    # Get previous feedback or empty list
    feedback = state.get(
        "human_feedback",
        []
    )

    # Create prompt for LLM
    prompt = f"""
LinkedIn Topic: {linkedin_topic}

Human Feedback:
{feedback[-1] if feedback else "No feedback yet"}

Generate a structured and well-written LinkedIn post based on the given topic.

Consider previous human feedback to refine the response.
"""

    # Send prompt to model
    response = llm.invoke([

        # Define assistant behavior
        SystemMessage(
            content="You are an expert LinkedIn content writer."
        ),

        # Send prompt as human message
        HumanMessage(content=prompt)
    ])

    # Extract generated content
    generated_linkedin_post = response.content

    # Print generated post
    print(
        f"[model_node] Generated post:\n"
        f"{generated_linkedin_post}\n"
    )

    # Return updated state
    return {

        # Save generated post
        "generated_post": [
            AIMessage(
                content=generated_linkedin_post
            )
        ]
    }


# -------------------------------
# HUMAN FEEDBACK NODE
# -------------------------------

# Function to collect user feedback
def human_node(state: State):

    # Print status
    print(
        "\n[human_node] Awaiting human feedback..."
    )

    # Get generated posts
    generated_post = state["generated_post"]

    # Pause graph execution
    user_feedback = interrupt({

        # Show latest generated post
        "generated_post":
        generated_post[-1].content,

        # Ask for feedback
        "message":
        "Provide feedback or type 'done' to finish"
    })

    # Print feedback received
    print(
        f"[human_node] "
        f"Received human feedback: "
        f"{user_feedback}"
    )

    # If user types done
    if user_feedback.lower() == "done":

        # Go to final node
        return Command(

            # Update feedback state
            update={
                "human_feedback":
                state["human_feedback"]
                + ["Finalised"]
            },

            # Move to end node
            goto="end_node"
        )

    # Otherwise go back to model
    return Command(

        # Save feedback
        update={
            "human_feedback":
            state["human_feedback"]
            + [user_feedback]
        },

        # Return to model node
        goto="model"
    )


# -------------------------------
# END NODE
# -------------------------------

# Final node
def end_node(state: State):

    # Print completion message
    print("\n[end_node] Process finished")

    # Print final LinkedIn post
    print("\nFinal Generated Post:")

    print(
        state["generated_post"][-1].content
    )

    # Print all feedback
    print("\nFinal Human Feedback:")

    print(state["human_feedback"])

    # Return final state
    return {

        # Return generated posts
        "generated_post":
        state["generated_post"],

        # Return feedback
        "human_feedback":
        state["human_feedback"]
    }


# -------------------------------
# BUILD GRAPH
# -------------------------------

# Create graph
graph = StateGraph(State)

# Add model node
graph.add_node(
    "model",
    model
)

# Add human feedback node
graph.add_node(
    "human_node",
    human_node
)

# Add end node
graph.add_node(
    "end_node",
    end_node)


# -------------------------------
# DEFINE FLOW
# -------------------------------

# Start → model
graph.add_edge(
    START,
    "model"
)

# Model → human feedback
graph.add_edge(
    "model",
    "human_node"
)

# Mark end node
graph.set_finish_point(
    "end_node"
)


# -------------------------------
# COMPILE GRAPH
# -------------------------------

# Create memory saver
checkpointer = MemorySaver()

# Compile graph
app = graph.compile(
    checkpointer=checkpointer
)


# -------------------------------
# THREAD CONFIG
# -------------------------------

# Create unique thread config
thread_config = {
    "configurable": {

        # Unique ID
        "thread_id":
        str(uuid.uuid4())
    }
}


# -------------------------------
# USER INPUT
# -------------------------------

# Ask user for topic
linkedin_topic = input(
    "Enter your LinkedIn topic: "
)


# -------------------------------
# INITIAL STATE
# -------------------------------

# Define initial graph state
initial_state = {

    # Store topic
    "linkedin_topic":
    linkedin_topic,

    # Empty post list
    "generated_post":
    [],

    # Empty feedback list
    "human_feedback":
    []
}


# -------------------------------
# RUN GRAPH
# -------------------------------

# Stream graph execution
for chunk in app.stream(
    initial_state,
    config=thread_config
):

    # Loop through graph nodes
    for node_id, value in chunk.items():

        # If graph pauses
        if node_id == "__interrupt__":

            # Continue asking feedback
            while True:

                # Get user feedback
                user_feedback = input(
                    "Provide feedback "
                    "(or type 'done' "
                    "when finished): "
                )

                # Resume graph
                result = app.invoke(
                    Command(
                        resume=user_feedback
                    ),
                    config=thread_config
                )

                # Stop loop if done
                if (
                    user_feedback.lower()
                    == "done"
                ):
                    break