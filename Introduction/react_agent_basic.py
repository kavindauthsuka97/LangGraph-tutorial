# Import ChatGroq to use a Groq model
from langchain_groq import ChatGroq

# Import function to load environment variables from .env
from dotenv import load_dotenv

# Import the new agent builder
from langchain.agents import create_agent

# Import tool decorator
from langchain.tools import tool

# Import Tavily search tool for web search
from langchain_community.tools import TavilySearchResults

# Import datetime to get current system time
import datetime


# Load API keys from the .env file
load_dotenv()


# Create the Groq chat model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Current Groq model
    temperature=0                     # Better for accurate reasoning
)


# Create the Tavily search tool
search_tool = TavilySearchResults(search_depth="basic")


# Create a custom tool to return the current system time
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Returns the current date and time in the specified format."""

    # Get the current local date and time
    current_time = datetime.datetime.now()

    # Format the date and time as text
    formatted_time = current_time.strftime(format)

    # Return the formatted result
    return formatted_time


# Put all tools into a list
tools = [search_tool, get_system_time]


# Build the agent with the model and tools
agent = create_agent(
    model=llm,    # Groq model
    tools=tools   # Tools the agent can use
)


# Run the agent with a user message
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "When was SpaceX's last launch and how many days ago was that from this instant"
            }
        ]
    }
)


# Print the final agent answer
print(result["messages"][-1].content)