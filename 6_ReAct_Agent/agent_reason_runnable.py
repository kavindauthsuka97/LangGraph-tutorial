import datetime  # Imports datetime module
from pathlib import Path  # Helps create reliable file paths

from dotenv import load_dotenv  # Loads environment variables from .env file

BASE_DIR = Path(__file__).resolve().parents[1]  # Points to the LangGraph folder
load_dotenv(BASE_DIR / ".env")  # Loads LangGraph/.env file

from langchain_groq import ChatGroq  # Imports Groq chat model
from langchain_core.tools import tool  # Imports tool decorator
from langchain_classic.agents import create_react_agent  # Imports ReAct agent creator
from langchain_classic import hub  # Imports LangChain Hub
from langchain_community.tools import TavilySearchResults  # Imports Tavily search tool

llm = ChatGroq(model="llama-3.1-8b-instant")  # recommended  # Creates a Groq LLM instance


@tool  # Turns this function into a LangChain tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):  # Defines a tool to get current time
    """Returns the current date and time in the specified format"""  # Tool description for the agent
    current_time = datetime.datetime.now()  # Gets the current date and time
    formatted_time = current_time.strftime(format)  # Formats the time as a string
    return formatted_time  # Returns the formatted time


search_tool = TavilySearchResults(search_depth="basic")  # Creates a Tavily search tool

react_prompt = hub.pull("hwchase17/react")  # Pulls the standard ReAct prompt

tools = [get_system_time, search_tool]  # Stores all tools in a list

react_agent_runnable = create_react_agent(  # Creates the ReAct agent
    tools=tools,  # Passes the tools to the agent
    llm=llm,  # Passes the Groq model to the agent
    prompt=react_prompt  # Passes the ReAct prompt
)